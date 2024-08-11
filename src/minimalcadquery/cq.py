"""
    Copyright (C) 2011-2015  Parametric Products Intellectual Holdings, LLC

    This file is part of CadQuery.

    CadQuery is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    CadQuery is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; If not, see <http://www.gnu.org/licenses/>
"""

from copy import copy
from typing import (
    TypeVar,
    Tuple,
    Optional,
    Iterable,
    Callable,
    List,
    Dict,
)
from typing_extensions import Literal

from .occ_impl.geom import Vector, Plane, Location
from .occ_impl.shapes import (
    Shape,
    Edge,
    Wire,
    Face,
    Solid,
    Compound,
    wiresToFaces,
)

import logging
logger = logging.getLogger(__name__)

CQObject = (Vector | Location | Shape)
VectorLike = (Tuple[float, float] | Tuple[float, float, float] | Vector)
CombineMode = (bool | Literal["cut", "a", "s"])  # a : additive, s: subtractive
TOL = 1e-6

T = TypeVar("T", bound="Workplane")
"""A type variable used to make the return type of a method the same as the
type of `self` or another argument.

This is useful when you want to allow a class to derive from
:class:`.Workplane`, and you want a (fluent) method in the derived class to
return an instance of the derived class, rather than of :class:`.Workplane`.
"""

class CQContext(object):
    """
    A shared context for modeling.

    All objects in the same CQ chain share a reference to this same object instance
    which allows for shared state when needed.
    """

    pendingWires: List[Wire]
    pendingEdges: List[Edge]
    firstPoint: Optional[Vector]
    tolerance: float
    tags: Dict[str, "Workplane"]

    def __init__(self):
        self.pendingWires = (
            []
        )  # a list of wires that have been created and need to be extruded
        # a list of created pending edges that need to be joined into wires
        self.pendingEdges = []
        # a reference to the first point for a set of edges.
        # Used to determine how to behave when close() is called
        self.firstPoint = None
        self.tolerance = 0.0001  # user specified tolerance
        self.tags = {}

    def popPendingWires(self, errorOnEmpty: bool = True) -> List[Wire]:
        """
        Get and clear pending wires.

        :raises ValueError: if errorOnEmpty is True and no wires are present.
        """
        # logger.info("In CQContext's popPendingWires() function......")        
        if errorOnEmpty and not self.pendingWires:
            raise ValueError("No pending wires present")
        out = self.pendingWires
        self.pendingWires = []
        return out

class Workplane(object):
    """
    Defines a coordinate system in space, in which 2D coordinates can be used.

    :param plane: the plane in which the workplane will be done
    :type plane: a Plane object, or a string in (XY|YZ|XZ|front|back|top|bottom|left|right)
    :param origin: the desired origin of the new workplane
    :type origin: a 3-tuple in global coordinates, or None to default to the origin
    :param obj: an object to use initially for the stack
    :type obj: a CAD primitive, or None to use the centerpoint of the plane as the initial
        stack value.
    :raises: ValueError if the provided plane is not a plane, a valid named workplane
    :return: A Workplane object, with coordinate system matching the supplied plane.

    The most common use is::

        s = Workplane("XY")

    After creation, the stack contains a single point, the origin of the underlying plane,
    and the *current point* is on the origin.

    .. note::
        You can also create workplanes on the surface of existing faces using
        :meth:`workplane`
    """

    objects: List[CQObject]
    ctx: CQContext
    parent: Optional["Workplane"]
    plane: Plane

    _tag: Optional[str]

    def __init__(self, inPlane="XY", origin=(0, 0, 0), obj=None):
        """
        make a workplane from a particular plane

        :param inPlane: the plane in which the workplane will be done
        :type inPlane: a Plane object, or a string in (XY|YZ|XZ|front|back|top|bottom|left|right)
        :param origin: the desired origin of the new workplane
        :type origin: a 3-tuple in global coordinates, or None to default to the origin
        :param obj: an object to use initially for the stack
        :type obj: a CAD primitive, or None to use the centerpoint of the plane as the initial
            stack value.
        :raises: ValueError if the provided plane is not a plane, or one of XY|YZ|XZ
        :return: A Workplane object, with coordinate system matching the supplied plane.

        The most common use is::

            s = Workplane("XY")

        After creation, the stack contains a single point, the origin of the underlying plane, and
        the *current point* is on the origin.
        """
        logger.info("In Workplane's __init__(self, inPlane=XY, origin=(0, 0, 0), obj=None): function......")

        if isinstance(inPlane, Plane):
            tmpPlane = inPlane
        elif isinstance(inPlane, str):
            tmpPlane = Plane.named(inPlane, origin)
        elif isinstance(inPlane, (Vector, Location, Shape)):
            obj = inPlane
            tmpPlane = Plane.named("XY", origin)
        else:
            raise ValueError(
                "Provided value {} is not a valid work plane".format(inPlane)
            )

        self.plane = tmpPlane
        # Changed so that workplane has the center as the first item on the stack
        if obj:
            self.objects = [obj]
        else:
            self.objects = []

        self.parent = None
        self.ctx = CQContext()
        self._tag = None

    def vals(self) -> List[CQObject]:
        """
        get the values in the current list

        :rtype: list of occ_impl objects
        :returns: the values of the objects on the stack.

        Contrast with :meth:`all`, which returns CQ objects for all of the items on the stack
        """
        # logger.info("In Workplane's vals() function......")        

        return self.objects

    def _findType(self, types, searchStack=True, searchParents=True):
        # logger.info("In Workplane's _findType() function......")        

        if searchStack:
            rv = [s for s in self.objects if isinstance(s, types)]
            if rv and types == (Solid, Compound):
                return Compound.makeCompound(rv)
            elif rv:
                return rv[0]

        if searchParents and self.parent is not None:
            return self.parent._findType(types, searchStack=True, searchParents=True)

        return None

    def newObject(self: T, objlist: Iterable[CQObject]) -> T:
        """
        Create a new workplane object from this one.

        Overrides CQ.newObject, and should be used by extensions, plugins, and
        subclasses to create new objects.

        :param objlist: new objects to put on the stack
        :type objlist: a list of CAD primitives
        :return: a new Workplane object with the current workplane as a parent.
        """
        # logger.info("In Workplane's newObject() function......")        

        # copy the current state to the new object
        ns = self.__class__()
        ns.plane = copy(self.plane)
        ns.parent = self
        ns.objects = list(objlist)
        ns.ctx = self.ctx
        return ns

    def _addPendingWire(self, wire: Wire) -> None:
        """
        Queue a Wire for later extrusion

        Internal Processing Note.  In OCCT, edges-->wires-->faces-->solids.

        but users do not normally care about these distinctions.  Users 'think' in terms
        of edges, and solids.

        CadQuery tracks edges as they are drawn, and automatically combines them into wires
        when the user does an operation that needs it.

        Similarly, CadQuery tracks pending wires, and automatically combines them into faces
        when necessary to make a solid.
        """
        # logger.info("In Workplane's _addPendingWire() function......")        

        self.ctx.pendingWires.append(wire)

    def eachpoint(
        self: T,
        callback: Callable[[Location], Shape],
        useLocalCoordinates: bool = False,
        combine: CombineMode = False,
        clean: bool = True,
    ) -> T:
        """
        Same as each(), except each item on the stack is converted into a point before it
        is passed into the callback function.

        :return: CadQuery object which contains a list of  vectors (points ) on its stack.

        :param useLocalCoordinates: should points be in local or global coordinates
        :param combine: True or "a" to combine the resulting solid with parent solids if found,
            "cut" or "s" to remove the resulting solid from the parent solids if found.
            False to keep the resulting solid separated from the parent solids.
        :param clean: call :meth:`clean` afterwards to have a clean shape


        The resulting object has a point on the stack for each object on the original stack.
        Vertices and points remain a point.  Faces, Wires, Solids, Edges, and Shells are converted
        to a point by using their center of mass.

        If the stack has zero length, a single point is returned, which is the center of the current
        workplane/coordinate system
        """
        # logger.info("In Workplane's eachpoint() function......")        

        # convert stack to a list of points
        pnts = []
        plane = self.plane
        loc = self.plane.location

        if len(self.objects) == 0:
            # nothing on the stack. here, we'll assume we should operate with the
            # origin as the context point
            pnts.append(Location())
        else:
            for o in self.objects:
                if isinstance(o, (Vector, Shape)):
                    pnts.append(loc.inverse * Location(plane, o.Center()))
                else:
                    pnts.append(o)

        if useLocalCoordinates:
            res = [callback(p).move(loc) for p in pnts]
        else:
            res = [callback(p * loc) for p in pnts]

        for r in res:
            if isinstance(r, Wire) and not r.forConstruction:
                self._addPendingWire(r)

        return self._combineWithBase(res, combine, clean)

    def rect(
        self: T,
        xLen: float,
        yLen: float,
        centered: (bool | Tuple[bool, bool]) = True,
        forConstruction: bool = False,
    ) -> T:
        """
        Make a rectangle for each item on the stack.

        :param xLen: length in the x direction (in workplane coordinates)
        :param yLen: length in the y direction (in workplane coordinates)
        :param centered: If True, the rectangle will be centered around the reference
          point. If False, the corner of the rectangle will be on the reference point and
          it will extend in the positive x and y directions. Can also use a 2-tuple to
          specify centering along each axis.
        :param forConstruction: should the new wires be reference geometry only?
        :type forConstruction: true if the wires are for reference, false if they are creating part
            geometry
        :return: a new CQ object with the created wires on the stack

        A common use case is to use a for-construction rectangle to define the centers of a hole
        pattern::

            s = Workplane().rect(4.0, 4.0, forConstruction=True).vertices().circle(0.25)

        Creates 4 circles at the corners of a square centered on the origin.

        Negative values for xLen and yLen are permitted, although they only have an effect when
        centered is False.

        Future Enhancements:
            * project points not in the workplane plane onto the workplane plane
        """
        # logger.info("In Workplane's rect() function......")        

        if isinstance(centered, bool):
            centered = (centered, centered)

        offset = Vector()
        if not centered[0]:
            offset += Vector(xLen / 2, 0, 0)
        if not centered[1]:
            offset += Vector(0, yLen / 2, 0)

        points = [
            Vector(xLen / -2.0, yLen / -2.0, 0),
            Vector(xLen / 2.0, yLen / -2.0, 0),
            Vector(xLen / 2.0, yLen / 2.0, 0),
            Vector(xLen / -2.0, yLen / 2.0, 0),
        ]

        points = [x + offset for x in points]

        # close the wire
        points.append(points[0])

        w = Wire.makePolygon(points, forConstruction)

        return self.eachpoint(lambda loc: w.moved(loc), True)

    def extrude(
        self: T,
        until: (float | Literal["next", "last"] | Face),
        combine: CombineMode = True,
        clean: bool = True,
        both: bool = False,
        taper: Optional[float] = None,
    ) -> T:
        """
        Use all un-extruded wires in the parent chain to create a prismatic solid.

        :param until: The distance to extrude, normal to the workplane plane. When a float is
            passed, the extrusion extends this far and a negative value is in the opposite direction
            to the normal of the plane. The string "next" extrudes until the next face orthogonal to
            the wire normal. "last" extrudes to the last face. If a object of type Face is passed then
            the extrusion will extend until this face. **Note that the Workplane must contain a Solid for extruding to a given face.**
        :param combine: True or "a" to combine the resulting solid with parent solids if found,
            "cut" or "s" to remove the resulting solid from the parent solids if found.
            False to keep the resulting solid separated from the parent solids.
        :param clean: call :meth:`clean` afterwards to have a clean shape
        :param both: extrude in both directions symmetrically
        :param taper: angle for optional tapered extrusion
        :return: a CQ object with the resulting solid selected.

        The returned object is always a CQ object, and depends on whether combine is True, and
        whether a context solid is already defined:

        *  if combine is False, the new value is pushed onto the stack. Note that when extruding
            until a specified face, combine can not be False
        *  if combine is true, the value is combined with the context solid if it exists,
            and the resulting solid becomes the new context solid.
        """
        # logger.info("In Workplane's extrude() function......")        

        # If subtractive mode is requested, use cutBlind
        if combine in ("cut", "s"):
            return self.cutBlind(until, clean, both, taper)

        # Handle `until` multiple values
        elif until in ("next", "last") and combine in (True, "a"):
            if until == "next":
                faceIndex = 0
            elif until == "last":
                faceIndex = -1

            r = self._extrude(None, both=both, taper=taper, upToFace=faceIndex)

        elif isinstance(until, Face) and combine:
            r = self._extrude(None, both=both, taper=taper, upToFace=until)

        elif isinstance(until, (int, float)):
            r = self._extrude(until, both=both, taper=taper, upToFace=None)

        elif isinstance(until, (str, Face)) and combine is False:
            raise ValueError(
                "`combine` can't be set to False when extruding until a face"
            )

        else:
            raise ValueError(
                f"Do not know how to handle until argument of type {type(until)}"
            )

        return self._combineWithBase(r, combine, clean)

    def _combineWithBase(
        self: T,
        obj: (Shape | Iterable[Shape]),
        mode: CombineMode = True,
        clean: bool = False,
    ) -> T:
        """
        Combines the provided object with the base solid, if one can be found.

        :param obj: The object to be combined with the context solid
        :param mode: The mode to combine with the base solid (True, False, "cut", "a" or "s")
        :return: a new object that represents the result of combining the base object with obj,
           or obj if one could not be found
        """
        # logger.info("In Workplane's _combineWithBase() function......")        

        if mode:
            # since we are going to do something convert the iterable if needed
            if not isinstance(obj, Shape):
                obj = Compound.makeCompound(obj)

            # dispatch on the mode
            if mode in ("cut", "s"):
                newS = self._cutFromBase(obj)
            elif mode in (True, "a"):
                newS = self._fuseWithBase(obj)

        else:
            # do not combine branch
            newS = self.newObject(obj if not isinstance(obj, Shape) else [obj])

        if clean:
            # NB: not calling self.clean() to not pollute the parents
            newS.objects = [
                obj.clean() if isinstance(obj, Shape) else obj for obj in newS.objects
            ]

        return newS

    def _fuseWithBase(self: T, obj: Shape) -> T:
        """
        Fuse the provided object with the base solid, if one can be found.

        :param obj:
        :return: a new object that represents the result of combining the base object with obj,
           or obj if one could not be found
        """
        # logger.info("In Workplane's _fuseWithBase() function......")        

        baseSolid = self._findType(
            (Solid, Compound), searchStack=True, searchParents=True
        )
        r = obj
        if baseSolid is not None:
            r = baseSolid.fuse(obj)
        elif isinstance(obj, Compound):
            r = obj.fuse()
        return self.newObject([r])

    def _getFaces(self) -> List[Face]:
        """
        Convert pending wires or sketches to faces for subsequent operation
        """
        # logger.info("In Workplane's _getFaces() function......")        

        rv: List[Face] = []

        if not rv:
            rv.extend(wiresToFaces(self.ctx.popPendingWires()))

        return rv

    def _extrude(
        self,
        distance: Optional[float] = None,
        both: bool = False,
        taper: Optional[float] = None,
        upToFace: Optional[(int | Face)] = None,
        additive: bool = True,
    ) -> (Solid | Compound):
        """
        Make a prismatic solid from the existing set of pending wires.

        :param distance: distance to extrude
        :param both: extrude in both directions symmetrically
        :param upToFace: if specified, extrude up to a face: 0 for the next, -1 for the last face
        :param additive: specify if extruding or cutting, required param for uptoface algorithm

        :return: OCCT solid(s), suitable for boolean operations.

        This method is a utility method, primarily for plugin and internal use.
        It is the basis for cutBlind, extrude, cutThruAll, and all similar methods.
        """
        # logger.info("In Workplane's _extrude() function......")        

        def getFacesList(face, eDir, direction, both=False):
            """
            Utility function to make the code further below more clean and tidy
            Performs some test and raise appropriate error when no Faces are found for extrusion
            """
            # logger.info("In Workplane's getFacesList() function......")        

            facesList = self.findSolid().facesIntersectedByLine(
                face.Center(), eDir, direction=direction
            )
            if len(facesList) == 0 and both:
                raise ValueError(
                    "Couldn't find a face to extrude/cut to for at least one of the two required directions of extrusion/cut."
                )

            if len(facesList) == 0:
                # if we don't find faces in the workplane normal direction we try the other
                # direction (as the user might have created a workplane with wrong orientation)
                facesList = self.findSolid().facesIntersectedByLine(
                    face.Center(), eDir.multiply(-1.0), direction=direction
                )
                if len(facesList) == 0:
                    raise ValueError(
                        "Couldn't find a face to extrude/cut to. Check your workplane orientation."
                    )
            return facesList

        # process sketches or pending wires
        faces = self._getFaces()

        # check for nested geometry and tapered extrusion
        for face in faces:
            if taper and face.innerWires():
                raise ValueError("Inner wires not allowed with tapered extrusion")

        # compute extrusion vector and extrude
        if upToFace is not None:
            eDir = self.plane.zDir
        elif distance is not None:
            eDir = self.plane.zDir.multiply(distance)

        direction = "AlongAxis" if additive else "Opposite"
        taper = 0.0 if taper is None else taper

        if upToFace is not None:
            res = self.findSolid()
            for face in faces:
                if isinstance(upToFace, int):
                    facesList = getFacesList(face, eDir, direction, both=both)
                    if (
                        res.isInside(face.outerWire().Center())
                        and additive
                        and upToFace == 0
                    ):
                        upToFace = 1  # extrude until next face outside the solid

                    limitFace = facesList[upToFace]
                else:
                    limitFace = upToFace

                res = res.dprism(
                    None, [face], taper=taper, upToFace=limitFace, additive=additive,
                )

                if both:
                    facesList2 = getFacesList(
                        face, eDir.multiply(-1.0), direction, both=both
                    )
                    limitFace2 = facesList2[upToFace]
                    res = res.dprism(
                        None,
                        [face],
                        taper=taper,
                        upToFace=limitFace2,
                        additive=additive,
                    )
        else:
            toFuse = []
            for face in faces:
                s1 = Solid.extrudeLinear(face, eDir, taper=taper)

                if both:
                    s2 = Solid.extrudeLinear(face, eDir.multiply(-1.0), taper=taper)
                    toFuse.append(s1.fuse(s2, glue=True))
                else:
                    toFuse.append(s1)

            res = Compound.makeCompound(toFuse)

        return res

    def box(
        self: T,
        length: float,
        width: float,
        height: float,
        centered: (bool | Tuple[bool, bool, bool]) = True,
        combine: CombineMode = True,
        clean: bool = True,
    ) -> T:
        """
        Return a 3d box with specified dimensions for each object on the stack.

        :param length: box size in X direction
        :param width: box size in Y direction
        :param height: box size in Z direction
        :param centered: If True, the box will be centered around the reference point.
            If False, the corner of the box will be on the reference point and it will
            extend in the positive x, y and z directions. Can also use a 3-tuple to
            specify centering along each axis.
        :param combine: should the results be combined with other solids on the stack
            (and each other)?
        :param clean: call :meth:`clean` afterwards to have a clean shape

        One box is created for each item on the current stack. If no items are on the stack, one box
        using the current workplane center is created.

        If combine is true, the result will be a single object on the stack. If a solid was found
        in the chain, the result is that solid with all boxes produced fused onto it otherwise, the
        result is the combination of all the produced boxes.

        If combine is false, the result will be a list of the boxes produced.

        Most often boxes form the basis for a part::

            # make a single box with lower left corner at origin
            s = Workplane().box(1, 2, 3, centered=False)

        But sometimes it is useful to create an array of them::

            # create 4 small square bumps on a larger base plate:
            s = (
                Workplane()
                .box(4, 4, 0.5)
                .faces(">Z")
                .workplane()
                .rect(3, 3, forConstruction=True)
                .vertices()
                .box(0.25, 0.25, 0.25, combine=True)
            )

        """
        # logger.info("In Workplane's box() function......")        

        if isinstance(centered, bool):
            centered = (centered, centered, centered)

        offset = Vector()
        if centered[0]:
            offset += Vector(-length / 2, 0, 0)
        if centered[1]:
            offset += Vector(0, -width / 2, 0)
        if centered[2]:
            offset += Vector(0, 0, -height / 2)

        box = Solid.makeBox(length, width, height, offset)

        return self.eachpoint(lambda loc: box.moved(loc), True, combine, clean)

    def clean(self: T) -> T:
        """
        Cleans the current solid by removing unwanted edges from the
        faces.

        Normally you don't have to call this function. It is
        automatically called after each related operation. You can
        disable this behavior with `clean=False` parameter if method
        has any. In some cases this can improve performance
        drastically but is generally dis-advised since it may break
        some operations such as fillet.

        Note that in some cases where lots of solid operations are
        chained, `clean()` may actually improve performance since
        the shape is 'simplified' at each step and thus next operation
        is easier.

        Also note that, due to limitation of the underlying engine,
        `clean` may fail to produce a clean output in some cases such as
        spherical faces.
        """
        # logger.info("In Workplane's clean() function......")        

        cleanObjects = [
            obj.clean() if isinstance(obj, Shape) else obj for obj in self.objects
        ]

        return self.newObject(cleanObjects)

# alias for backward compatibility
CQ = Workplane
