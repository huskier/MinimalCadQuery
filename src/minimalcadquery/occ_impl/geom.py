import math

from typing import Tuple, Type, Optional

from OCP.gp import (
    gp_Vec,
    gp_Ax1,
    gp_Ax3,
    gp_Pnt,
    gp_Dir,
    gp_Trsf,
    gp_GTrsf,
    gp_XYZ,
)

from OCP.TopLoc import TopLoc_Location

import logging
logger = logging.getLogger(__name__)

Real = (float | int)

TOL = 1e-2

VectorLike = ("Vector" | Tuple[Real, Real] | Tuple[Real, Real, Real])


class Vector(object):
    """Create a 3-dimensional vector

    :param args: a 3D vector, with x-y-z parts.

    you can either provide:
        * nothing (in which case the null vector is return)
        * a gp_Vec
        * a vector ( in which case it is copied )
        * a 3-tuple
        * a 2-tuple (z assumed to be 0)
        * three float values: x, y, and z
        * two float values: x,y
    """
    _wrapped: gp_Vec

    def __init__(self, *args):
        # print("In Vector's __init__(self, *args): function......") 
        logger.info("In Vector's __init__(self, *args): function......")
        if len(args) == 3:
            fV = gp_Vec(*args)
        elif len(args) == 2:
            fV = gp_Vec(*args, 0)
        elif len(args) == 1:
            if isinstance(args[0], Vector):
                fV = gp_Vec(args[0].wrapped.XYZ())
            elif isinstance(args[0], (tuple, list)):
                arg = args[0]
                if len(arg) == 3:
                    fV = gp_Vec(*arg)
                elif len(arg) == 2:
                    fV = gp_Vec(*arg, 0)
            elif isinstance(args[0], (gp_Vec, gp_Pnt, gp_Dir)):
                fV = gp_Vec(args[0].XYZ())
            elif isinstance(args[0], gp_XYZ):
                fV = gp_Vec(args[0])
            else:
                raise TypeError("Expected three floats, OCC gp_, or 3-tuple")
        elif len(args) == 0:
            fV = gp_Vec(0, 0, 0)
        else:
            raise TypeError("Expected three floats, OCC gp_, or 3-tuple")

        self._wrapped = fV

    @property
    def x(self) -> float:
        # print("In Vector's x(self) -> float: function......")      
        return self.wrapped.X()

    @property
    def y(self) -> float:
        # print("In Vector's y(self) -> float: function......")      

        return self.wrapped.Y()

    @property
    def z(self) -> float:
        # print("In Vector's z(self) -> float: function......")      

        return self.wrapped.Z()

    @property
    def Length(self) -> float:
        # print("In Vector's Length(self) -> float: function......")      

        return self.wrapped.Magnitude()

    @property
    def wrapped(self) -> gp_Vec:
        # print("In Vector's wrapped(self) -> gp_Vec: function......")      

        return self._wrapped

    def toTuple(self) -> Tuple[float, float, float]:
        # print("In Vector's toTuple(self) -> Tuple[float, float, float]: function......")      

        return (self.x, self.y, self.z)

    def cross(self, v: "Vector") -> "Vector":
        # print("In Vector's cross(self, v: Vector) -> Vector: function......")      

        return Vector(self.wrapped.Crossed(v.wrapped))

    def add(self, v: "Vector") -> "Vector":
        # print("In Vector's add(self, v: Vector) -> Vector: function......")      

        return Vector(self.wrapped.Added(v.wrapped))

    def __add__(self, v: "Vector") -> "Vector":
        # print("In Vector's __add__(self, v: Vector) -> Vector: function......")      

        return self.add(v)


    def multiply(self, scale: float) -> "Vector":
        """Return a copy multiplied by the provided scalar"""
        # print("In Vector's multiply(self, scale: float) -> Vector: function......")      

        return Vector(self.wrapped.Multiplied(scale))

    def normalized(self) -> "Vector":
        """Return a normalized version of this vector"""
        # print("In Vector's normalized(self) -> Vector: function......")      

        return Vector(self.wrapped.Normalized())

    def toPnt(self) -> gp_Pnt:
        # print("In Vector's toPnt(self) -> gp_Pnt: function......")      

        return gp_Pnt(self.wrapped.XYZ())

    def toDir(self) -> gp_Dir:
        # print("In Vector's toDir(self) -> gp_Dir: function......")      

        return gp_Dir(self.wrapped.XYZ())

class Matrix:
    pass

class Plane(object):
    """A 2D coordinate system in space

    A 2D coordinate system in space, with the x-y axes on the plane, and a
    particular point as the origin.

    A plane allows the use of 2D coordinates, which are later converted to
    global, 3d coordinates when the operations are complete.

    Frequently, it is not necessary to create work planes, as they can be
    created automatically from faces.
    """

    xDir: Vector
    yDir: Vector
    zDir: Vector
    _origin: Vector

    lcs: gp_Ax3
    rG: Matrix
    fG: Matrix

    # equality tolerances
    _eq_tolerance_origin = 1e-6
    _eq_tolerance_dot = 1e-6

    @classmethod
    def named(cls: Type["Plane"], stdName: str, origin=(0, 0, 0)) -> "Plane":
        """Create a predefined Plane based on the conventional names.

        :param stdName: one of (XY|YZ|ZX|XZ|YX|ZY|front|back|left|right|top|bottom)
        :type stdName: string
        :param origin: the desired origin, specified in global coordinates
        :type origin: 3-tuple of the origin of the new plane, in global coordinates.

        Available named planes are as follows. Direction references refer to
        the global directions.

        =========== ======= ======= ======
        Name        xDir    yDir    zDir
        =========== ======= ======= ======
        XY          +x      +y      +z
        YZ          +y      +z      +x
        ZX          +z      +x      +y
        XZ          +x      +z      -y
        YX          +y      +x      -z
        ZY          +z      +y      -x
        front       +x      +y      +z
        back        -x      +y      -z
        left        +z      +y      -x
        right       -z      +y      +x
        top         +x      -z      +y
        bottom      +x      +z      -y
        =========== ======= ======= ======
        """
        # print("In Plane's named() function......")      

        namedPlanes = {
            # origin, xDir, normal
            "XY": Plane(origin, (1, 0, 0), (0, 0, 1)),
            "YZ": Plane(origin, (0, 1, 0), (1, 0, 0)),
            "ZX": Plane(origin, (0, 0, 1), (0, 1, 0)),
            "XZ": Plane(origin, (1, 0, 0), (0, -1, 0)),
            "YX": Plane(origin, (0, 1, 0), (0, 0, -1)),
            "ZY": Plane(origin, (0, 0, 1), (-1, 0, 0)),
            "front": Plane(origin, (1, 0, 0), (0, 0, 1)),
            "back": Plane(origin, (-1, 0, 0), (0, 0, -1)),
            "left": Plane(origin, (0, 0, 1), (-1, 0, 0)),
            "right": Plane(origin, (0, 0, -1), (1, 0, 0)),
            "top": Plane(origin, (1, 0, 0), (0, 1, 0)),
            "bottom": Plane(origin, (1, 0, 0), (0, -1, 0)),
        }

        try:
            return namedPlanes[stdName]
        except KeyError:
            raise ValueError("Supported names are {}".format(list(namedPlanes.keys())))

    def __init__(
        self,
        origin: (Tuple[float, float, float] | Vector),
        xDir: Optional[(Tuple[float, float, float] | Vector)] = None,
        normal: (Tuple[float, float, float] | Vector) = (0, 0, 1),
    ):
        """
        Create a Plane with an arbitrary orientation

        :param origin: the origin in global coordinates
        :param xDir: an optional vector representing the xDirection.
        :param normal: the normal direction for the plane
        :raises ValueError: if the specified xDir is not orthogonal to the provided normal
        """
        # print("In Plane's __init__() function......")      

        zDir = Vector(normal)
        if zDir.Length == 0.0:
            raise ValueError("normal should be non null")

        self.zDir = zDir.normalized()

        if xDir is None:
            ax3 = gp_Ax3(Vector(origin).toPnt(), Vector(normal).toDir())
            xDir = Vector(ax3.XDirection())
        else:
            xDir = Vector(xDir)
            if xDir.Length == 0.0:
                raise ValueError("xDir should be non null")
        self._setPlaneDir(xDir)
        self.origin = Vector(origin)

    @property
    def origin(self) -> Vector:
        # print("In Plane's origin() function......")      

        return self._origin

    @origin.setter
    def origin(self, value):
        # print("In Plane's origin() setter function......")      

        self._origin = Vector(value)
        self._calcTransforms()

    def _setPlaneDir(self, xDir):
        """Set the vectors parallel to the plane, i.e. xDir and yDir"""
        # print("In Plane's _setPlaneDir() function......")      

        xDir = Vector(xDir)
        self.xDir = xDir.normalized()
        self.yDir = self.zDir.cross(self.xDir).normalized()

    def _calcTransforms(self):
        """Computes transformation matrices to convert between coordinates

        Computes transformation matrices to convert between local and global
        coordinates.
        """

        # print("In Plane's _calcTransforms() function......")      

        # r is the forward transformation matrix from world to local coordinates
        # ok i will be really honest, i cannot understand exactly why this works
        # something bout the order of the translation and the rotation.
        # the double-inverting is strange, and I don't understand it.
        forward = Matrix()
        inverse = Matrix()

        forwardT = gp_Trsf()
        inverseT = gp_Trsf()

        global_coord_system = gp_Ax3()
        local_coord_system = gp_Ax3(
            gp_Pnt(*self.origin.toTuple()),
            gp_Dir(*self.zDir.toTuple()),
            gp_Dir(*self.xDir.toTuple()),
        )

        forwardT.SetTransformation(global_coord_system, local_coord_system)
        forward.wrapped = gp_GTrsf(forwardT)

        inverseT.SetTransformation(local_coord_system, global_coord_system)
        inverse.wrapped = gp_GTrsf(inverseT)

        self.lcs = local_coord_system
        self.rG = inverse
        self.fG = forward
    
    @property
    def location(self) -> "Location":
        # print("In Plane's location() function......")      

        return Location(self)
    
class Location(object):
    """Location in 3D space. Depending on usage can be absolute or relative.

    This class wraps the TopLoc_Location class from OCCT. It can be used to move Shape
    objects in both relative and absolute manner. It is the preferred type to locate objects
    in CQ.
    """

    wrapped: TopLoc_Location

    def __init__(self, *args):
        # print("In Location's __init__(self, *args): function......")      

        T = gp_Trsf()

        if len(args) == 0:
            pass
        elif len(args) == 1:
            t = args[0]

            if isinstance(t, (Vector, tuple)):
                T.SetTranslationPart(Vector(t).wrapped)
            elif isinstance(t, Plane):
                cs = gp_Ax3(t.origin.toPnt(), t.zDir.toDir(), t.xDir.toDir())
                T.SetTransformation(cs)
                T.Invert()
            elif isinstance(t, TopLoc_Location):
                self.wrapped = t
                return
            elif isinstance(t, gp_Trsf):
                T = t
            else:
                raise TypeError("Unexpected parameters")
        elif len(args) == 2:
            t, v = args
            cs = gp_Ax3(Vector(v).toPnt(), t.zDir.toDir(), t.xDir.toDir())
            T.SetTransformation(cs)
            T.Invert()
        else:
            t, ax, angle = args
            T.SetRotation(
                gp_Ax1(Vector().toPnt(), Vector(ax).toDir()), angle * math.pi / 180.0
            )
            T.SetTranslationPart(Vector(t).wrapped)

        self.wrapped = TopLoc_Location(T)
