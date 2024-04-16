import math

from typing import overload, Sequence, Union, Tuple, Type, Optional

from OCP.gp import (
    gp_Vec,
    gp_Ax1,
    gp_Ax3,
    gp_Pnt,
    gp_Dir,
    gp_Pln,
    gp_Trsf,
    gp_GTrsf,
    gp_XYZ,
    gp_EulerSequence,
    gp,
)
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopoDS import TopoDS_Shape
from OCP.TopLoc import TopLoc_Location

from ..types import Real

TOL = 1e-2

#VectorLike = Union["Vector", Tuple[Real, Real], Tuple[Real, Real, Real]]
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

    @overload
    def __init__(self, x: float, y: float, z: float) -> None:
        ...

    @overload
    def __init__(self, x: float, y: float) -> None:
        ...

    @overload
    def __init__(self, v: "Vector") -> None:
        ...

    @overload
    def __init__(self, v: Sequence[float]) -> None:
        ...

    @overload
    def __init__(self, v: Union[gp_Vec, gp_Pnt, gp_Dir, gp_XYZ]) -> None:
        ...

    @overload
    def __init__(self) -> None:
        ...

    def __init__(self, *args):
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
        return self.wrapped.X()

    @x.setter
    def x(self, value: float) -> None:
        self.wrapped.SetX(value)

    @property
    def y(self) -> float:
        return self.wrapped.Y()

    @y.setter
    def y(self, value: float) -> None:
        self.wrapped.SetY(value)

    @property
    def z(self) -> float:
        return self.wrapped.Z()

    @z.setter
    def z(self, value: float) -> None:
        self.wrapped.SetZ(value)

    @property
    def Length(self) -> float:
        return self.wrapped.Magnitude()

    @property
    def wrapped(self) -> gp_Vec:
        return self._wrapped

    def toTuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def cross(self, v: "Vector") -> "Vector":
        return Vector(self.wrapped.Crossed(v.wrapped))

    def dot(self, v: "Vector") -> float:
        return self.wrapped.Dot(v.wrapped)

    def sub(self, v: "Vector") -> "Vector":
        return Vector(self.wrapped.Subtracted(v.wrapped))

    def __sub__(self, v: "Vector") -> "Vector":
        return self.sub(v)

    def add(self, v: "Vector") -> "Vector":
        return Vector(self.wrapped.Added(v.wrapped))

    def __add__(self, v: "Vector") -> "Vector":
        return self.add(v)

    def multiply(self, scale: float) -> "Vector":
        """Return a copy multiplied by the provided scalar"""
        return Vector(self.wrapped.Multiplied(scale))

    def __mul__(self, scale: float) -> "Vector":
        return self.multiply(scale)

    def __truediv__(self, denom: float) -> "Vector":
        return self.multiply(1.0 / denom)

    def __rmul__(self, scale: float) -> "Vector":
        return self.multiply(scale)

    def normalized(self) -> "Vector":
        """Return a normalized version of this vector"""
        return Vector(self.wrapped.Normalized())

    def Center(self) -> "Vector":
        """Return the vector itself

        The center of myself is myself.
        Provided so that vectors, vertices, and other shapes all support a
        common interface, when Center() is requested for all objects on the
        stack.
        """
        return self

    def toPnt(self) -> gp_Pnt:

        return gp_Pnt(self.wrapped.XYZ())

    def toDir(self) -> gp_Dir:

        return gp_Dir(self.wrapped.XYZ())

    def transform(self, T: "Matrix") -> "Vector":

        # to gp_Pnt to obey cq transformation convention (in OCP.vectors do not translate)
        pnt = self.toPnt()
        pnt_t = pnt.Transformed(T.wrapped.Trsf())

        return Vector(gp_Vec(pnt_t.XYZ()))

class Matrix:
    """A 3d , 4x4 transformation matrix.

    Used to move geometry in space.

    The provided "matrix" parameter may be None, a gp_GTrsf, or a nested list of
    values.

    If given a nested list, it is expected to be of the form:

        [[m11, m12, m13, m14],
         [m21, m22, m23, m24],
         [m31, m32, m33, m34]]

    A fourth row may be given, but it is expected to be: [0.0, 0.0, 0.0, 1.0]
    since this is a transform matrix.
    """

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

    @classmethod
    def XY(cls, origin=(0, 0, 0), xDir=Vector(1, 0, 0)):
        plane = Plane.named("XY", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def YZ(cls, origin=(0, 0, 0), xDir=Vector(0, 1, 0)):
        plane = Plane.named("YZ", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def ZX(cls, origin=(0, 0, 0), xDir=Vector(0, 0, 1)):
        plane = Plane.named("ZX", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def XZ(cls, origin=(0, 0, 0), xDir=Vector(1, 0, 0)):
        plane = Plane.named("XZ", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def YX(cls, origin=(0, 0, 0), xDir=Vector(0, 1, 0)):
        plane = Plane.named("YX", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def ZY(cls, origin=(0, 0, 0), xDir=Vector(0, 0, 1)):
        plane = Plane.named("ZY", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def front(cls, origin=(0, 0, 0), xDir=Vector(1, 0, 0)):
        plane = Plane.named("front", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def back(cls, origin=(0, 0, 0), xDir=Vector(-1, 0, 0)):
        plane = Plane.named("back", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def left(cls, origin=(0, 0, 0), xDir=Vector(0, 0, 1)):
        plane = Plane.named("left", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def right(cls, origin=(0, 0, 0), xDir=Vector(0, 0, -1)):
        plane = Plane.named("right", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def top(cls, origin=(0, 0, 0), xDir=Vector(1, 0, 0)):
        plane = Plane.named("top", origin)
        plane._setPlaneDir(xDir)
        return plane

    @classmethod
    def bottom(cls, origin=(0, 0, 0), xDir=Vector(1, 0, 0)):
        plane = Plane.named("bottom", origin)
        plane._setPlaneDir(xDir)
        return plane

    def __init__(
        self,
        origin: Union[Tuple[float, float, float], Vector],
        xDir: Optional[Union[Tuple[float, float, float], Vector]] = None,
        normal: Union[Tuple[float, float, float], Vector] = (0, 0, 1),
    ):
        """
        Create a Plane with an arbitrary orientation

        :param origin: the origin in global coordinates
        :param xDir: an optional vector representing the xDirection.
        :param normal: the normal direction for the plane
        :raises ValueError: if the specified xDir is not orthogonal to the provided normal
        """
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

    def _eq_iter(self, other):
        """Iterator to successively test equality"""
        cls = type(self)
        yield isinstance(other, Plane)  # comparison is with another Plane
        # origins are the same
        yield abs(self.origin - other.origin) < cls._eq_tolerance_origin
        # z-axis vectors are parallel (assumption: both are unit vectors)
        yield abs(self.zDir.dot(other.zDir) - 1) < cls._eq_tolerance_dot
        # x-axis vectors are parallel (assumption: both are unit vectors)
        yield abs(self.xDir.dot(other.xDir) - 1) < cls._eq_tolerance_dot

    def __eq__(self, other):
        return all(self._eq_iter(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"Plane(origin={str(self.origin.toTuple())}, xDir={str(self.xDir.toTuple())}, normal={str(self.zDir.toTuple())})"

    @property
    def origin(self) -> Vector:
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = Vector(value)
        self._calcTransforms()

    def setOrigin2d(self, x, y):
        """
        Set a new origin in the plane itself

        Set a new origin in the plane itself. The plane's orientation and
        xDrection are unaffected.

        :param float x: offset in the x direction
        :param float y: offset in the y direction
        :return: void

        The new coordinates are specified in terms of the current 2D system.
        As an example:

        p = Plane.XY()
        p.setOrigin2d(2, 2)
        p.setOrigin2d(2, 2)

        results in a plane with its origin at (x, y) = (4, 4) in global
        coordinates. Both operations were relative to local coordinates of the
        plane.
        """
        self.origin = self.toWorldCoords((x, y))

    def _setPlaneDir(self, xDir):
        """Set the vectors parallel to the plane, i.e. xDir and yDir"""
        xDir = Vector(xDir)
        self.xDir = xDir.normalized()
        self.yDir = self.zDir.cross(self.xDir).normalized()

    def _calcTransforms(self):
        """Computes transformation matrices to convert between coordinates

        Computes transformation matrices to convert between local and global
        coordinates.
        """
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

        return Location(self)
    
class Location(object):
    """Location in 3D space. Depending on usage can be absolute or relative.

    This class wraps the TopLoc_Location class from OCCT. It can be used to move Shape
    objects in both relative and absolute manner. It is the preferred type to locate objects
    in CQ.
    """

    wrapped: TopLoc_Location

    @overload
    def __init__(self) -> None:
        """Empty location with not rotation or translation with respect to the original location."""
        ...

    @overload
    def __init__(self, t: VectorLike) -> None:
        """Location with translation t with respect to the original location."""
        ...

    @overload
    def __init__(self, t: Plane) -> None:
        """Location corresponding to the location of the Plane t."""
        ...

    @overload
    def __init__(self, t: Plane, v: VectorLike) -> None:
        """Location corresponding to the angular location of the Plane t with translation v."""
        ...

    @overload
    def __init__(self, t: TopLoc_Location) -> None:
        """Location wrapping the low-level TopLoc_Location object t"""
        ...

    @overload
    def __init__(self, t: gp_Trsf) -> None:
        """Location wrapping the low-level gp_Trsf object t"""
        ...

    @overload
    def __init__(self, t: VectorLike, ax: VectorLike, angle: float) -> None:
        """Location with translation t and rotation around ax by angle
        with respect to the original location."""
        ...

    def __init__(self, *args):

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
