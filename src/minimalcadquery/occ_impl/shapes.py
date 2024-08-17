from typing import (
    Optional,
    Tuple,
    Iterable,
    List,
    Iterator,
    Any,
    TypeVar,
    cast as tcast,
)
from typing_extensions import Protocol

from .geom import Vector, VectorLike, Location

# change default OCCT logging level
from OCP.Message import Message, Message_Gravity

for printer in Message.DefaultMessenger_s().Printers():
    printer.SetTraceLevel(Message_Gravity.Message_Fail)

import OCP.TopAbs as ta  # Topology type enum

from OCP.gp import (
    gp_Ax2,
)

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
)

from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakePrism,
)

from OCP.TopExp import TopExp  # Topology explorer

from OCP.TopoDS import (
    TopoDS,
    TopoDS_Shape,
    TopoDS_Builder,
    TopoDS_Compound,
    TopoDS_Iterator,
    TopoDS_Wire,
    TopoDS_Face,
    TopoDS_Solid,
    TopoDS_CompSolid,
)

from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_Fuse,
)

from OCP.BRepLib import BRepLib_FindSurface

from OCP.TopTools import (
    TopTools_IndexedMapOfShape,
)

from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Face

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

from OCP.LocOpe import LocOpe_DPrism

from OCP.BOPAlgo import BOPAlgo_GlueEnum

from OCP.IFSelect import IFSelect_ReturnStatus

from OCP.TopAbs import TopAbs_ShapeEnum

from OCP.Interface import Interface_Static

from math import radians, cos

from typing import Literal

import logging
logger = logging.getLogger(__name__)

Shapes = Literal[
    "Vertex", "Edge", "Wire", "Face", "Shell", "Solid", "CompSolid", "Compound"
]

Real = (float | int)

TOLERANCE = 1e-6
HASH_CODE_MAX = 2147483647  # max 32bit signed int, required by OCC.Core.HashCode

shape_LUT = {
    #ta.TopAbs_VERTEX: "Vertex",
    ta.TopAbs_EDGE: "Edge",
    ta.TopAbs_WIRE: "Wire",
    ta.TopAbs_FACE: "Face",
    #ta.TopAbs_SHELL: "Shell",
    ta.TopAbs_SOLID: "Solid",
    ta.TopAbs_COMPSOLID: "CompSolid",
    ta.TopAbs_COMPOUND: "Compound",
}

inverse_shape_LUT = {v: k for k, v in shape_LUT.items()}

downcast_LUT = {
    ta.TopAbs_VERTEX: TopoDS.Vertex_s,
    ta.TopAbs_EDGE: TopoDS.Edge_s,
    ta.TopAbs_WIRE: TopoDS.Wire_s,
    ta.TopAbs_FACE: TopoDS.Face_s,
    ta.TopAbs_SHELL: TopoDS.Shell_s,
    ta.TopAbs_SOLID: TopoDS.Solid_s,
    ta.TopAbs_COMPSOLID: TopoDS.CompSolid_s,
    ta.TopAbs_COMPOUND: TopoDS.Compound_s,
}

T = TypeVar("T", bound="Shape")

def shapetype(obj: TopoDS_Shape) -> TopAbs_ShapeEnum:
    logger.info("shapetype() is called")

    if obj.IsNull():
        raise ValueError("Null TopoDS_Shape object")

    return obj.ShapeType()


def downcast(obj: TopoDS_Shape) -> TopoDS_Shape:
    """
    Downcasts a TopoDS object to suitable specialized type
    """
    logger.info("downcast() is called")

    f_downcast: Any = downcast_LUT[shapetype(obj)]
    rv = f_downcast(obj)

    return rv

class Shape(object):
    """
    Represents a shape in the system. Wraps TopoDS_Shape.
    """

    wrapped: TopoDS_Shape
    forConstruction: bool

    def __init__(self, obj: TopoDS_Shape):
        logger.info("Shape's __init__() is called")
        self.wrapped = downcast(obj)

        self.forConstruction = False
        # Helps identify this solid through the use of an ID
        self.label = ""

    def clean(self: T) -> T:
        """Experimental clean using ShapeUpgrade"""
        logger.info("Shape's clean() is called")

        upgrader = ShapeUpgrade_UnifySameDomain(self.wrapped, True, True, True)
        upgrader.AllowInternalEdges(False)
        upgrader.Build()

        return self.__class__(upgrader.Shape())

    @classmethod
    def cast(cls, obj: TopoDS_Shape, forConstruction: bool = False) -> "Shape":
        "Returns the right type of wrapper, given a OCCT object"
        logger.info("Shape's cast() is called")

        tr = None

        # define the shape lookup table for casting
        constructor_LUT = {
            #ta.TopAbs_VERTEX: Vertex,
            ta.TopAbs_EDGE: Edge,
            ta.TopAbs_WIRE: Wire,
            ta.TopAbs_FACE: Face,
            #ta.TopAbs_SHELL: Shell,
            ta.TopAbs_SOLID: Solid,
            ta.TopAbs_COMPSOLID: CompSolid,
            ta.TopAbs_COMPOUND: Compound,
        }

        t = shapetype(obj)
        # NB downcast is needed to handle TopoDS_Shape types
        tr = constructor_LUT[t](downcast(obj))
        tr.forConstruction = forConstruction

        return tr

    def exportStep(self, fileName: str, **kwargs) -> IFSelect_ReturnStatus:
        """
        Export this shape to a STEP file.

        kwargs is used to provide optional keyword arguments to configure the exporter.

        :param fileName: Path and filename for writing.
        :param write_pcurves: Enable or disable writing parametric curves to the STEP file. Default True.

            If False, writes STEP file without pcurves. This decreases the size of the resulting STEP file.
        :type write_pcurves: bool
        :param precision_mode: Controls the uncertainty value for STEP entities. Specify -1, 0, or 1. Default 0.
            See OCCT documentation.
        :type precision_mode: int
        """
        logger.info("Shape's exportStep() is called")

        # Handle the extra settings for the STEP export
        pcurves = 1
        if "write_pcurves" in kwargs and not kwargs["write_pcurves"]:
            pcurves = 0
        precision_mode = kwargs["precision_mode"] if "precision_mode" in kwargs else 0

        writer = STEPControl_Writer()
        Interface_Static.SetIVal_s("write.surfacecurve.mode", pcurves)
        Interface_Static.SetIVal_s("write.precision.mode", precision_mode)
        writer.Transfer(self.wrapped, STEPControl_AsIs)

        return writer.Write(fileName)

    def _entities(self, topo_type: Shapes) -> Iterable[TopoDS_Shape]:
        logger.info("Shape's _entities() is called")

        shape_set = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.wrapped, inverse_shape_LUT[topo_type], shape_set)

        return tcast(Iterable[TopoDS_Shape], shape_set)

    def Faces(self) -> List["Face"]:
        """
        :returns: All the faces in this Shape
        """
        logger.info("Shape's Faces() is called")

        return [Face(i) for i in self._entities("Face")]
    
    def move(self: T, loc: Location) -> T:
        """
        Apply a location in relative sense (i.e. update current location) to self
        """
        logger.info("Shape's move() is called")

        self.wrapped.Move(loc.wrapped)

        return self

    def moved(self: T, loc: Location) -> T:
        """
        Apply a location in relative sense (i.e. update current location) to a copy of self
        """
        logger.info("Shape's moved() is called")

        r = self.__class__(self.wrapped.Moved(loc.wrapped))
        r.forConstruction = self.forConstruction

        return r

    def __iter__(self) -> Iterator["Shape"]:
        """
        Iterate over subshapes.

        """
        logger.info("Shape's __iter__() is called")

        it = TopoDS_Iterator(self.wrapped)

        while it.More():
            yield Shape.cast(it.Value())
            it.Next()

class Mixin1D(object):

    pass

class Edge(Shape, Mixin1D):
    """
    A trimmed curve that represents the border of a face
    """
    pass

class Wire(Shape, Mixin1D):
    """
    A series of connected, ordered Edges, that typically bounds a Face
    """

    wrapped: TopoDS_Wire

    @classmethod
    def makePolygon(
        cls,
        listOfVertices: Iterable[VectorLike],
        forConstruction: bool = False,
        close: bool = False,
    ) -> "Wire":
        """
        Construct a polygonal wire from points.
        """

        logger.info("Wire's makePolygon() is called")

        wire_builder = BRepBuilderAPI_MakePolygon()

        for v in listOfVertices:
            wire_builder.Add(Vector(v).toPnt())

        if close:
            wire_builder.Close()

        w = cls(wire_builder.Wire())
        w.forConstruction = forConstruction

        return w

class Face(Shape):
    """
    a bounded surface that represents part of the boundary of a solid
    """

    wrapped: TopoDS_Face

    @classmethod
    def makeFromWires(cls, outerWire: Wire, innerWires: List[Wire] = []) -> "Face":
        """
        Makes a planar face from one or more wires
        """
        logger.info("Face's makeFromWires() is called")

        if innerWires and not outerWire.IsClosed():
            raise ValueError("Cannot build face(s): outer wire is not closed")

        # check if wires are coplanar
        ws = Compound.makeCompound([outerWire] + innerWires)
        if not BRepLib_FindSurface(ws.wrapped, OnlyPlane=True).Found():
            raise ValueError("Cannot build face(s): wires not planar")

        # fix outer wire
        sf_s = ShapeFix_Shape(outerWire.wrapped)
        sf_s.Perform()
        wo = TopoDS.Wire_s(sf_s.Shape())

        face_builder = BRepBuilderAPI_MakeFace(wo, True)

        for w in innerWires:
            if not w.IsClosed():
                raise ValueError("Cannot build face(s): inner wire is not closed")
            face_builder.Add(w.wrapped)

        face_builder.Build()

        if not face_builder.IsDone():
            raise ValueError(f"Cannot build face(s): {face_builder.Error()}")

        face = face_builder.Face()

        sf_f = ShapeFix_Face(face)
        sf_f.FixOrientation()
        sf_f.Perform()

        return cls(sf_f.Result())

class Mixin3D(object):

    pass

class Solid(Shape, Mixin3D):
    """
    a single solid
    """

    wrapped: TopoDS_Solid

    @classmethod
    def makeBox(
        cls,
        length: float,
        width: float,
        height: float,
        pnt: VectorLike = Vector(0, 0, 0),
        dir: VectorLike = Vector(0, 0, 1),
    ) -> "Solid":
        """
        makeBox(length,width,height,[pnt,dir]) -- Make a box located in pnt with the dimensions (length,width,height)
        By default pnt=Vector(0,0,0) and dir=Vector(0,0,1)
        """
        logger.info("Solid's makeBox() is called")

        return cls(
            BRepPrimAPI_MakeBox(
                gp_Ax2(Vector(pnt).toPnt(), Vector(dir).toDir()), length, width, height
            ).Shape()
        )

    @classmethod
    def extrudeLinear(
        cls, face: Face, vecNormal: VectorLike, taper: Real = 0,
    ) -> "Solid":
        logger.info("Solid's extrudeLinear() is called")

        if taper == 0:
            prism_builder: Any = BRepPrimAPI_MakePrism(
                face.wrapped, Vector(vecNormal).wrapped, True
            )
        else:
            faceNormal = face.normalAt()
            d = 1 if vecNormal.getAngle(faceNormal) < radians(90.0) else -1

            # Divided by cos of taper angle to ensure the height chosen by the user is respected
            prism_builder = LocOpe_DPrism(
                face.wrapped,
                (d * vecNormal.Length) / cos(radians(taper)),
                d * radians(taper),
            )

        return cls(prism_builder.Shape())

class CompSolid(Shape, Mixin3D):
    """
    a single compsolid
    """

    wrapped: TopoDS_CompSolid

class Compound(Shape, Mixin3D):
    """
    a collection of disconnected solids
    """

    wrapped: TopoDS_Compound

    @staticmethod
    def _makeCompound(listOfShapes: Iterable[TopoDS_Shape]) -> TopoDS_Compound:
        logger.info("Compound's _makeCompound() is called")

        comp = TopoDS_Compound()
        comp_builder = TopoDS_Builder()
        comp_builder.MakeCompound(comp)

        for s in listOfShapes:
            comp_builder.Add(comp, s)

        return comp

    @classmethod
    def makeCompound(cls, listOfShapes: Iterable[Shape]) -> "Compound":
        """
        Create a compound out of a list of shapes
        """
        logger.info("Compound's makeCompound() is called")

        return cls(cls._makeCompound((s.wrapped for s in listOfShapes)))

    def fuse(
        self, *toFuse: Shape, glue: bool = False, tol: Optional[float] = None
    ) -> "Compound":
        """
        Fuse shapes together
        """
        logger.info("Compound's fuse() is called")

        fuse_op = BRepAlgoAPI_Fuse()
        if glue:
            fuse_op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
        if tol:
            fuse_op.SetFuzzyValue(tol)

        args = tuple(self) + toFuse

        if len(args) <= 1:
            rv: Shape = args[0]
        else:
            rv = self._bool_op(args[:1], args[1:], fuse_op)

        return tcast(Compound, rv)

def wiresToFaces(wireList: List[Wire]) -> List[Face]:
    """
    Convert wires to a list of faces.
    """
    logger.info("wiresToFaces() is called")

    return Face.makeFromWires(wireList[0], wireList[1:]).Faces()