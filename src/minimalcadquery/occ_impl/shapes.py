from typing import (
    Optional,
    Tuple,
    Union,
    Iterable,
    List,
    Sequence,
    Iterator,
    Dict,
    Any,
    overload,
    TypeVar,
    cast as tcast,
)
from typing_extensions import Literal, Protocol

from io import BytesIO

from .geom import Vector, VectorLike, Plane, Location, Matrix
from .shape_protocols import geom_LUT_FACE, geom_LUT_EDGE, Shapes, Geoms

from ..selectors import (
    Selector,
    StringSyntaxSelector,
)

# change default OCCT logging level
from OCP.Message import Message, Message_Gravity

for printer in Message.DefaultMessenger_s().Printers():
    printer.SetTraceLevel(Message_Gravity.Message_Fail)

import OCP.TopAbs as ta  # Topology type enum
import OCP.GeomAbs as ga  # Geometry type enum

from OCP.Precision import Precision

from OCP.gp import (
    gp_Vec,
    gp_Pnt,
    gp_Ax1,
    gp_Ax2,
    gp_Ax3,
    gp_Dir,
    gp_Circ,
    gp_Trsf,
    gp_Pln,
    gp_Pnt2d,
    gp_Dir2d,
    gp_Elips,
)

# Array of points (used for B-spline construction):
from OCP.TColgp import TColgp_HArray1OfPnt, TColgp_HArray2OfPnt

# Array of vectors (used for B-spline interpolation):
from OCP.TColgp import TColgp_Array1OfVec

# Array of booleans (used for B-spline interpolation):
from OCP.TColStd import TColStd_HArray1OfBoolean

# Array of floats (used for B-spline interpolation):
from OCP.TColStd import TColStd_HArray1OfReal

from OCP.BRepAdaptor import (
    BRepAdaptor_Curve,
    BRepAdaptor_CompCurve,
    BRepAdaptor_Surface,
)

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_GTransform,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_Transformed,
    BRepBuilderAPI_RightCorner,
    BRepBuilderAPI_RoundCorner,
    BRepBuilderAPI_MakeSolid,
)

# properties used to store mass calculation result
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp_Face, BRepGProp  # used for mass calculation

from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeTorus,
    BRepPrimAPI_MakeWedge,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeSphere,
)
from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter

from OCP.TopExp import TopExp  # Topology explorer

# used for getting underlying geometry -- is this equivalent to brep adaptor?
from OCP.BRep import BRep_Tool, BRep_Builder

from OCP.TopoDS import (
    TopoDS,
    TopoDS_Shape,
    TopoDS_Builder,
    TopoDS_Compound,
    TopoDS_Iterator,
    TopoDS_Wire,
    TopoDS_Face,
    TopoDS_Edge,
    TopoDS_Vertex,
    TopoDS_Solid,
    TopoDS_Shell,
    TopoDS_CompSolid,
)

from OCP.GC import GC_MakeArcOfCircle, GC_MakeArcOfEllipse  # geometry construction
from OCP.GCE2d import GCE2d_MakeSegment
from OCP.gce import gce_MakeLin, gce_MakeDir
from OCP.GeomAPI import (
    GeomAPI_Interpolate,
    GeomAPI_ProjectPointOnSurf,
    GeomAPI_PointsToBSpline,
    GeomAPI_PointsToBSplineSurface,
)

from OCP.BRepFill import BRepFill

from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Fuse,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_BooleanOperation,
    BRepAlgoAPI_Splitter,
)

from OCP.Geom import (
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Surface,
    Geom_Plane,
)
from OCP.Geom2d import Geom2d_Line

from OCP.BRepLib import BRepLib, BRepLib_FindSurface

from OCP.BRepOffsetAPI import (
    BRepOffsetAPI_ThruSections,
    BRepOffsetAPI_MakePipeShell,
    BRepOffsetAPI_MakeThickSolid,
    BRepOffsetAPI_MakeOffset,
)

from OCP.BRepFilletAPI import (
    BRepFilletAPI_MakeChamfer,
    BRepFilletAPI_MakeFillet,
    BRepFilletAPI_MakeFillet2d,
)

from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListOfShape,
    TopTools_MapOfShape,
    TopTools_IndexedMapOfShape,
)


from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid, ShapeFix_Face

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer

from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

from OCP.BRepTools import BRepTools, BRepTools_WireExplorer

from OCP.LocOpe import LocOpe_DPrism

from OCP.BRepCheck import BRepCheck_Analyzer

from OCP.Font import (
    Font_FontMgr,
    Font_FA_Regular,
    Font_FA_Italic,
    Font_FA_Bold,
    Font_SystemFont,
)

from OCP.StdPrs import StdPrs_BRepFont, StdPrs_BRepTextBuilder as Font_BRepTextBuilder
from OCP.Graphic3d import (
    Graphic3d_HTA_LEFT,
    Graphic3d_HTA_CENTER,
    Graphic3d_HTA_RIGHT,
    Graphic3d_VTA_BOTTOM,
    Graphic3d_VTA_CENTER,
    Graphic3d_VTA_TOP,
)

from OCP.NCollection import NCollection_Utf8String

from OCP.BRepFeat import BRepFeat_MakeDPrism

from OCP.BRepClass3d import BRepClass3d_SolidClassifier

from OCP.TCollection import TCollection_AsciiString

from OCP.TopLoc import TopLoc_Location

from OCP.GeomAbs import (
    GeomAbs_Shape,
    GeomAbs_C0,
    GeomAbs_Intersection,
    GeomAbs_JoinType,
)
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Mode

from OCP.BOPAlgo import BOPAlgo_GlueEnum

from OCP.IFSelect import IFSelect_ReturnStatus

from OCP.TopAbs import TopAbs_ShapeEnum, TopAbs_Orientation

from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopTools import TopTools_HSequenceOfShape

from OCP.GCPnts import GCPnts_AbscissaPoint

from OCP.GeomFill import (
    GeomFill_Frenet,
    GeomFill_CorrectedFrenet,
    GeomFill_TrihedronLaw,
)

from OCP.BRepProj import BRepProj_Projection
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from OCP.IVtkOCC import IVtkOCC_Shape, IVtkOCC_ShapeMesher
from OCP.IVtkVTK import IVtkVTK_ShapeData

# for catching exceptions
from OCP.Standard import Standard_NoSuchObject, Standard_Failure

from OCP.Prs3d import Prs3d_IsoAspect
from OCP.Quantity import Quantity_Color
from OCP.Aspect import Aspect_TOL_SOLID

from OCP.Interface import Interface_Static

from OCP.ShapeCustom import ShapeCustom, ShapeCustom_RestrictionParameters

from OCP.BRepAlgo import BRepAlgo

from math import pi, sqrt, inf, radians, cos

import warnings

from multimethod import multimethod, DispatchError

Real = (float | int)

TOLERANCE = 1e-6
HASH_CODE_MAX = 2147483647  # max 32bit signed int, required by OCC.Core.HashCode

shape_LUT = {
    ta.TopAbs_VERTEX: "Vertex",
    ta.TopAbs_EDGE: "Edge",
    ta.TopAbs_WIRE: "Wire",
    ta.TopAbs_FACE: "Face",
    ta.TopAbs_SHELL: "Shell",
    ta.TopAbs_SOLID: "Solid",
    ta.TopAbs_COMPSOLID: "CompSolid",
    ta.TopAbs_COMPOUND: "Compound",
}

shape_properties_LUT = {
    ta.TopAbs_VERTEX: None,
    ta.TopAbs_EDGE: BRepGProp.LinearProperties_s,
    ta.TopAbs_WIRE: BRepGProp.LinearProperties_s,
    ta.TopAbs_FACE: BRepGProp.SurfaceProperties_s,
    ta.TopAbs_SHELL: BRepGProp.SurfaceProperties_s,
    ta.TopAbs_SOLID: BRepGProp.VolumeProperties_s,
    ta.TopAbs_COMPOUND: BRepGProp.VolumeProperties_s,
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

geom_LUT = {
    ta.TopAbs_VERTEX: "Vertex",
    ta.TopAbs_EDGE: BRepAdaptor_Curve,
    ta.TopAbs_WIRE: "Wire",
    ta.TopAbs_FACE: BRepAdaptor_Surface,
    ta.TopAbs_SHELL: "Shell",
    ta.TopAbs_SOLID: "Solid",
    ta.TopAbs_SOLID: "CompSolid",
    ta.TopAbs_COMPOUND: "Compound",
}

ancestors_LUT = {
    "Vertex": ta.TopAbs_EDGE,
    "Edge": ta.TopAbs_WIRE,
    "Wire": ta.TopAbs_FACE,
    "Face": ta.TopAbs_SHELL,
    "Shell": ta.TopAbs_SOLID,
}

T = TypeVar("T", bound="Shape")

class cqmultimethod(multimethod):
    def __call__(self, *args, **kwargs):

        try:
            return super().__call__(*args, **kwargs)
        except DispatchError:
            return next(iter(self.values()))(*args, **kwargs)


def shapetype(obj: TopoDS_Shape) -> TopAbs_ShapeEnum:

    if obj.IsNull():
        raise ValueError("Null TopoDS_Shape object")

    return obj.ShapeType()


def downcast(obj: TopoDS_Shape) -> TopoDS_Shape:
    """
    Downcasts a TopoDS object to suitable specialized type
    """

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
        self.wrapped = downcast(obj)

        self.forConstruction = False
        # Helps identify this solid through the use of an ID
        self.label = ""

    def clean(self: T) -> T:
        """Experimental clean using ShapeUpgrade"""

        upgrader = ShapeUpgrade_UnifySameDomain(self.wrapped, True, True, True)
        upgrader.AllowInternalEdges(False)
        upgrader.Build()

        return self.__class__(upgrader.Shape())

    @classmethod
    def cast(cls, obj: TopoDS_Shape, forConstruction: bool = False) -> "Shape":
        "Returns the right type of wrapper, given a OCCT object"

        tr = None

        # define the shape lookup table for casting
        constructor_LUT = {
            ta.TopAbs_VERTEX: Vertex,
            ta.TopAbs_EDGE: Edge,
            ta.TopAbs_WIRE: Wire,
            ta.TopAbs_FACE: Face,
            ta.TopAbs_SHELL: Shell,
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

    def isNull(self) -> bool:
        """
        Returns true if this shape is null. In other words, it references no
        underlying shape with the potential to be given a location and an
        orientation.
        """
        return self.wrapped.IsNull()

    def isSame(self, other: "Shape") -> bool:
        """
        Returns True if other and this shape are same, i.e. if they share the
        same TShape with the same Locations. Orientations may differ. Also see
        :py:meth:`isEqual`
        """
        return self.wrapped.IsSame(other.wrapped)

    def ShapeType(self) -> Shapes:
        return tcast(Shapes, shape_LUT[shapetype(self.wrapped)])

    def _entities(self, topo_type: Shapes) -> Iterable[TopoDS_Shape]:

        shape_set = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.wrapped, inverse_shape_LUT[topo_type], shape_set)

        return tcast(Iterable[TopoDS_Shape], shape_set)

    def Faces(self) -> List["Face"]:
        """
        :returns: All the faces in this Shape
        """

        return [Face(i) for i in self._entities("Face")]
    
    def move(self: T, loc: Location) -> T:
        """
        Apply a location in relative sense (i.e. update current location) to self
        """

        self.wrapped.Move(loc.wrapped)

        return self

    def moved(self: T, loc: Location) -> T:
        """
        Apply a location in relative sense (i.e. update current location) to a copy of self
        """

        r = self.__class__(self.wrapped.Moved(loc.wrapped))
        r.forConstruction = self.forConstruction

        return r

    def __eq__(self, other) -> bool:

        return self.isSame(other) if isinstance(other, Shape) else False

    def _bool_op(
        self,
        args: Iterable["Shape"],
        tools: Iterable["Shape"],
        op: (BRepAlgoAPI_BooleanOperation | BRepAlgoAPI_Splitter),
        parallel: bool = True,
    ) -> "Shape":
        """
        Generic boolean operation

        :param parallel: Sets the SetRunParallel flag, which enables parallel execution of boolean operations in OCC kernel
        """

        arg = TopTools_ListOfShape()
        for obj in args:
            arg.Append(obj.wrapped)

        tool = TopTools_ListOfShape()
        for obj in tools:
            tool.Append(obj.wrapped)

        op.SetArguments(arg)
        op.SetTools(tool)

        op.SetRunParallel(parallel)
        op.Build()

        return Shape.cast(op.Shape())

    def cut(self, *toCut: "Shape", tol: Optional[float] = None) -> "Shape":
        """
        Remove the positional arguments from this Shape.

        :param tol: Fuzzy mode tolerance
        """

        cut_op = BRepAlgoAPI_Cut()

        if tol:
            cut_op.SetFuzzyValue(tol)

        return self._bool_op((self,), toCut, cut_op)

    def fuse(
        self, *toFuse: "Shape", glue: bool = False, tol: Optional[float] = None
    ) -> "Shape":
        """
        Fuse the positional arguments with this Shape.

        :param glue: Sets the glue option for the algorithm, which allows
            increasing performance of the intersection of the input shapes
        :param tol: Fuzzy mode tolerance
        """

        fuse_op = BRepAlgoAPI_Fuse()
        if glue:
            fuse_op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
        if tol:
            fuse_op.SetFuzzyValue(tol)

        rv = self._bool_op((self,), toFuse, fuse_op)

        return rv

    def __iter__(self) -> Iterator["Shape"]:
        """
        Iterate over subshapes.

        """

        it = TopoDS_Iterator(self.wrapped)

        while it.More():
            yield Shape.cast(it.Value())
            it.Next()

class ShapeProtocol(Protocol):
    @property
    def wrapped(self) -> TopoDS_Shape:
        ...

    def __init__(self, wrapped: TopoDS_Shape) -> None:
        ...

    def Faces(self) -> List["Face"]:
        ...

    def geomType(self) -> Geoms:
        ...


class Vertex(Shape):
    """
    A Single Point in Space
    """

    pass

class Mixin1DProtocol(ShapeProtocol, Protocol):
    pass

T1D = TypeVar("T1D", bound=Mixin1DProtocol)


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

    def _geomAdaptor(self) -> BRepAdaptor_CompCurve:
        """
        Return the underlying geometry
        """

        return BRepAdaptor_CompCurve(self.wrapped)

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

class Shell(Shape):
    """
    the outer boundary of a surface
    """

    pass

TS = TypeVar("TS", bound=ShapeProtocol)


class Mixin3D(object):

    pass

class Solid(Shape, Mixin3D):
    """
    a single solid
    """

    wrapped: TopoDS_Solid

    @staticmethod
    def isSolid(obj: Shape) -> bool:
        """
        Returns true if the object is a solid, false otherwise
        """
        if hasattr(obj, "ShapeType"):
            if obj.ShapeType == "Solid" or (
                obj.ShapeType == "Compound" and len(obj.Solids()) > 0
            ):
                return True
        return False

    @classmethod
    def makeSolid(cls, shell: Shell) -> "Solid":
        """
        Makes a solid from a single shell.
        """

        return cls(ShapeFix_Solid().SolidFromShell(shell.wrapped))

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
        return cls(
            BRepPrimAPI_MakeBox(
                gp_Ax2(Vector(pnt).toPnt(), Vector(dir).toDir()), length, width, height
            ).Shape()
        )

    @cqmultimethod
    def extrudeLinear(
        cls,
        outerWire: Wire,
        innerWires: List[Wire],
        vecNormal: VectorLike,
        taper: Real = 0,
    ) -> "Solid":
        """
        Attempt to extrude the list of wires into a prismatic solid in the provided direction

        :param outerWire: the outermost wire
        :param innerWires: a list of inner wires
        :param vecNormal: a vector along which to extrude the wires
        :param taper: taper angle, default=0
        :return: a Solid object

        The wires must not intersect

        Extruding wires is very non-trivial.  Nested wires imply very different geometry, and
        there are many geometries that are invalid. In general, the following conditions must be met:

        * all wires must be closed
        * there cannot be any intersecting or self-intersecting wires
        * wires must be listed from outside in
        * more than one levels of nesting is not supported reliably

        This method will attempt to sort the wires, but there is much work remaining to make this method
        reliable.
        """

        if taper == 0:
            face = Face.makeFromWires(outerWire, innerWires)
        else:
            face = Face.makeFromWires(outerWire)

        return cls.extrudeLinear(face, vecNormal, taper)

    @classmethod
    @extrudeLinear.register
    def extrudeLinear(
        cls, face: Face, vecNormal: VectorLike, taper: Real = 0,
    ) -> "Solid":

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

        return cls(cls._makeCompound((s.wrapped for s in listOfShapes)))

    def __bool__(self) -> bool:
        """
        Check if empty.
        """

        return TopoDS_Iterator(self.wrapped).More()

    def cut(self, *toCut: "Shape", tol: Optional[float] = None) -> "Compound":
        """
        Remove the positional arguments from this Shape.

        :param tol: Fuzzy mode tolerance
        """

        cut_op = BRepAlgoAPI_Cut()

        if tol:
            cut_op.SetFuzzyValue(tol)

        return tcast(Compound, self._bool_op(self, toCut, cut_op))

    def fuse(
        self, *toFuse: Shape, glue: bool = False, tol: Optional[float] = None
    ) -> "Compound":
        """
        Fuse shapes together
        """

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

        # fuse_op.RefineEdges()
        # fuse_op.FuseEdges()

        return tcast(Compound, rv)

def sortWiresByBuildOrder(wireList: List[Wire]) -> List[List[Wire]]:
    """Tries to determine how wires should be combined into faces.

    Assume:
        The wires make up one or more faces, which could have 'holes'
        Outer wires are listed ahead of inner wires
        there are no wires inside wires inside wires
        ( IE, islands -- we can deal with that later on )
        none of the wires are construction wires

    Compute:
        one or more sets of wires, with the outer wire listed first, and inner
        ones

    Returns, list of lists.
    """

    # check if we have something to sort at all
    if len(wireList) < 2:
        return [
            wireList,
        ]

    # make a Face, NB: this might return a compound of faces
    faces = Face.makeFromWires(wireList[0], wireList[1:])

    rv = []
    for face in faces.Faces():
        rv.append([face.outerWire(),] + face.innerWires())

    return rv

def wiresToFaces(wireList: List[Wire]) -> List[Face]:
    """
    Convert wires to a list of faces.
    """
    return Face.makeFromWires(wireList[0], wireList[1:]).Faces()