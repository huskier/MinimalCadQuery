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

    def geomType(self) -> Geoms:
        """
        Gets the underlying geometry type.

        Implementations can return any values desired, but the values the user
        uses in type filters should correspond to these.

        As an example, if a user does::

            CQ(object).faces("%mytype")

        The expectation is that the geomType attribute will return 'mytype'

        The return values depend on the type of the shape:

        | Vertex:  always 'Vertex'
        | Edge:   LINE, CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA, BEZIER,
        |         BSPLINE, OFFSET, OTHER
        | Face:   PLANE, CYLINDER, CONE, SPHERE, TORUS, BEZIER, BSPLINE,
        |         REVOLUTION, EXTRUSION, OFFSET, OTHER
        | Solid:  'Solid'
        | Shell:  'Shell'
        | Compound: 'Compound'
        | Wire:   'Wire'

        :returns: A string according to the geometry type
        """

        tr: Any = geom_LUT[shapetype(self.wrapped)]

        if isinstance(tr, str):
            rv = tr
        elif tr is BRepAdaptor_Curve:
            rv = geom_LUT_EDGE[tr(self.wrapped).GetType()]
        else:
            rv = geom_LUT_FACE[tr(self.wrapped).GetType()]

        return tcast(Geoms, rv)

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

    @staticmethod
    def _center_of_mass(shape: "Shape") -> Vector:

        Properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.wrapped, Properties)

        return Vector(Properties.CentreOfMass())

    def Center(self) -> Vector:
        """
        :returns: The point of the center of mass of this Shape
        """

        return Shape.centerOfMass(self)
  
    @staticmethod
    def centerOfMass(obj: "Shape") -> Vector:
        """
        Calculates the center of 'mass' of an object.

        :param obj: Compute the center of mass of this object
        """
        Properties = GProp_GProps()
        calc_function = shape_properties_LUT[shapetype(obj.wrapped)]

        if calc_function:
            calc_function(obj.wrapped, Properties)
            return Vector(Properties.CentreOfMass())
        else:
            raise NotImplementedError

    def ShapeType(self) -> Shapes:
        return tcast(Shapes, shape_LUT[shapetype(self.wrapped)])

    def _entities(self, topo_type: Shapes) -> Iterable[TopoDS_Shape]:

        shape_set = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.wrapped, inverse_shape_LUT[topo_type], shape_set)

        return tcast(Iterable[TopoDS_Shape], shape_set)

    def _entitiesFrom(
        self, child_type: Shapes, parent_type: Shapes
    ) -> Dict["Shape", List["Shape"]]:

        res = TopTools_IndexedDataMapOfShapeListOfShape()

        TopExp.MapShapesAndAncestors_s(
            self.wrapped,
            inverse_shape_LUT[child_type],
            inverse_shape_LUT[parent_type],
            res,
        )

        out: Dict[Shape, List[Shape]] = {}
        for i in range(1, res.Extent() + 1):
            out[Shape.cast(res.FindKey(i))] = [
                Shape.cast(el) for el in res.FindFromIndex(i)
            ]

        return out

    def Faces(self) -> List["Face"]:
        """
        :returns: All the faces in this Shape
        """

        return [Face(i) for i in self._entities("Face")]
    
    def _filter(
        self, selector: Optional[Union[Selector, str]], objs: Iterable["Shape"]
    ) -> "Shape":

        selectorObj: Selector
        if selector:
            if isinstance(selector, str):
                selectorObj = StringSyntaxSelector(selector)
            else:
                selectorObj = selector
            selected = selectorObj.filter(list(objs))
        else:
            selected = list(objs)

        if len(selected) == 1:
            rv = selected[0]
        else:
            rv = Compound.makeCompound(selected)

        return rv

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
        op: Union[BRepAlgoAPI_BooleanOperation, BRepAlgoAPI_Splitter],
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

    def intersect(self, *toIntersect: "Shape", tol: Optional[float] = None) -> "Shape":
        """
        Intersection of the positional arguments and this Shape.

        :param tol: Fuzzy mode tolerance
        """

        intersect_op = BRepAlgoAPI_Common()

        if tol:
            intersect_op.SetFuzzyValue(tol)

        return self._bool_op((self,), toIntersect, intersect_op)

    def facesIntersectedByLine(
        self,
        point: VectorLike,
        axis: VectorLike,
        tol: float = 1e-4,
        direction: Optional[Literal["AlongAxis", "Opposite"]] = None,
    ):
        """
        Computes the intersections between the provided line and the faces of this Shape

        :param point: Base point for defining a line
        :param axis: Axis on which the line rests
        :param tol: Intersection tolerance
        :param direction: Valid values: "AlongAxis", "Opposite";
            If specified, will ignore all faces that are not in the specified direction
            including the face where the point lies if it is the case
        :returns: A list of intersected faces sorted by distance from point
        """

        oc_point = (
            gp_Pnt(*point.toTuple()) if isinstance(point, Vector) else gp_Pnt(*point)
        )
        oc_axis = (
            gp_Dir(Vector(axis).wrapped)
            if not isinstance(axis, Vector)
            else gp_Dir(axis.wrapped)
        )

        line = gce_MakeLin(oc_point, oc_axis).Value()
        shape = self.wrapped

        intersectMaker = BRepIntCurveSurface_Inter()
        intersectMaker.Init(shape, line, tol)

        faces_dist = []  # using a list instead of a dictionary to be able to sort it
        while intersectMaker.More():
            interPt = intersectMaker.Pnt()
            interDirMk = gce_MakeDir(oc_point, interPt)

            distance = oc_point.SquareDistance(interPt)

            # interDir is not done when `oc_point` and `oc_axis` have the same coord
            if interDirMk.IsDone():
                interDir: Any = interDirMk.Value()
            else:
                interDir = None

            if direction == "AlongAxis":
                if (
                    interDir is not None
                    and not interDir.IsOpposite(oc_axis, tol)
                    and distance > tol
                ):
                    faces_dist.append((intersectMaker.Face(), distance))

            elif direction == "Opposite":
                if (
                    interDir is not None
                    and interDir.IsOpposite(oc_axis, tol)
                    and distance > tol
                ):
                    faces_dist.append((intersectMaker.Face(), distance))

            elif direction is None:
                faces_dist.append(
                    (intersectMaker.Face(), abs(distance))
                )  # will sort all intersected faces by distance whatever the direction is
            else:
                raise ValueError(
                    "Invalid direction specification.\nValid specification are 'AlongAxis' and 'Opposite'."
                )

            intersectMaker.Next()

        faces_dist.sort(key=lambda x: x[1])
        faces = [face[0] for face in faces_dist]

        return [Face(face) for face in faces]

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
    def _geomAdaptor(self) -> Union[BRepAdaptor_Curve, BRepAdaptor_CompCurve]:
        ...

    def paramAt(self, d: float) -> float:
        ...

    def positionAt(
        self, d: float, mode: Literal["length", "parameter"] = "length",
    ) -> Vector:
        ...

    def locationAt(
        self,
        d: float,
        mode: Literal["length", "parameter"] = "length",
        frame: Literal["frenet", "corrected"] = "frenet",
        planar: bool = False,
    ) -> Location:
        ...


T1D = TypeVar("T1D", bound=Mixin1DProtocol)


class Mixin1D(object):
    def _bounds(self: Mixin1DProtocol) -> Tuple[float, float]:

        curve = self._geomAdaptor()
        return curve.FirstParameter(), curve.LastParameter()

    def startPoint(self: Mixin1DProtocol) -> Vector:
        """

        :return: a vector representing the start point of this edge

        Note, circles may have the start and end points the same
        """

        curve = self._geomAdaptor()
        umin = curve.FirstParameter()

        return Vector(curve.Value(umin))

    def endPoint(self: Mixin1DProtocol) -> Vector:
        """

        :return: a vector representing the end point of this edge.

        Note, circles may have the start and end points the same
        """

        curve = self._geomAdaptor()
        umax = curve.LastParameter()

        return Vector(curve.Value(umax))

    def paramAt(self: Mixin1DProtocol, d: float) -> float:
        """
        Compute parameter value at the specified normalized distance.

        :param d: normalized distance [0, 1]
        :return: parameter value
        """

        curve = self._geomAdaptor()

        l = GCPnts_AbscissaPoint.Length_s(curve)
        return GCPnts_AbscissaPoint(curve, l * d, curve.FirstParameter()).Parameter()

    def tangentAt(
        self: Mixin1DProtocol,
        locationParam: float = 0.5,
        mode: Literal["length", "parameter"] = "length",
    ) -> Vector:
        """
        Compute tangent vector at the specified location.

        :param locationParam: distance or parameter value (default: 0.5)
        :param mode: position calculation mode (default: parameter)
        :return: tangent vector
        """

        curve = self._geomAdaptor()

        tmp = gp_Pnt()
        res = gp_Vec()

        if mode == "length":
            param = self.paramAt(locationParam)
        else:
            param = locationParam

        curve.D1(param, tmp, res)

        return Vector(gp_Dir(res))

    def normal(self: Mixin1DProtocol) -> Vector:
        """
        Calculate the normal Vector. Only possible for planar curves.

        :return: normal vector
        """

        curve = self._geomAdaptor()
        gtype = self.geomType()

        if gtype == "CIRCLE":
            circ = curve.Circle()
            rv = Vector(circ.Axis().Direction())
        elif gtype == "ELLIPSE":
            ell = curve.Ellipse()
            rv = Vector(ell.Axis().Direction())
        else:
            fs = BRepLib_FindSurface(self.wrapped, OnlyPlane=True)
            surf = fs.Surface()

            if isinstance(surf, Geom_Plane):
                pln = surf.Pln()
                rv = Vector(pln.Axis().Direction())
            else:
                raise ValueError("Normal not defined")

        return rv

    def Center(self: Mixin1DProtocol) -> Vector:

        Properties = GProp_GProps()
        BRepGProp.LinearProperties_s(self.wrapped, Properties)

        return Vector(Properties.CentreOfMass())

    def Length(self: Mixin1DProtocol) -> float:

        return GCPnts_AbscissaPoint.Length_s(self._geomAdaptor())

    def radius(self: Mixin1DProtocol) -> float:
        """
        Calculate the radius.

        Note that when applied to a Wire, the radius is simply the radius of the first edge.

        :return: radius
        :raises ValueError: if kernel can not reduce the shape to a circular edge
        """
        geom = self._geomAdaptor()
        try:
            circ = geom.Circle()
        except (Standard_NoSuchObject, Standard_Failure) as e:
            raise ValueError("Shape could not be reduced to a circle") from e
        return circ.Radius()

    def IsClosed(self: Mixin1DProtocol) -> bool:

        return BRep_Tool.IsClosed_s(self.wrapped)

    def positionAt(
        self: Mixin1DProtocol,
        d: float,
        mode: Literal["length", "parameter"] = "length",
    ) -> Vector:
        """Generate a position along the underlying curve.

        :param d: distance or parameter value
        :param mode: position calculation mode (default: length)
        :return: A Vector on the underlying curve located at the specified d value.
        """

        curve = self._geomAdaptor()

        if mode == "length":
            param = self.paramAt(d)
        else:
            param = d

        return Vector(curve.Value(param))

    def positions(
        self: Mixin1DProtocol,
        ds: Iterable[float],
        mode: Literal["length", "parameter"] = "length",
    ) -> List[Vector]:
        """Generate positions along the underlying curve

        :param ds: distance or parameter values
        :param mode: position calculation mode (default: length)
        :return: A list of Vector objects.
        """

        return [self.positionAt(d, mode) for d in ds]

    def locationAt(
        self: Mixin1DProtocol,
        d: float,
        mode: Literal["length", "parameter"] = "length",
        frame: Literal["frenet", "corrected"] = "frenet",
        planar: bool = False,
    ) -> Location:
        """Generate a location along the underlying curve.

        :param d: distance or parameter value
        :param mode: position calculation mode (default: length)
        :param frame: moving frame calculation method (default: frenet)
        :param planar: planar mode
        :return: A Location object representing local coordinate system at the specified distance.
        """

        curve = self._geomAdaptor()

        if mode == "length":
            param = self.paramAt(d)
        else:
            param = d

        law: GeomFill_TrihedronLaw
        if frame == "frenet":
            law = GeomFill_Frenet()
        else:
            law = GeomFill_CorrectedFrenet()

        law.SetCurve(curve)

        tangent, normal, binormal = gp_Vec(), gp_Vec(), gp_Vec()

        law.D0(param, tangent, normal, binormal)
        pnt = curve.Value(param)

        T = gp_Trsf()
        if planar:
            T.SetTransformation(
                gp_Ax3(pnt, gp_Dir(0, 0, 1), gp_Dir(normal.XYZ())), gp_Ax3()
            )
        else:
            T.SetTransformation(
                gp_Ax3(pnt, gp_Dir(tangent.XYZ()), gp_Dir(normal.XYZ())), gp_Ax3()
            )

        return Location(TopLoc_Location(T))

    def locations(
        self: Mixin1DProtocol,
        ds: Iterable[float],
        mode: Literal["length", "parameter"] = "length",
        frame: Literal["frenet", "corrected"] = "frenet",
        planar: bool = False,
    ) -> List[Location]:
        """Generate location along the curve

        :param ds: distance or parameter values
        :param mode: position calculation mode (default: length)
        :param frame: moving frame calculation method (default: frenet)
        :param planar: planar mode
        :return: A list of Location objects representing local coordinate systems at the specified distances.
        """

        return [self.locationAt(d, mode, frame, planar) for d in ds]

    def project(
        self: T1D, face: "Face", d: VectorLike, closest: bool = True
    ) -> Union[T1D, List[T1D]]:
        """
        Project onto a face along the specified direction
        """

        bldr = BRepProj_Projection(self.wrapped, face.wrapped, Vector(d).toDir())
        shapes = Compound(bldr.Shape())

        # select the closest projection if requested
        rv: Union[T1D, List[T1D]]

        if closest:

            dist_calc = BRepExtrema_DistShapeShape()
            dist_calc.LoadS1(self.wrapped)

            min_dist = inf

            for el in shapes:
                dist_calc.LoadS2(el.wrapped)
                dist_calc.Perform()
                dist = dist_calc.Value()

                if dist < min_dist:
                    min_dist = dist
                    rv = tcast(T1D, el)

        else:
            rv = [tcast(T1D, el) for el in shapes]

        return rv


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
    def fillet(self: Any, radius: float, edgeList: Iterable[Edge]) -> Any:
        """
        Fillets the specified edges of this solid.

        :param radius: float > 0, the radius of the fillet
        :param edgeList:  a list of Edge objects, which must belong to this solid
        :return: Filleted solid
        """
        nativeEdges = [e.wrapped for e in edgeList]

        fillet_builder = BRepFilletAPI_MakeFillet(self.wrapped)

        for e in nativeEdges:
            fillet_builder.Add(radius, e)

        return self.__class__(fillet_builder.Shape())

    def chamfer(
        self: Any, length: float, length2: Optional[float], edgeList: Iterable[Edge]
    ) -> Any:
        """
        Chamfers the specified edges of this solid.

        :param length: length > 0, the length (length) of the chamfer
        :param length2: length2 > 0, optional parameter for asymmetrical chamfer. Should be `None` if not required.
        :param edgeList:  a list of Edge objects, which must belong to this solid
        :return: Chamfered solid
        """
        nativeEdges = [e.wrapped for e in edgeList]

        # make a edge --> faces mapping
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(
            self.wrapped, ta.TopAbs_EDGE, ta.TopAbs_FACE, edge_face_map
        )

        # note: we prefer 'length' word to 'radius' as opposed to FreeCAD's API
        chamfer_builder = BRepFilletAPI_MakeChamfer(self.wrapped)

        if length2:
            d1 = length
            d2 = length2
        else:
            d1 = length
            d2 = length

        for e in nativeEdges:
            face = edge_face_map.FindFromKey(e).First()
            chamfer_builder.Add(
                d1, d2, e, TopoDS.Face_s(face)
            )  # NB: edge_face_map return a generic TopoDS_Shape
        return self.__class__(chamfer_builder.Shape())

    def shell(
        self: Any,
        faceList: Optional[Iterable[Face]],
        thickness: float,
        tolerance: float = 0.0001,
        kind: Literal["arc", "intersection"] = "arc",
    ) -> Any:
        """
        Make a shelled solid of self.

        :param faceList: List of faces to be removed, which must be part of the solid. Can
          be an empty list.
        :param thickness: Floating point thickness. Positive shells outwards, negative
          shells inwards.
        :param tolerance: Modelling tolerance of the method, default=0.0001.
        :return: A shelled solid.
        """

        kind_dict = {
            "arc": GeomAbs_JoinType.GeomAbs_Arc,
            "intersection": GeomAbs_JoinType.GeomAbs_Intersection,
        }

        occ_faces_list = TopTools_ListOfShape()
        shell_builder = BRepOffsetAPI_MakeThickSolid()

        if faceList:
            for f in faceList:
                occ_faces_list.Append(f.wrapped)

        shell_builder.MakeThickSolidByJoin(
            self.wrapped,
            occ_faces_list,
            thickness,
            tolerance,
            Intersection=True,
            Join=kind_dict[kind],
        )
        shell_builder.Build()

        if faceList:
            rv = self.__class__(shell_builder.Shape())

        else:  # if no faces provided a watertight solid will be constructed
            s1 = self.__class__(shell_builder.Shape()).Shells()[0].wrapped
            s2 = self.Shells()[0].wrapped

            # s1 can be outer or inner shell depending on the thickness sign
            if thickness > 0:
                sol = BRepBuilderAPI_MakeSolid(s1, s2)
            else:
                sol = BRepBuilderAPI_MakeSolid(s2, s1)

            # fix needed for the orientations
            rv = self.__class__(sol.Shape()).fix()

        return rv

    def isInside(
        self: ShapeProtocol, point: VectorLike, tolerance: float = 1.0e-6
    ) -> bool:
        """
        Returns whether or not the point is inside a solid or compound
        object within the specified tolerance.

        :param point: tuple or Vector representing 3D point to be tested
        :param tolerance: tolerance for inside determination, default=1.0e-6
        :return: bool indicating whether or not point is within solid
        """
        if isinstance(point, Vector):
            point = point.toTuple()

        solid_classifier = BRepClass3d_SolidClassifier(self.wrapped)
        solid_classifier.Perform(gp_Pnt(*point), tolerance)

        return solid_classifier.State() == ta.TopAbs_IN or solid_classifier.IsOnAFace()

    @cqmultimethod
    def dprism(
        self: TS,
        basis: Optional[Face],
        profiles: List[Wire],
        depth: Optional[Real] = None,
        taper: Real = 0,
        upToFace: Optional[Face] = None,
        thruAll: bool = True,
        additive: bool = True,
    ) -> "Solid":
        """
        Make a prismatic feature (additive or subtractive)

        :param basis: face to perform the operation on
        :param profiles: list of profiles
        :param depth: depth of the cut or extrusion
        :param upToFace: a face to extrude until
        :param thruAll: cut thruAll
        :return: a Solid object
        """

        sorted_profiles = sortWiresByBuildOrder(profiles)
        faces = [Face.makeFromWires(p[0], p[1:]) for p in sorted_profiles]

        return self.dprism(basis, faces, depth, taper, upToFace, thruAll, additive)

    @dprism.register
    def dprism(
        self: TS,
        basis: Optional[Face],
        faces: List[Face],
        depth: Optional[Real] = None,
        taper: Real = 0,
        upToFace: Optional[Face] = None,
        thruAll: bool = True,
        additive: bool = True,
    ) -> "Solid":

        shape: Union[TopoDS_Shape, TopoDS_Solid] = self.wrapped
        for face in faces:
            feat = BRepFeat_MakeDPrism(
                shape,
                face.wrapped,
                basis.wrapped if basis else TopoDS_Face(),
                radians(taper),
                additive,
                False,
            )

            if upToFace is not None:
                feat.Perform(upToFace.wrapped)
            elif thruAll or depth is None:
                feat.PerformThruAll()
            else:
                feat.Perform(depth)

            shape = feat.Shape()

        return self.__class__(shape)


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

    def intersect(
        self, *toIntersect: "Shape", tol: Optional[float] = None
    ) -> "Compound":
        """
        Intersection of the positional arguments and this Shape.

        :param tol: Fuzzy mode tolerance
        """

        intersect_op = BRepAlgoAPI_Common()

        if tol:
            intersect_op.SetFuzzyValue(tol)

        return tcast(Compound, self._bool_op(self, toIntersect, intersect_op))

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


def edgesToWires(edges: Iterable[Edge], tol: float = 1e-6) -> List[Wire]:
    """
    Convert edges to a list of wires.
    """

    edges_in = TopTools_HSequenceOfShape()
    wires_out = TopTools_HSequenceOfShape()

    for e in edges:
        edges_in.Append(e.wrapped)

    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edges_in, tol, False, wires_out)

    return [Wire(el) for el in wires_out]
