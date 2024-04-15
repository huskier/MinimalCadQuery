from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cadquery")
except PackageNotFoundError:
    # package is not installed
    __version__ = "2.5-dev"

# these items point to the OCC implementation
from .occ_impl.geom import Plane, Vector, Matrix, Location
from .occ_impl.shapes import (
    Shape,
    Vertex,
    Edge,
    Face,
    Wire,
    Solid,
    Shell,
    Compound,
    sortWiresByBuildOrder,
)
from .occ_impl import exporters
from .occ_impl import importers

# these items are the common implementation

# the order of these matter
from .selectors import (
    StringSyntaxSelector,
    Selector,
)
#from .sketch import Sketch
from .cq import CQ, Workplane
#from .assembly import Assembly, Color, Constraint
from . import selectors
#from . import plugins


__all__ = [
    "CQ",
    "Workplane",
    "plugins",
    "selectors",
    "Plane",
    "Matrix",
    "Vector",
    "Location",
    "sortWiresByBuildOrder",
    "Shape",
    "Vertex",
    "Edge",
    "Wire",
    "Face",
    "Solid",
    "Shell",
    "Compound",
    "exporters",
    "importers",
    "StringSyntaxSelector",
    "Selector",
#    "plugins",
]
