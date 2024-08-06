from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("minimalcadquery")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.X.x"

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
)
from .occ_impl import exporters

from .cq import CQ, Workplane


__all__ = [
    "CQ",
    "Workplane",
    "plugins",
    "Plane",
    "Matrix",
    "Vector",
    "Location",
    "Shape",
    "Vertex",
    "Edge",
    "Wire",
    "Face",
    "Solid",
    "Shell",
    "Compound",
    "exporters",
]
