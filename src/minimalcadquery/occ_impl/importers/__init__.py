from math import pi
from typing import List, Literal

import OCP.IFSelect
from OCP.STEPControl import STEPControl_Reader

from ... import cq
from ..shapes import Shape

RAD2DEG = 360.0 / (2 * pi)


class ImportTypes:
    STEP = "STEP"
    DXF = "DXF"
    BREP = "BREP"


class UNITS:
    MM = "mm"
    IN = "in"

# Loads a STEP file into a CQ.Workplane object
def importStep(fileName: str) -> "cq.Workplane":
    """
    Accepts a file name and loads the STEP file into a cadquery Workplane

    :param fileName: The path and name of the STEP file to be imported
    """

    pass