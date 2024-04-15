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


def importShape(
    importType: Literal["STEP", "DXF", "BREP"], fileName: str, *args, **kwargs
) -> "cq.Workplane":
    """
    Imports a file based on the type (STEP, STL, etc)

    :param importType: The type of file that we're importing
    :param fileName: The name of the file that we're importing
    """

    # Check to see what type of file we're working with
    if importType == ImportTypes.STEP:
        return importStep(fileName)
    #elif importType == ImportTypes.DXF:
    #    return importDXF(fileName, *args, **kwargs)
    #elif importType == ImportTypes.BREP:
    #    return importBrep(fileName)
    else:
        raise RuntimeError("Unsupported import type: {!r}".format(importType))

# Loads a STEP file into a CQ.Workplane object
def importStep(fileName: str) -> "cq.Workplane":
    """
    Accepts a file name and loads the STEP file into a cadquery Workplane

    :param fileName: The path and name of the STEP file to be imported
    """

    # Now read and return the shape
    reader = STEPControl_Reader()
    readStatus = reader.ReadFile(fileName)
    if readStatus != OCP.IFSelect.IFSelect_RetDone:
        raise ValueError("STEP File could not be loaded")
    for i in range(reader.NbRootsForTransfer()):
        reader.TransferRoot(i + 1)

    occ_shapes = []
    for i in range(reader.NbShapes()):
        occ_shapes.append(reader.Shape(i + 1))

    # Make sure that we extract all the solids
    solids = []
    for shape in occ_shapes:
        solids.append(Shape.cast(shape))

    return cq.Workplane("XY").newObject(solids)