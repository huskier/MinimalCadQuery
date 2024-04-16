import tempfile
import os
import io as StringIO

from typing import IO, Optional, Union, cast, Dict, Any
from typing_extensions import Literal

from ...cq import Workplane
from ..shapes import Shape, Compound

class ExportTypes:
    STEP = "STEP"
    SVG = "SVG"

ExportLiterals = Literal[
    "STEP"
]

def toCompound(shape: Workplane) -> Compound:

    return Compound.makeCompound(val for val in shape.vals() if isinstance(val, Shape))

def export(
    w: Union[Shape, Workplane],
    fname: str,
    exportType: Optional[ExportLiterals] = None,
    tolerance: float = 0.1,
    angularTolerance: float = 0.1,
    opt: Optional[Dict[str, Any]] = None,
):

    """
    Export Workplane or Shape to file. Multiple entities are converted to compound.

    :param w:  Shape or Workplane to be exported.
    :param fname: output filename.
    :param exportType: the exportFormat to use. If None will be inferred from the extension. Default: None.
    :param tolerance: the deflection tolerance, in model units. Default 0.1.
    :param angularTolerance: the angular tolerance, in radians. Default 0.1.
    :param opt: additional options passed to the specific exporter. Default None.
    """

    shape: Shape
    f: IO

    if not opt:
        opt = {}

    if isinstance(w, Workplane):
        shape = toCompound(w)
    else:
        shape = w

    if exportType is None:
        t = fname.split(".")[-1].upper()
        if t in ExportTypes.__dict__.values():
            exportType = cast(ExportLiterals, t)
        else:
            raise ValueError("Unknown extensions, specify export type explicitly")

    if exportType == ExportTypes.STEP:
        shape.exportStep(fname, **opt)

    else:
        raise ValueError("Unknown export type")


#@deprecate()
def toString(shape, exportType, tolerance=0.1, angularTolerance=0.05):
    s = StringIO.StringIO()
    exportShape(shape, exportType, s, tolerance, angularTolerance)
    return s.getvalue()


#@deprecate()
def exportShape(
    w: Union[Shape, Workplane],
    exportType: ExportLiterals,
    fileLike: IO,
    tolerance: float = 0.1,
    angularTolerance: float = 0.1,
):
    """
    :param shape:  the shape to export. it can be a shape object, or a cadquery object. If a cadquery
    object, the first value is exported
    :param exportType: the exportFormat to use
    :param fileLike: a file like object to which the content will be written.
    The object should be already open and ready to write. The caller is responsible
    for closing the object
    :param tolerance: the linear tolerance, in model units. Default 0.1.
    :param angularTolerance: the angular tolerance, in radians. Default 0.1.
    """

    def tessellate(shape, angularTolerance):

        return shape.tessellate(tolerance, angularTolerance)

    shape: Shape
    if isinstance(w, Workplane):
        shape = toCompound(w)
    else:
        shape = w

    if exportType == ExportTypes.SVG:
        #fileLike.write(getSVG(shape))
        pass
    else:

        # all these types required writing to a file and then
        # re-reading. this is due to the fact that FreeCAD writes these
        (h, outFileName) = tempfile.mkstemp()
        # weird, but we need to close this file. the next step is going to write to
        # it from c code, so it needs to be closed.
        os.close(h)

        if exportType == ExportTypes.STEP:
            shape.exportStep(outFileName)
        else:
            raise ValueError("No idea how i got here")

        res = readAndDeleteFile(outFileName)
        fileLike.write(res)


#@deprecate()
def readAndDeleteFile(fileName):
    """
    Read data from file provided, and delete it when done
    return the contents as a string
    """
    res = ""
    with open(fileName, "r") as f:
        res = "{}".format(f.read())

    os.remove(fileName)
    return res
