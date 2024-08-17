from typing import IO, Optional, cast, Dict, Any
from typing_extensions import Literal

from ...cq import Workplane
from ..shapes import Shape, Compound

import logging
logger = logging.getLogger(__name__)

class ExportTypes:
    STEP = "STEP"

ExportLiterals = Literal[
    "STEP"
]

def toCompound(shape: Workplane) -> Compound:
    logger.info("exporters' toCompound() is called")

    return Compound.makeCompound(val for val in shape.vals() if isinstance(val, Shape))

def export(
    w: (Shape | Workplane),
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

    logger.info("exporters' export() is called")

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