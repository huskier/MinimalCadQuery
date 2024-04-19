"""
    Copyright (C) 2011-2015  Parametric Products Intellectual Holdings, LLC

    This file is part of CadQuery.

    CadQuery is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    CadQuery is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; If not, see <http://www.gnu.org/licenses/>
"""

from abc import abstractmethod, ABC
import math
from .occ_impl.geom import Vector
from .occ_impl.shape_protocols import (
    ShapeProtocol,
    Shape1DProtocol,
    FaceProtocol,
    geom_LUT_EDGE,
    geom_LUT_FACE,
)
from pyparsing import (
    pyparsing_common,
    Literal,
    Word,
    nums,
    Optional,
    Combine,
    oneOf,
    Group,
    infixNotation,
    opAssoc,
)
from functools import reduce
from typing import Iterable, List, Sequence, TypeVar, cast

Shape = TypeVar("Shape", bound=ShapeProtocol)

class Selector(object):
    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        print("In Selector's filter() function......")
        return list(objectList)

class StringSyntaxSelector(Selector):
    def __init__(self, selectorString):
        pass

