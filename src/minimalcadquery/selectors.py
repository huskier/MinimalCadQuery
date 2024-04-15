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
    """
    Filters a list of objects.

    Filters must provide a single method that filters objects.
    """

    def filter(self, objectList: Sequence[Shape]) -> List[Shape]:
        """
        Filter the provided list.

        The default implementation returns the original list unfiltered.

        :param objectList: list to filter
        :type objectList: list of OCCT primitives
        :return: filtered list
        """
        return list(objectList)

class StringSyntaxSelector(Selector):
    r"""
    Filter lists objects using a simple string syntax. All of the filters available in the string syntax
    are also available ( usually with more functionality ) through the creation of full-fledged
    selector objects. see :py:class:`Selector` and its subclasses

    Filtering works differently depending on the type of object list being filtered.

    :param selectorString: A two-part selector string, [selector][axis]

    :return: objects that match the specified selector

    ***Modifiers*** are ``('|','+','-','<','>','%')``

        :\|:
            parallel to ( same as :py:class:`ParallelDirSelector` ). Can return multiple objects.
        :#:
            perpendicular to (same as :py:class:`PerpendicularDirSelector` )
        :+:
            positive direction (same as :py:class:`DirectionSelector` )
        :-:
            negative direction (same as :py:class:`DirectionSelector`  )
        :>:
            maximize (same as :py:class:`DirectionMinMaxSelector` with directionMax=True)
        :<:
            minimize (same as :py:class:`DirectionMinMaxSelector` with directionMax=False )
        :%:
            curve/surface type (same as :py:class:`TypeSelector`)

    ***axisStrings*** are: ``X,Y,Z,XY,YZ,XZ`` or ``(x,y,z)`` which defines an arbitrary direction

    It is possible to combine simple selectors together using logical operations.
    The following operations are supported

        :and:
            Logical AND, e.g. >X and >Y
        :or:
            Logical OR, e.g. \|X or \|Y
        :not:
            Logical NOT, e.g. not #XY
        :exc(ept):
            Set difference (equivalent to AND NOT): \|X exc >Z

    Finally, it is also possible to use even more complex expressions with nesting
    and arbitrary number of terms, e.g.

        (not >X[0] and #XY) or >XY[0]

    Selectors are a complex topic: see :ref:`selector_reference` for more information
    """

    def __init__(self, selectorString):
        """
        Feed the input string through the parser and construct an relevant complex selector object
        """
        '''
        self.selectorString = selectorString
        parse_result = _expression_grammar.parseString(selectorString, parseAll=True)
        self.mySelector = parse_result.asList()[0]
        '''
        pass

