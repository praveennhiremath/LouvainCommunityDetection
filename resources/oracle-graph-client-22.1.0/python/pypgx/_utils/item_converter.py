#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from datetime import date, time, datetime

import pypgx._utils.pgx_types as types

from jnius import autoclass
import collections.abc
from collections.abc import Mapping, Sequence
from pypgx._utils.error_messages import ARG_MUST_BE, VERTEX_ID_OR_PGXVERTEX
from typing import Any, Optional, TYPE_CHECKING, Union, Iterable

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph
    from pypgx.api._pgx_entity import PgxVertex

Boolean = autoclass('java.lang.Boolean')
Double = autoclass('java.lang.Double')
Long = autoclass('java.lang.Long')
String = autoclass('java.lang.String')
Map = autoclass('java.util.HashMap')


def convert_to_python_type(item: Any, graph: Optional["PgxGraph"] = None) -> Any:
    """Convert a Java value to a Python value.

    The typical use case for this function is converting a Java object whose type is not known
    until runtime. For example, an element from a PGQL result set.
    """

    import pypgx.api._pgx_entity as entity
    from pypgx.api._pgx_graph import PgxGraph

    if isinstance(item, types.pgx_entities['vertex']):
        if graph is None:
            raise ValueError("Graph must be set if the item type is PgxVertex")
        elif not isinstance(graph, PgxGraph):
            raise TypeError(ARG_MUST_BE.format(arg='graph', type=PgxGraph))
        return entity.PgxVertex(graph, item)
    elif isinstance(item, types.pgx_entities['edge']):
        if graph is None:
            raise ValueError("Graph must be set if the item type is PgxEdge")
        elif not isinstance(graph, PgxGraph):
            raise TypeError(ARG_MUST_BE.format(arg='graph', type=PgxGraph))
        return entity.PgxEdge(graph, item)
    elif isinstance(item, types.local_date):
        return datetime.strptime(item.toString(), '%Y-%m-%d').date()
    elif isinstance(item, types.local_time):
        # Format may or may not have milliseconds in the string. Test for both.
        try:
            return datetime.strptime(item.toString(), '%H:%M:%S.%f').time()
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%H:%M:%S').time()
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%H:%M').time()
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%H').time()
        except ValueError:
            raise ValueError(item.toString() + " cannot be parsed into datetime")
    elif isinstance(item, types.timestamp):
        # Format may or may not have milliseconds in the string. Test for both.
        try:
            return datetime.strptime(item.toString(), '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%Y-%m-%dT%H:%M')
        except ValueError:
            pass
        try:
            return datetime.strptime(item.toString(), '%Y-%m-%dT%H')
        except ValueError:
            raise ValueError(item.toString() + " cannot be parsed into datetime")
    elif isinstance(item, types.time_with_timezone):
        # Adjust timezone to be readable by .strptime()
        item_str = item.toString()
        if item_str[-1] == 'Z':
            item_str = item_str[:-1] + '+0000'
        else:
            item_str = item_str[:-3] + item_str[-2:]
        # Format may or may not have milliseconds in the string. Test for both.
        try:
            return datetime.strptime(item_str, '%H:%M:%S.%f%z').timetz()
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%H:%M:%S%z').timetz()
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%H:%M%z').timetz()
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%H%z').timetz()
        except ValueError:
            raise ValueError(item.toString() + " cannot be parsed into datetime")
    elif isinstance(item, types.timestamp_with_timezone):
        # Adjust timezone to be readable by .strptime()
        item_str = item.toString()
        if item_str[-1] == 'Z':
            item_str = item_str[:-1] + '+0000'
        else:
            item_str = item_str[:-3] + item_str[-2:]
        # Format may or may not have milliseconds in the string. Test for both.
        try:
            return datetime.strptime(item_str, '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%Y-%m-%dT%H:%M%z')
        except ValueError:
            pass
        try:
            return datetime.strptime(item_str, '%Y-%m-%dT%H%z')
        except ValueError:
            raise ValueError(item.toString() + " cannot be parsed into datetime")
    elif isinstance(item, types.legacy_date):
        return datetime.strptime(item.toString(), '%a %b %d %H:%M:%S %Z %Y')
    elif isinstance(item, types.point2d):
        return item.getX(), item.getY()
    elif isinstance(item, types.java_set):
        # Note: a set will only be converted correctly if its elements are automatically converted
        # correctly (e.g. strings).
        return set(item.toArray())
    elif isinstance(item, types.java_list):
        return [convert_to_python_type(element) for element in item]
    else:
        return item


def convert_to_java_type(item: Any) -> Any:
    import pypgx.api._pgx_entity as entity

    if isinstance(item, entity.PgxVertex):
        return item._vertex
    elif isinstance(item, entity.PgxEdge):
        return item._edge
    elif isinstance(item, date) and not isinstance(item, datetime):
        return types.local_date.parse(String(item.isoformat()))
    elif isinstance(item, time):
        if item.tzinfo:
            return types.time_with_timezone.parse(String(item.isoformat()))
        else:
            return types.local_time.parse(String(item.isoformat()))
    elif isinstance(item, datetime):
        if item.tzinfo:
            return types.timestamp_with_timezone.parse(String(item.isoformat()))
        else:
            return types.timestamp.parse(String(item.isoformat()))
    elif isinstance(item, float):
        return Double(item)
    elif isinstance(item, int) and (item > 2147483647 or item < -2147483648):
        # Converting a python int which doesn't fit in a java int will result
        # in an error, so we need to convert it to long
        return Long(item)
    elif isinstance(item, bool):
        return Boolean(item)
    else:
        return item


def convert_to_java_list(sequence: Sequence):
    if isinstance(sequence, Sequence):
        java_list = types.array_list()
        for item in sequence:
            java_list.add(item)
        return java_list
    raise TypeError('Argument must be a Sequence')


def convert_to_java_map(dict_like: Mapping) -> Map:
    if isinstance(dict_like, Mapping):
        java_map = Map()
        for key in dict_like:
            java_map.put(key, dict_like[key])
        return java_map
    raise TypeError('Value must be dictionary-like')


def convert_python_to_java_vertex(
    graph: "PgxGraph", vertex: Union["PgxVertex", int]
) -> Any:
    import pypgx.api._pgx_entity as entity  # Avoid circular imports

    if isinstance(vertex, entity.PgxVertex):
        return vertex._vertex
    elif isinstance(vertex, int):
        return graph.get_vertex(vertex)._vertex
    else:
        raise TypeError(VERTEX_ID_OR_PGXVERTEX.format(var='vertices'))


def convert_python_to_java_vertex_list(
    graph: "PgxGraph",
    vertices: Iterable[Union["PgxVertex", int]],
) -> Any:
    if isinstance(vertices, collections.abc.Iterable):
        # Convert `vertices` from a Python iterable to a Java ArrayList.
        vids = autoclass('java.util.ArrayList')()
        for vertex in vertices:
            vids.add(convert_python_to_java_vertex(graph, vertex))
        return vids
    raise TypeError('Vertices has to be a list.')
