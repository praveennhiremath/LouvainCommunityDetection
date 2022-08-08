#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from collections.abc import Sequence
from itertools import islice

from pypgx.api._pgx_id import PgxId
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx.api._pgx_entity import PgxEdge, PgxVertex, PgxEntity
from pypgx._utils.error_handling import java_handler, java_caster
from pypgx._utils.error_messages import (
    COMPARE_VECTOR,
    INVALID_TYPE_OR_ITERABLE_TYPE,
    INVALID_OPTION,
)
from pypgx._utils.item_converter import convert_to_java_type, convert_to_python_type
from pypgx._utils.pgx_types import property_types
from pypgx._utils.pyjnius_helper import PyjniusHelper
from pypgx.api._pgx_map import PgxMap
from typing import Any, Iterator, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class PgxProperty(PgxContextManager):
    """A property of a `PgxGraph`.

    .. note: This is a base class of :class:`VertexProperty` and :class:`EdgeProperty`,
       and is not instantiated on its own.
    """

    _java_class = 'oracle.pgx.api.Property'

    def __init__(self, graph: "PgxGraph", java_prop) -> None:
        self._prop = java_prop
        self.name = java_prop.getName()
        self.entity_type = java_prop.getEntityType().toString()
        self.type = java_prop.getType().toString()
        self.is_transient = java_prop.isTransient()
        self.dimension = java_prop.getDimension()
        self.size = java_prop.size()
        self.graph = graph
        self.is_vector_property = self.dimension > 0

    @property
    def is_published(self) -> bool:
        """Check if this property is published.

        :return: `True` if this property is published, `False` otherwise.
        :rtype: bool
        """
        return self._prop.isPublished()

    def publish(self) -> None:
        """Publish the property into a shared graph so it can be shared between sessions.

        :return: None
        """
        java_handler(self._prop.publish, [])

    def rename(self, name: str) -> None:
        """Rename this property.

        :param name: New name
        :return: None
        """
        java_handler(self._prop.rename, [name])
        self.name = name

    def clone(self, name: Optional[str] = None) -> "PgxProperty":
        """Create a copy of this property.

        :param name: name of copy to be created. If `None`, guaranteed unique name will be
            generated.
        :return: property result
        :rtype: this class
        """
        cloned_prop = java_handler(self._prop.clone, [name])
        return self.__class__(self.graph, cloned_prop)

    def get_top_k_values(self, k: int) -> List[Tuple[PgxEntity, Any]]:
        """Get the top k vertex/edge value pairs according to their value.

        :param k: How many top values to retrieve, must be in the range between 0 and number of
            nodes/edges (inclusive)
        :return: list of `k` key-value tuples where the keys vertices/edges and the values are
            property values, sorted in ascending order
        :rtype: list of tuple(PgxVertex or PgxEdge, Any)

        """
        if self.dimension > 0:
            raise RuntimeError(COMPARE_VECTOR)
        else:
            top_k = java_handler(self._prop.getTopKValues, [k])
            it = top_k.iterator()
            return list(
                (
                    convert_to_python_type(item.getKey(), self.graph),
                    convert_to_python_type(item.getValue(), self.graph),
                )
                for item in islice(it, 0, k)
            )

    def get_bottom_k_values(self, k: int) -> List[Tuple[PgxEntity, Any]]:
        """Get the bottom k vertex/edge value pairs according to their value.

        :param k: How many top values to retrieve, must be in the range between 0 and number of
            nodes/edges (inclusive)
        """
        if self.dimension > 0:
            raise RuntimeError(COMPARE_VECTOR)
        else:
            bottom_k = java_handler(self._prop.getBottomKValues, [k])
            it = bottom_k.iterator()
            return list(
                (
                    convert_to_python_type(item.getKey(), self.graph),
                    convert_to_python_type(item.getValue(), self.graph),
                )
                for item in islice(it, 0, k)
            )

    def get_values(self) -> List[Tuple[PgxEntity, Any]]:
        """Get the values of this property as a list."""
        return list(self)

    def fill(self, value: Any) -> None:
        """Fill this property with a given value.

        :param value: The value
        """
        if self.dimension > 0:
            vec = None
            if self.entity_type == 'vertex':
                vec = self._prop.get(self.graph.get_random_vertex().id)
            else:
                vec = self._prop.get(self.graph.get_random_edge().id)
            t = type(vec.get(0))
            if isinstance(value, Sequence) and self.dimension == len(value):
                for idx in range(vec.getDimension()):
                    java_caster(vec.set, (idx, 'integer'), (value[idx], self.type))
            elif isinstance(value, t):
                for idx in range(vec.getDimension()):
                    java_caster(vec.set, (idx, 'integer'), (value, self.type))
            else:
                raise TypeError(
                    INVALID_TYPE_OR_ITERABLE_TYPE.format(
                        var='value', type=t.__name__, size=self.dimension
                    )
                )
            java_handler(self._prop.fill, [vec])
        else:
            value = convert_to_java_type(value)
            java_handler(self._prop.fill, [value])

    def expand(self) -> Union["PgxProperty", List["PgxProperty"]]:
        """If this is a vector property, expands this property into a list of scalar properties of
        same type.

        The first property will contain the first element of the vector, the second property the
        second element and so on.
        """
        if self.dimension > 0:
            expanded = java_handler(self._prop.expand, [])
            expanded_list = []
            for p in expanded:
                expanded_list.append(self.__class__(self.graph, p))
            return expanded_list
        else:
            return self

    def close(self) -> None:
        """Free resources on the server taken up by this Property.

        :return: None
        """
        java_handler(self._prop.close, [])

    def get_property_id(self) -> PgxId:
        """Get an internal identifier for this property.

        Only meant for internal usage.

        :return: the internal identifier of this property
        """
        return PgxId(self._prop.getPropertyId())

    def wrap(self, property_value: Any, property_type: str) -> Any:
        """Take a property value and wraps it pgx entities if applicable

        :param property_value: property value
        :type property_value: Any
        :param property_type: A valid property type.
        :type property_type: str
        """
        if property_type not in property_types.keys():
            raise ValueError(
                INVALID_OPTION.format(var='property_type', opts=list(property_types.keys()))
            )
        if isinstance(property_value, (int, str)) and property_type == 'vertex':
            item = java_handler(self.graph._graph.getVertex, [property_value])
            return convert_to_python_type(item, self.graph)
        elif isinstance(property_value, (int, str)) and property_type == 'edge':
            item = java_handler(self.graph._graph.getEdge, [property_value])
            return convert_to_python_type(item, self.graph)
        else:
            return property_value

    def is_vector_property(self) -> bool:
        """Return True if it is a vector property, False otherwise"""
        return self._prop.isVectorProperty()

    def destroy(self) -> None:
        """Free resources on the server taken up by this Property.

        :return: None
        """
        java_handler(self._prop.destroy, [])

    def __iter__(self) -> Iterator[Any]:
        it = self._prop.getValues().iterator()
        if self.dimension > 0:
            return (
                (convert_to_python_type(item.getKey(), self.graph), list(item.getValue().toArray()))
                for item in islice(it, 0, self.size)
            )
        else:
            return (
                (
                    convert_to_python_type(item.getKey(), self.graph),
                    convert_to_python_type(item.getValue(), self.graph),
                )
                for item in islice(it, 0, self.size)
            )

    def __getitem__(self, key: Union[slice, PgxEntity, int, str]) -> Any:
        if isinstance(key, slice):
            it = self._prop.getValues().iterator()
            if self.dimension > 0:
                return list(
                    (
                        convert_to_python_type(item.getKey(), self.graph),
                        list(item.getValue().toArray()),
                    )
                    for item in islice(it, key.start, key.stop, key.step)
                )
            else:
                return list(
                    (
                        convert_to_python_type(item.getKey(), self.graph),
                        convert_to_python_type(item.getValue(), self.graph),
                    )
                    for item in islice(it, key.start, key.stop, key.step)
                )
        else:
            return self.get(key)

    def __setitem__(self, key: Union[PgxEntity, int, str], value: Any) -> None:
        self.set(key, value)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "{}(name: {}, type: {}, graph: {})".format(
            self.__class__.__name__, self.name, self.type, self.graph.name
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((str(self), str(self.graph.name)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._prop.equals(other._prop)


class VertexProperty(PgxProperty):
    """A vertex property of a :class:`PgxGraph`."""

    _java_class = 'oracle.pgx.api.VertexProperty'

    def __init__(self, graph: "PgxGraph", java_prop) -> None:
        super().__init__(graph, java_prop)

    @staticmethod
    def _from_java(java_prop):
        # need to import here to avoid import loop
        from pypgx.api._pgx_session import PgxSession
        from pypgx.api._pgx_graph import PgxGraph

        java_graph = java_handler(java_prop.getGraph, [])
        java_session = java_handler(java_graph.getSession, [])
        graph = PgxGraph(PgxSession(java_session), java_graph)
        return VertexProperty(graph, java_prop)

    def set(self, key: Union[PgxVertex, int, str], value: Any) -> None:
        """Set a property value.

        :param key: The key (vertex/edge) whose property to set
        :param value: The property value
        """
        if isinstance(key, (int, str)):
            key = self.graph.get_vertex(key)
        key = convert_to_java_type(key)
        if self.dimension > 0:
            vec = java_handler(PyjniusHelper.getFromPropertyByKey, [self._prop, key])
            t = type(vec.get(0))
            if isinstance(value, Sequence) and self.dimension == len(value):
                for idx, v in enumerate(value):
                    java_caster(vec.set, (idx, None), (v, self.type))
            elif isinstance(value, t):
                for idx in range(vec.getDimension()):
                    java_caster(vec.set, (idx, 'integer'), (value, self.type))
            else:
                raise TypeError(
                    INVALID_TYPE_OR_ITERABLE_TYPE.format(
                        var='value', type=t.__name__, size=self.dimension
                    )
                )
            java_handler(self._prop.set, [key, vec])
        else:
            value = convert_to_java_type(value)
            java_caster(self._prop.set, (key, None), (value, self.type))

    def set_values(self, values: PgxMap) -> None:
        """Set the labels values.

        :param values: pgxmap with ids and values
        :type values: PgxMap
        """
        values_keys = values.keys()
        for key in values_keys:
            self.set(key, values.get(key))

    def get(self, key: Union[PgxVertex, int, str]) -> Any:
        """
        :param key: The key (vertex/edge) whose property to get
        """
        if isinstance(key, (int, str)):
            key = self.graph.get_vertex(key)
        key = convert_to_java_type(key)
        value = java_handler(PyjniusHelper.getFromPropertyByKey, [self._prop, key])
        if self.dimension > 0:
            return list(value.toArray())
        else:
            return convert_to_python_type(value, self.graph)


class EdgeProperty(PgxProperty):
    """An edge property of a :class:`PgxGraph`."""

    _java_class = 'oracle.pgx.api.EdgeProperty'

    def __init__(self, graph: "PgxGraph", java_prop) -> None:
        super().__init__(graph, java_prop)

    def set(self, key: Union[PgxEdge, int], value: Any) -> None:
        """Set a property value.

        :param key: The key (vertex/edge) whose property to set
        :param value: The property value
        """
        if isinstance(key, (int, str)):
            key = self.graph.get_edge(key)
        key = convert_to_java_type(key)
        if self.dimension > 0:
            vec = java_handler(PyjniusHelper.getFromPropertyByKey, [self._prop, key])
            t = type(vec.get(0))
            if isinstance(value, Sequence) and self.dimension == len(value):
                for idx in range(vec.getDimension()):
                    java_handler(vec.set, [idx, value[idx]])
            elif isinstance(value, t):
                for idx in range(vec.getDimension()):
                    java_handler(vec.set, [idx, value])
            else:
                raise TypeError(
                    INVALID_TYPE_OR_ITERABLE_TYPE.format(
                        var='value', type=t.__name__, size=self.dimension
                    )
                )
            java_handler(self._prop.set, [key, vec])
        else:
            value = convert_to_java_type(value)
            java_handler(self._prop.set, [key, value])

    def set_values(self, values: PgxMap) -> None:
        """Set the labels values.

        :param values: pgxmap with ids and values
        :type values: PgxMap
        """
        values_keys = values.keys()
        for key in values_keys:
            self.set(key, values.get(key))

    def get(self, key: Union[PgxEdge, int]) -> Any:
        """
        :param key: The key (vertex/edge) whose property to get
        """
        if isinstance(key, (int, str)):
            key = self.graph.get_edge(key)
        key = convert_to_java_type(key)
        value = java_handler(PyjniusHelper.getFromPropertyByKey, [self._prop, key])
        if self.dimension > 0:
            return list(value.toArray())
        else:
            return convert_to_python_type(value, self.graph)


class VertexLabels(VertexProperty):
    """Class for storing labels for vertices.

    A vertex can have multiple labels. In effect this is a :class:`VertexProperty`
    where a set of strings is associated to each vertex.
    """

    _java_class = 'oracle.pgx.api.VertexLabels'

    def __init__(self, graph: "PgxGraph", java_labels) -> None:
        self._labels = java_labels
        super().__init__(graph, java_labels)

    def __getitem__(
        self, key: Union[slice, PgxVertex, int, str]
    ) -> Union[List[Tuple[PgxVertex, Set[str]]], Set[str]]:
        if isinstance(key, slice):
            it = self._labels.getValues().iterator()
            return list(
                (PgxVertex(self.graph, item.getKey()), set(item.getValue()))
                for item in islice(it, key.start, key.stop, key.step)
            )
        else:
            return set(self.get(key))

    def get_values(self) -> List[Tuple[PgxVertex, Set[str]]]:
        """Get the values of this label as a list.

        :return: a list of key-value tuples, where each key is a vertex and each key is the set of
            labels assigned to that vertex
        :rtype: list of tuple(PgxVertex, set of str)
        """
        return list(self)

    def __iter__(self) -> Iterator[Tuple[PgxVertex, Set[str]]]:
        it = self._labels.getValues().iterator()
        return (
            (PgxVertex(self.graph, item.getKey()), set(item.getValue()))
            for item in islice(it, 0, self.size)
        )


class EdgeLabel(EdgeProperty):
    """Class for storing a label type edge property."""

    _java_class = 'oracle.pgx.api.EdgeLabel'

    def __init__(self, graph: "PgxGraph", java_label) -> None:
        self._label = java_label
        super().__init__(graph, java_label)
