#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from pypgx.api._pgx_graph import PgxGraph
from pypgx._utils.error_handling import java_handler, java_caster
from pypgx._utils.item_converter import convert_to_java_type
from pypgx._utils.error_messages import UNHASHABLE_TYPE
from typing import Any, Optional, Union, TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession


class GraphBuilder:
    """A graph builder for constructing a :class:`PgxGraph`."""

    _java_class = 'oracle.pgx.api.GraphBuilder'

    def __init__(self, session: "PgxSession", java_graph_builder, id_type: str) -> None:
        self._builder = java_graph_builder
        self.session = session
        self.id_type = id_type

    def add_vertex(self, vertex: Optional[Union[str, int]] = None) -> "VertexBuilder":
        """Add the vertex with the given id to the graph builder.

        If the vertex doesn't exist it is added, if it exists a builder for that vertex is
        returned Throws an UnsupportedOperationException if vertex ID generation strategy is set
        to IdGenerationStrategy.AUTO_GENERATED.

        :param vertex: The ID of the new vertex
        :returns: A vertexBuilder instance
        """
        if vertex is None:
            vb = java_handler(self._builder.addVertex, [])
        else:
            vb = java_caster(self._builder.addVertex, (vertex, self.id_type))
        return VertexBuilder(self.session, vb, self.id_type)

    def reset_vertex(self, vertex: Union["VertexBuilder", str, int]):
        """Reset any change for the given vertex.

        :param vertex: The id or the vertexBuilder object to reset
        :returns: self
        """
        if isinstance(vertex, VertexBuilder):
            vertex = vertex._builder.getId()

        java_caster(self._builder.resetVertex, (vertex, self.id_type))
        return self

    def reset_edge(self, edge: Union["EdgeBuilder", int]):
        """Reset any change for the given edge.

        :param edge: The id or the EdgeBuilder object to reset
        :returns: self
        """
        if isinstance(edge, EdgeBuilder):
            edge = edge._builder.getId()

        java_handler(self._builder.resetEdge, [edge])
        return self

    def add_edge(
        self,
        src: Union["VertexBuilder", int],
        dst: Union["VertexBuilder", int],
        edge_id: Optional[int] = None,
    ) -> "EdgeBuilder":
        """
        :param src: Source vertexBuilder or id
        :param dst: Destination VertexBuilder or ID
        :param edge_id: the ID of the new edge
        """
        if isinstance(src, VertexBuilder):
            src = src._builder

        if isinstance(dst, VertexBuilder):
            dst = dst._builder

        eb = java_handler(
            self._builder.addEdge, [edge_id, src, dst] if edge_id is not None else [src, dst]
        )
        return EdgeBuilder(self.session, eb, self.id_type)

    def build(self, name: Optional[str] = None) -> PgxGraph:
        """
        :param name: The new name of the graph. If None, a name is generated.
        :return: PgxGraph object
        """
        graph = java_handler(self._builder.build, [name])
        return PgxGraph(self.session, graph)

    def __repr__(self) -> str:
        s = self._builder.toString()
        changes = s[s.find('with added') :]
        return "{}(session id: {}, {})".format(self.__class__.__name__, self.session.id, changes)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class VertexBuilder(GraphBuilder):
    """A vertex builder for defining vertices added with the :class:`GraphBuilder`."""

    _java_class = 'oracle.pgx.api.VertexBuilder'

    def __init__(self, session: "PgxSession", java_vertex_builder, id_type: str) -> None:
        super().__init__(session, java_vertex_builder, id_type)
        self._builder = java_vertex_builder

    def set_property(self, key: str, value: Any) -> "VertexBuilder":
        """Set the property value of this vertex with the given key to the given value.

        The first time this method is called, the type of value defines the type of the property.

        :param key: The property key
        :param value: The value of the vertex property
        :returns: The VertexProperty object
        """
        value = convert_to_java_type(value)
        java_handler(self._builder.setProperty, [key, value])
        return self

    def add_label(self, label: str) -> "VertexBuilder":
        """Add the given label to this vertex.

        :param label: The label to be added.
        :returns: The VertexProperty object
        """
        java_handler(self._builder.addLabel, [label])
        return self

    @property
    def id(self) -> Union[str, int]:
        """Get the id of the element (vertex or edge) this builder belongs to."""
        return self._builder.getId()

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class EdgeBuilder(GraphBuilder):
    """An edge builder for defining edges added with the :class:`GraphBuilder`."""

    _java_class = 'oracle.pgx.api.EdgeBuilder'

    def __init__(self, session: "PgxSession", java_edge_builder, id_type: str) -> None:
        super().__init__(session, java_edge_builder, id_type)
        self._builder = java_edge_builder

    def set_property(self, key: str, value: Any) -> "EdgeBuilder":
        """Set the property value of this edge with the given key to the given value.

        The first time this method is called, the type of value defines the type of the property.

        :param key: The property key
        :param value: The value of the vertex property
        :returns: The EdgeBuilder object
        """
        value = convert_to_java_type(value)
        java_handler(self._builder.setProperty, [key, value])
        return self

    def set_label(self, label: str) -> "EdgeBuilder":
        """Set the new value of the label.

        :param label: The new value of the label
        :returns: The EdgeBuilder object
        """
        java_handler(self._builder.setLabel, [label])
        return self

    @property
    def id(self) -> int:
        """Get the id of the element (vertex or edge) this builder belongs to."""
        return self._builder.getId()

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
