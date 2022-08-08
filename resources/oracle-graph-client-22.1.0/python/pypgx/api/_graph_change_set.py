#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from pypgx.api._graph_builder import EdgeBuilder, GraphBuilder, VertexBuilder
from pypgx.api._pgx_graph import PgxGraph
from pypgx._utils.error_handling import java_caster, java_handler
from pypgx._utils.error_messages import UNHASHABLE_TYPE
from typing import Optional, Union, Any, TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession


class GraphChangeSet(GraphBuilder):
    """Class which stores changes of a particular graph."""

    _java_class = 'oracle.pgx.api.GraphChangeSet'

    def __init__(
        self, session: "PgxSession", java_graph_change_set, id_type: str = 'integer'
    ) -> None:
        """Construct a new change set.

        :param session: A 'PgxSession' object.
        :param java_graph_change_set: An instance of the corresponding java
            'GraphChangeSet' interface.
        :param id_type: A string describing the type of the ids. Optional.
            Defaults to 'integer'.
        """
        super().__init__(session, java_graph_change_set, id_type)
        self.session = session
        self.id_type = id_type
        self._change_set = java_graph_change_set

    def add_edge(
        self,
        src_vertex: Union[int, VertexBuilder],
        dst_vertex: Union[int, VertexBuilder],
        edge_id: Optional[int] = None,
    ) -> EdgeBuilder:
        """Add an edge with the given edge ID and the given source and destination vertices.

        :param src_vertex: source vertexBuilder or id.
        :param dst_vertex: Destination VertexBuilder or ID.
        :param edge_id: the ID of the new edge. Optional. Defaults to 'None'.
        :returns: An 'EdgeBuilder' instance containing the added edge.
        """
        if isinstance(src_vertex, VertexBuilder):
            src_vertex = src_vertex._builder.getId()

        if isinstance(dst_vertex, VertexBuilder):
            dst_vertex = dst_vertex._builder.getId()

        if edge_id is not None:
            eb = java_caster(
                self._change_set.addEdge,
                *[(edge_id, None), (src_vertex, self.id_type), (dst_vertex, self.id_type)]
            )

        else:
            eb = java_caster(
                self._change_set.addEdge, *[(src_vertex, self.id_type), (dst_vertex, self.id_type)]
            )

        return EdgeBuilder(self.session, eb, self.id_type)

    def add_vertex(self, vertex_id: Union[str, int]) -> VertexBuilder:
        """Add the vertex with the given id to the graph builder.

        :param vertex_id: The vertex id of the vertex to add.
        :returns: A 'VertexBuilder' instance containing the added vertex.
        """
        vb = java_caster(self._change_set.addVertex, (vertex_id, self.id_type))
        return VertexBuilder(self.session, vb, self.id_type)

    def build_new_snapshot(self) -> "PgxGraph":
        """Build a new snapshot of the graph out of this GraphChangeSet.

        The resulting PgxGraph is a new snapshot of the PgxGraph object this was created from.

        :returns: A new object of type 'PgxGraph'
        """
        pgx_graph_java = java_handler(self._change_set.buildNewSnapshot, [])
        return PgxGraph(self.session, pgx_graph_java)

    def remove_edge(self, edge_id: int) -> "GraphChangeSet":
        """Remove an edge from the graph.

        :param edge_id: The edge id of the edge to remove.
        :returns: self
        """
        java_handler(self._change_set.removeEdge, [edge_id])
        return self

    def remove_vertex(self, vertex_id: Union[int, str]) -> "GraphChangeSet":
        """Remove a vertex from the graph.

        :param vertex_id: The vertex id of the vertex to remove.
        :returns: self
        """
        java_caster(self._change_set.removeVertex, (vertex_id, self.id_type))
        return self

    def reset_edge(self, edge_id: int) -> "GraphChangeSet":
        """Reset any change for the edge with the given ID.

        :param edge_id: The edge id of the edge to reset.
        :returns: self
        """
        java_handler(self._change_set.resetEdge, [edge_id])
        return self

    def reset_vertex(self, vertex: Union[int, str]) -> "GraphChangeSet":
        """Reset any change for the referenced vertex.

        :param vertex: Either an instance of 'VertexBuilder' or a vertex id.
        :returns: self
        """
        if isinstance(vertex, VertexBuilder):
            vertex = vertex._builder.getId()
        java_caster(self._change_set.resetVertex, (vertex, self.id_type))
        return self

    def set_retain_edge_ids(self, retain_edge_ids: bool) -> "GraphChangeSet":
        """Control whether the edge ids provided in this graph builder are to be retained in the
        final graph.

        :param retain_edge_ids: A boolean value.
        :returns: self
        """
        java_handler(self._change_set.setRetainEdgeIds, [retain_edge_ids])
        return self

    def set_retain_ids(self, retain_ids: bool) -> "GraphChangeSet":
        """Control for both vertex and edge ids whether to retain them in the final graph.

        :param retain_ids: A boolean value.
        :returns: self
        """
        java_handler(self._change_set.setRetainIds, [retain_ids])
        return self

    def set_retain_vertex_ids(self, retain_vertex_ids: bool) -> "GraphChangeSet":
        """Control whether to retain the vertex ids provided in this graph builder are to be
        retained in the final graph.

        :param retain_vertex_ids: A boolean value.
        :returns: self
        """
        java_handler(self._change_set.setRetainVertexIds, [retain_vertex_ids])
        return self

    def update_edge(self, edge_id: int) -> "EdgeModifier":
        """Return an 'EdgeModifier' with which you can update edge properties and the edge label.

        :param edge_id: The edge id of the edge to be updated
        :returns: An 'EdgeModifier'
        """
        java_edge_modifier = java_handler(self._change_set.updateEdge, [edge_id])
        return EdgeModifier(self.session, java_edge_modifier, self.id_type)

    def update_vertex(self, vertex_id: Union[int, str]) -> "VertexModifier":
        """Return a 'VertexModifier' with which you can update vertex properties.

        :param vertex_id: The vertex id of the vertex to be updated
        :returns: A 'VertexModifier'
        """
        java_vertex_modifier = java_caster(self._change_set.updateVertex, (vertex_id, self.id_type))
        return VertexModifier(self.session, java_vertex_modifier, self.id_type)

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class VertexModifier(GraphChangeSet, VertexBuilder):
    """A class to modify existing vertices of a graph."""

    _java_class = 'oracle.pgx.api.VertexModifier'

    def __init__(
        self, session: "PgxSession", java_vertex_modifier, id_type: str = 'integer'
    ) -> None:
        """Construct a new vertex modifier.

        :param session: The current 'PgxSession' object.
        :param java_vertex_modifier: An instance of the corresponding java
            'VertexModifier' interface.
        :param id_type: A string describing the type of the ids. Optional.
            Defaults to 'integer'.
        """
        GraphChangeSet.__init__(self, session, java_vertex_modifier, id_type)
        VertexBuilder.__init__(self, session, java_vertex_modifier, id_type)
        self.session = session
        self.id_type = id_type
        self._vertex_modifier = java_vertex_modifier

    def add_label(self, label: str) -> "VertexModifier":
        """Add the given label to this vertex.

        :param label: The label to add.
        :returns: self
        """
        java_handler(self._vertex_modifier.addLabel, [label])
        return self

    def get_id(self) -> int:
        """Get the id of the element (vertex or edge) this builder belongs to.

        :returns: The id of this builder.
        """
        v_id = java_handler(self._vertex_modifier.getId, [])
        return v_id

    def remove_label(self, label: str) -> "VertexModifier":
        """Remove the given label from the vertex.

        :param label: The label to remove.
        :returns: self
        """
        java_handler(self._vertex_modifier.removeLabel, [label])
        return self

    def set_property(self, key: str, value: Any) -> "VertexModifier":
        """Set the property value of this vertex with the given key to the given value.

        :param key: A string with the name of the property to set.
        :param value: The value to which this property shall be set.
        :returns: self
        """
        java_handler(self._vertex_modifier.setProperty, [key, value])
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class EdgeModifier(GraphChangeSet, EdgeBuilder):
    """A class to modify existing edges of a graph."""

    _java_class = 'oracle.pgx.api.EdgeModifier'

    def __init__(self, session: "PgxSession", java_edge_modifier, id_type: str = 'integer') -> None:
        """Construct a new edge modifier.

        :param session: The current 'PgxSession' object.
        :param java_vertex_modifier: An instance of the corresponding java
            'EdgeModifier' interface.
        :param id_type: A string describing the type of the ids. Optional.
            Defaults to 'integer'.
        """
        GraphChangeSet.__init__(self, session, java_edge_modifier, id_type)
        EdgeBuilder.__init__(self, session, java_edge_modifier, id_type)
        self.session = session
        self.id_type = id_type
        self._edge_modifier = java_edge_modifier

    def set_label(self, label: str) -> "EdgeModifier":
        """Set the new value of the label.

        :param label: The label to be set.
        :returns: self
        """
        java_handler(self._edge_modifier.setLabel, [label])
        return self

    def set_property(self, key: str, value: Any) -> "EdgeModifier":
        """Set the property value of this edge with the given key to the given value.

        :param key: A string with the name of the property to set.
        :param value: The value to which this property shall be set.
        :returns: self
        """
        java_handler(self._edge_modifier.setProperty, [key, value])
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
