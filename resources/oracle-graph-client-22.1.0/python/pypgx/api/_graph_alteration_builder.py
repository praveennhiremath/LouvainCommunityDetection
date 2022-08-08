#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from typing import Optional, TYPE_CHECKING

from pypgx._utils.error_handling import java_handler

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class GraphAlterationBuilder:
    """Builder to describe the alterations (graph schema modification) to perform to a graph.

    It is for example possible to add or remove vertex and edge providers.
    """

    _java_class = "oracle.pgx.api.graphalteration.GraphAlterationBuilder"

    def __init__(self, java_graph_alteration_builder) -> None:
        self._graph_alteration_builder = java_graph_alteration_builder

    def set_data_source_version(self, data_source_version: str) -> None:
        """Set the version information for the built graph or snapshot.

        :param data_source_version: the version information.
        """
        java_handler(self._graph_alteration_builder.setDataSourceVersion, [data_source_version])

    def cascade_edge_provider_removals(
        self, cascade_edge_provider_removals: bool
    ) -> "GraphAlterationBuilder":
        """Specify if the edge providers associated to a vertex provider
        (the vertex provider is either the source or destination provider for that edge provider)
        being removed should be automatically removed too or not.
        By default, edge providers are not automatically removed whenever an associated vertex
        is removed. In that setting, if the associated edge providers are not specifically removed,
        an exception will be thrown to indicate that issue.

        :param cascade_edge_provider_removals: whether or not to automatically
                remove associated edge providers of removed vertex providers.
        :returns: a GraphAlterationBuilder instance with new changes.
        """
        graph_builder = java_handler(
            self._graph_alteration_builder.cascadeEdgeProviderRemovals,
            [cascade_edge_provider_removals],
        )
        self._graph_alteration_builder = graph_builder
        return self

    def add_vertex_provider(self, path_to_vertex_provider_config: str) -> "GraphAlterationBuilder":
        """Add a vertex provider for which the configuration is in a file at the specified path.

        :param path_to_vertex_provider_config: the path to the JSON configuration
                of the vertex provider.
        :returns: a GraphAlterationBuilder instance with the added vertex provider.
        """
        graph_builder = java_handler(
            self._graph_alteration_builder.addVertexProvider, [path_to_vertex_provider_config]
        )
        self._graph_alteration_builder = graph_builder
        return self

    def remove_vertex_provider(self, vertex_provider_name: str) -> "GraphAlterationBuilder":
        """Remove the vertex provider that has the given name.
        Also removes the associated edge providers if True was specified when calling
        `cascade_edge_provider_removals(boolean)`.

        :param vertex_provider_name: the name of the provider to remove.
        :returns: a GraphAlterationBuilder instance with the vertex_provider removed.
        """
        graph_builder = java_handler(
            self._graph_alteration_builder.removeVertexProvider, [vertex_provider_name]
        )
        self._graph_alteration_builder = graph_builder
        return self

    def add_edge_provider(self, path_to_edge_provider_config: str) -> "GraphAlterationBuilder":
        """Add an edge provider for which the configuration is in a file at the specified path.

        :param path_to_edge_provider_config: the path to the JSON configuration of the edge provider
        :returns: a GraphAlterationBuilder instance containing the added edge provider.
        """
        graph_builder = java_handler(
            self._graph_alteration_builder.addEdgeProvider, [path_to_edge_provider_config]
        )
        self._graph_alteration_builder = graph_builder
        return self

    def remove_edge_provider(self, edge_provider_name: str) -> "GraphAlterationBuilder":
        """Remove the edge provider that has the given name.

        :param edge_provider_name: the name of the provider to remove.
        :returns: a GraphAlterationBuilder instance with the edge_provider removed.
        """
        graph_builder = java_handler(
            self._graph_alteration_builder.removeEdgeProvider, [edge_provider_name]
        )
        self._graph_alteration_builder = graph_builder
        return self

    def build(self, new_graph_name: Optional[str] = None) -> "PgxGraph":
        """Create a new graph that is the result of the alteration of the current graph.

        :param new_graph_name: name of the new graph to create.
        :returns: a PgxGraph instance of the current alteration builder.
        """
        from pypgx.api._pgx_graph import PgxGraph  # need to import here to avoid import loop

        pgx_graph = java_handler(self._graph_alteration_builder.build, [new_graph_name])
        return PgxGraph(self, pgx_graph)

    def build_new_snapshot(self) -> "PgxGraph":
        """Create a new snapshot for the current graph that is the result of
        the alteration of the current snapshot.

        :returns: a PgxGraph instance of the current alteration builder.
        """
        from pypgx.api._pgx_graph import PgxGraph  # need to import here to avoid import loop

        pgx_graph = java_handler(self._graph_alteration_builder.buildNewSnapshot)
        return PgxGraph(self, pgx_graph)
