#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from pypgx._utils.error_handling import java_handler
from pypgx.api._graph_config import GraphConfig
from pypgx._utils.pgx_types import id_types
from jnius import autoclass
from pypgx._utils.error_messages import UNHASHABLE_TYPE
from typing import Optional, NoReturn

_JavaGraphMetaData = autoclass('oracle.pgx.api.GraphMetaData')


class GraphMetaData:
    """Meta information about PgxGraph."""

    _java_class = 'oracle.pgx.api.GraphMetaData'

    def __init__(
        self,
        java_graph_meta_data=None,
        vertex_id_type: Optional[str] = None,
        edge_id_type: Optional[str] = None,
    ) -> None:
        """Initialize this GraphMetaData object.

        Although all parameters are optional, the function expects either
        'java_graph_meta_data' to be different from 'None' or 'vertex_id_type'
        and 'edge_id_type' to be different from 'None'.

        :param java_graph_meta_data: A java object of type 'GraphMetaData' or one of its subclasses
        :param vertex_id_type: A string describing the type of the vertex id
        :param edge_id_type: A string describing the type of the edge id

        """
        args = [java_graph_meta_data, vertex_id_type, edge_id_type]
        is_none = list(map(lambda arg: arg is None, args))

        # Check whether either 'java_graph_meta_data' is not 'None' or
        # 'vertex_id_type' and 'edge_id_type' is not 'None'
        if not (is_none == [True, False, False] or is_none == [False, True, True]):
            raise TypeError(
                "You must specify either a Java 'GraphMetaData' object or a "
                "'vertex_id_type' and an 'edge_id_type'"
            )

        if vertex_id_type is not None:

            java_graph_meta_data = java_handler(
                _JavaGraphMetaData, [id_types[vertex_id_type], id_types[edge_id_type]]
            )

        self._graph_meta_data = java_graph_meta_data

    def get_config(self) -> Optional[GraphConfig]:
        """Get the graph configuration object used to specify the data source of this graph.

        :returns: Returns the 'GraphConfig' object of this 'GraphMetaData' object.
        """
        java_graph_config = java_handler(self._graph_meta_data.getConfig, [])
        if java_graph_config is None:
            return None
        return GraphConfig(java_graph_config)

    def get_creation_request_timestamp(self) -> int:
        """Get the timestamp (milliseconds since Jan 1st 1970) when this graph was requested to
        be created.

        :returns: A long value containing the timestamp.
        """
        return java_handler(self._graph_meta_data.getCreationRequestTimestamp, [])

    def get_creation_timestamp(self) -> int:
        """Get the timestamp (milliseconds since Jan 1st 1970) when this graph finished creation.

        :returns: A long value containing the timestamp.
        """
        return java_handler(self._graph_meta_data.getCreationTimestamp, [])

    def get_data_source_version(self) -> str:
        """Get the format-specific version identifier provided by the data-source.

        :returns: A string containing the version.
        """
        return java_handler(self._graph_meta_data.getDataSourceVersion, [])

    def get_memory_mb(self) -> int:
        """Get the estimated number of memory this graph (including its properties) consumes in
        memory (in megabytes).

        :returns: A long value containing the estimated amount of memory.
        """
        return java_handler(self._graph_meta_data.getMemoryMb, [])

    def get_num_edges(self) -> int:
        """Get the number of edges.

        :returns: A long value containing the number of edges.
        """
        return java_handler(self._graph_meta_data.getNumEdges, [])

    def get_num_vertices(self) -> int:
        """Get the number of vertices.

        :returns: A long value containing the number of vertices.
        """
        return java_handler(self._graph_meta_data.getNumVertices, [])

    def hash_code(self) -> int:
        """Return the hash code of this object.

        :returns: An int value containing the hash code.
        """
        return java_handler(self._graph_meta_data.hashCode, [])

    def is_directed(self) -> bool:
        """Return if the graph is directed.

        :returns: 'True' if the graph is directed and 'False' otherwise.
        """
        return java_handler(self._graph_meta_data.isDirected, [])

    def is_partitioned(self) -> bool:
        """Return if the graph is partitioned or not.

        :returns: 'True' if the graph is partitioned and 'False' otherwise.
        """
        return java_handler(self._graph_meta_data.isPartitioned, [])

    def set_config(self, config: GraphConfig) -> None:
        """Set a new 'GraphConfig'.

        :param config: An object of type 'GraphConfig'.
        """
        java_handler(self._graph_meta_data.setConfig, [config._graph_config])

    def set_creation_request_timestamp(self, creation_request_timestamp: int) -> None:
        """Set a new creation-request timestamp.

        :param creation_request_timestamp: A long value containing the new
            creation-request timestamp.
        """
        java_handler(
            self._graph_meta_data.setCreationRequestTimestamp, [creation_request_timestamp]
        )

    def set_creation_timestamp(self, creation_timestamp: int) -> None:
        """Set a new creation timestamp.

        :param creation_timestamp: A long value containing the new creation timestamp.
        """
        java_handler(self._graph_meta_data.setCreationTimestamp, [creation_timestamp])

    def set_data_source_version(self, data_source_version: str) -> None:
        """Set a new data source version.

        :param data_source_version: A string containing the new version.
        """
        java_handler(self._graph_meta_data.setDataSourceVersion, [data_source_version])

    def set_directed(self, directed: bool) -> None:
        """Assign a new truth value to the 'directed' variable.

        :param directed: A boolean value
        """
        java_handler(self._graph_meta_data.setDirected, [directed])

    def set_memory_mb(self, memory_mb: int) -> None:
        """Set a new amount of memory usage.

        :param memory_mb: A long value containing the new amount of memory.
        """
        java_handler(self._graph_meta_data.setMemoryMb, [memory_mb])

    def set_num_edges(self, num_edges: int) -> None:
        """Set a new amount of edges.

        :param num_edges: A long value containing the new amount of edges.
        """
        java_handler(self._graph_meta_data.setNumEdges, [num_edges])

    def set_num_vertices(self, num_vertices: int) -> None:
        """Set a new amount of vertices.

        :param num_vertices: A long value containing the new amount of edges.
        """
        java_handler(self._graph_meta_data.setNumVertices, [num_vertices])

    def __repr__(self) -> str:
        return java_handler(self._graph_meta_data.toString, [])

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
