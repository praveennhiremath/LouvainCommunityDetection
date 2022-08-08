#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import json
from typing import Dict, List, Optional

from pypgx._utils.error_handling import java_handler


class GraphConfig:
    """A class for representing graph configurations.

    :ivar bool is_file_format: whether the format is a file-based format
    :ivar bool has_vertices_and_edges_separated_file_format: whether given format has vertices
        and edges separated in different files
    :ivar bool is_single_file_format: whether given format has vertices and edges combined
        in same file
    :ivar bool is_multiple_file_format: whether given format has vertices and edges separated
        in different files
    :ivar bool supports_edge_label: whether given format supports edge label
    :ivar bool supports_vertex_labels: whether given format supports vertex labels
    :ivar bool supports_vector_properties: whether given format supports vector properties
    :ivar bool supports_property_column: whether given format supports property columns
    :ivar str name: the graph name of this graph configuration. Note: for file-based graph
        configurations, this is the file name of the URI this configuration points to.
    :ivar int num_vertex_properties: the number of vertex properties in this graph configuration
    :ivar int num_edge_properties: the number of edge properties in this graph configuration
    :ivar str format: Graph data format. The possible formats are in the table below.

    ==============   ======================================================
    Format string    Description
    ==============   ======================================================
    PGB              PGX Binary File Format (formerly EBIN)
    EDGE_LIST        Edge List file format
    TWO_TABLES       Two-Tables format (vertices, edges)
    ADJ_LIST         Adjacency List File Format
    FLAT_FILE        Flat File Format
    GRAPHML          GraphML File Format
    PG               Property Graph (PG) Database Format
    RDF              Resource Description Framework (RDF) Database Format
    CSV              Comma-Separated Values (CSV) Format
    ==============   ======================================================
    """

    _java_class = 'oracle.pgx.config.GraphConfig'

    def __init__(self, java_graph_config) -> None:
        self._graph_config = java_graph_config
        self._config_dict = json.loads(java_graph_config.toString())
        format_java = java_graph_config.getFormat()

        # The following attributes do not exist for partitioned graphs. Therefore only access them
        # if 'java_graph_config' does not belong to a partitioned graph.
        if format_java is not None:
            self.format = format_java.toString()
            self.is_file_format = java_graph_config.isFileFormat()
            self.has_vertices_and_edges_separated_file_format = (
                java_graph_config.hasVerticesAndEdgesSeparatedFileFormat()
            )
            self.is_single_file_format = java_graph_config.isSingleFileFormat()
            self.is_multiple_file_format = java_graph_config.isMultipleFileFormat()
            self.supports_edge_label = java_graph_config.supportsEdgeLabel()
            self.supports_vertex_labels = java_graph_config.supportsVertexLabels()
            self.supports_vector_properties = java_graph_config.supportsVectorProperties()
            self.supports_property_column = java_graph_config.supportsPropertyColumn()
        else:
            self.format = None
            self.is_file_format = None
            self.has_vertices_and_edges_separated_file_format = None
            self.is_single_file_format = None
            self.is_multiple_file_format = None
            self.supports_edge_label = None
            self.supports_vertex_labels = None
            self.supports_vector_properties = None
            self.supports_property_column = None

        self.name = java_graph_config.getName()
        self.num_vertex_properties = java_graph_config.numNodeProperties()
        self.num_edge_properties = java_graph_config.numEdgeProperties()

    @property
    def vertex_props(self) -> List[str]:
        """Get the vertex property names as a list.

        :rtype: list of str
        """
        props = []
        prop_list = java_handler(self._graph_config.getVertexProps, [])
        for prop in prop_list:
            props.append(prop.toString())
        return props

    @property
    def edge_props(self) -> List[str]:
        """Get the edge property names as a list.

        :rtype: list of str
        """
        props = []
        prop_list = java_handler(self._graph_config.getEdgeProps, [])
        for prop in prop_list:
            props.append(prop.toString())
        return props

    @property
    def vertex_property_types(self) -> Dict[str, str]:
        """Get the vertex property types as a dictionary.

        :return: dict mapping property names to their types
        :rtype: dict of str: str
        """
        prop_types = {}
        prop_map = java_handler(self._graph_config.getVertexPropertyTypes, [])
        for prop_name in prop_map.keySet():
            prop_types[prop_name] = prop_map.get(prop_name).toString()
        return prop_types

    @property
    def edge_property_types(self) -> Dict[str, str]:
        """Get the edge property types as a dictionary.

        :return: dict mapping property names to their types
        :rtype: dict of str: str
        """
        prop_types = {}
        prop_map = java_handler(self._graph_config.getEdgePropertyTypes, [])
        for prop_name in prop_map.keySet():
            prop_types[prop_name] = prop_map.get(prop_name).toString()
        return prop_types

    @property
    def vertex_id_type(self) -> Optional[str]:
        """Get the type of the vertex ID.

        :return: a str indicating the type of the vertex ID,
            one of "INTEGER", "LONG" or "STRING", or None
        :rtype: str or None
        """
        v_type = self._graph_config.getVertexIdType()
        if v_type is not None:
            return v_type.toString()
        else:
            return None

    @property
    def edge_id_type(self) -> Optional[str]:
        """Get the type of the edge ID.

        :return: a str indicating the type of the vertex ID,
            one of "INTEGER", "LONG" or "STRING", or None
        :rtype: str or None
        """
        e_type = self._graph_config.getEdgeIdType()
        if e_type is not None:
            return e_type.toString()
        else:
            return None

    def __repr__(self) -> str:
        return str(self._graph_config.toString())

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._graph_config.equals(other._graph_config)
