#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass

from pypgx._utils.error_handling import java_handler, _cast_to_java
from pypgx.api._graph_config import GraphConfig


class GraphConfigFactory:
    """A factory class for creating graph configs."""

    _java_class = 'oracle.pgx.config.GraphConfigFactory'

    _graph_config_factory_class = autoclass('oracle.pgx.config.GraphConfigFactory')
    _pgx_config_java_class = autoclass('oracle.pgx.config.PgxConfig$Field')
    _graph_config_java = autoclass('oracle.pgx.config.GraphConfig')

    strict_mode = java_handler(_pgx_config_java_class.STRICT_MODE.getDefaultVal, []) == 1

    def __init__(self, java_graph_config_factory) -> None:
        """Initialize this factory object.

        :param java_graph_config_factory: A java object of type 'GraphConfigFactory' or one of
        its subclasses
        """
        self._graph_config_factory = java_graph_config_factory

    @staticmethod
    def init(want_strict_mode: bool = True) -> None:
        """Setter function for the 'strictMode' class variable.

        :param want_strict_mode: A boolean value which will be assigned to 'strictMode'
            (Default value = True)
        """
        GraphConfigFactory.strict_mode = want_strict_mode
        java_handler(GraphConfigFactory._graph_config_factory_class.init, [want_strict_mode])

    @staticmethod
    def for_partitioned() -> "GraphConfigFactory":
        """Return a new graph config factory to parse partitioned graph config."""
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forPartitioned, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_any_format() -> "GraphConfigFactory":
        """Return a new factory to parse any graph config from various input sources."""
        java_object = java_handler(GraphConfigFactory._graph_config_factory_class.forAnyFormat, [])
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_file_formats() -> "GraphConfigFactory":
        """Return a new graph config factory to parse file-based graph configs from various input
        sources."""
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forFileFormats, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_two_tables_rdbms() -> "GraphConfigFactory":
        """Return a new graph config factory to create graph configs targeting the Oracle RDBMS
        database in the two-tables format."""
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forTwoTablesRdbms, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_two_tables_text() -> "GraphConfigFactory":
        """Return a new graph config factory to create graph configs targeting files in the
        two-tables format."""
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forTwoTablesText, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_property_graph_rdbms() -> "GraphConfigFactory":
        """Return a new graph config factory to create graph configs targeting the Oracle
        RDBMS database in the property graph format.
        """
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forPropertyGraphRdbms, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_property_graph_nosql() -> "GraphConfigFactory":
        """Return a new graph config factory to create graph configs targeting the Oracle NoSQL
        database in the property graph format.
        """
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forPropertyGraphNosql, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_property_graph_hbase() -> "GraphConfigFactory":
        """Return a new graph config factory to create graph configs targeting the Apache HBase
        database in the property graph format.
        """
        java_object = java_handler(
            GraphConfigFactory._graph_config_factory_class.forPropertyGraphHbase, []
        )
        return GraphConfigFactory(java_object)

    @staticmethod
    def for_rdf() -> "GraphConfigFactory":
        """Return a new RDF graph config factory."""
        java_object = java_handler(GraphConfigFactory._graph_config_factory_class.forRdf, [])
        return GraphConfigFactory(java_object)

    def from_file_path(self, path: str) -> GraphConfig:
        """Parse a configuration object given as path to a JSON file.

        Relative paths found in JSON are resolved relative to given file.

        :param path: The path to the JSON file
        """
        java_config = java_handler(self._graph_config_factory.fromFilePath, [path])
        java_config_casted = _cast_to_java(java_config, GraphConfigFactory._graph_config_java)
        return GraphConfig(java_config_casted)

    def from_json(self, json: str) -> GraphConfig:
        """Parse a configuration object given a JSON string.

        :param json: The input JSON string
        """
        java_config = java_handler(self._graph_config_factory.fromJson, [json])
        java_config_casted = _cast_to_java(java_config, GraphConfigFactory._graph_config_java)
        return GraphConfig(java_config_casted)

    def from_input_stream(self, stream) -> GraphConfig:
        """Parse a configuration object given an input stream.

        :param stream: A JAVA 'InputStream' objeft from where to read the configuration
        """
        java_config = java_handler(self._graph_config_factory.fromInputStream, [stream])
        java_config_casted = _cast_to_java(java_config, GraphConfigFactory._graph_config_java)
        return GraphConfig(java_config_casted)

    def from_properties(self, properties) -> GraphConfig:
        """Parse a configuration object from a properties object.

        :param properties: A JAVA 'Properties' object
        """
        java_config = java_handler(self._graph_config_factory.fromProperties, [properties])
        java_config_casted = _cast_to_java(java_config, GraphConfigFactory._graph_config_java)
        return GraphConfig(java_config_casted)

    def from_path(self, path: str) -> GraphConfig:
        """Parse a configuration object given a path.

        :param path: The path from where to parse the configuration.
        """
        java_config = java_handler(self._graph_config_factory.fromPath, [path])
        java_config_casted = _cast_to_java(java_config, GraphConfigFactory._graph_config_java)
        return GraphConfig(java_config_casted)

    def __repr__(self) -> str:
        return self._graph_config_factory.toString()

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._graph_config_factory.equals(other._graph_config_factory)
