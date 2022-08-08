#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from collections.abc import Iterable
from jnius import autoclass

from pypgx.api._all_paths import AllPaths
from pypgx.api._graph_alteration_builder import GraphAlterationBuilder
from pypgx.api._mutation_strategy import MutationStrategy
from pypgx.api._mutation_strategy_builder import MergingStrategyBuilder, PickingStrategyBuilder
from pypgx.api._pgql_result_set import PgqlResultSet
from pypgx.api._pgx_collection import (
    EdgeSequence,
    EdgeSet,
    VertexSequence,
    VertexSet,
    PgxCollection,
)
from pypgx.api._pgx_entity import PgxEdge, PgxVertex
from pypgx.api._pgx_map import PgxMap
from pypgx.api._property import EdgeProperty, VertexProperty, EdgeLabel, VertexLabels
from pypgx.api._scalar import Scalar
from pypgx.api._synchronizer import Synchronizer
from pypgx.api._operation import Operation
from pypgx.api._partition import PgxPartition
from pypgx.api._pgx_path import PgxPath
from pypgx.api._server_instance import ServerInstance
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx.api._pgx_id import PgxId
from pypgx.api._graph_config import GraphConfig
from pypgx.api._graph_meta_data import GraphMetaData
from pypgx.api.filters import GraphFilter, EdgeFilter, VertexFilter
from pypgx.api.redaction._redaction_rule_config import PgxRedactionRuleConfig
from pypgx._utils.error_handling import java_caster, java_handler
from pypgx._utils.error_messages import ARG_MUST_BE, INVALID_OPTION
from pypgx._utils.pgx_types import (
    authorization_types,
    collection_types,
    edge_props,
    format_types,
    on_invalid_change_types,
    property_types,
    pgx_resource_permissions,
    vector_types,
    vertex_props,
)
from pypgx._utils.pgx_types import (
    degree_type,
    mode,
    multi_edges,
    self_edges,
    sort_order,
    trivial_vertices,
    id_generation_strategies,
)
from pypgx.api.auth import PermissionEntity
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession
    from pypgx.api._prepared_statement import PreparedStatement
    from pypgx.api._graph_change_set import GraphChangeSet


class PgxGraph(PgxContextManager):
    """A reference to a graph on the server side.

    Operations on instances of this class are executed on the server side onto the referenced
    graph. Note that a session can have multiple objects referencing the same graph: the result
    of any operation mutating the graph on any of those references will be visible on all of them.
    """

    _java_class = 'oracle.pgx.api.PgxGraph'

    def __init__(self, session: "PgxSession", java_graph) -> None:
        self._update_variables(session, java_graph)

    def _update_variables(self, session, java_graph):
        self._graph = java_graph
        self.session = session
        self.name = java_graph.getName()
        self.is_transient = java_graph.isTransient()
        self.num_vertices = java_graph.getNumVertices()
        self.num_edges = java_graph.getNumEdges()
        self.memory_mb = java_graph.getMemoryMb()
        self.data_source_version = java_graph.getDataSourceVersion()
        self.is_directed = java_graph.isDirected()
        self.creation_request_timestamp = java_graph.getCreationRequestTimestamp()
        self.creation_timestamp = java_graph.getCreationTimestamp()
        # vertex id type can be null in case of graph with no Ids strategy
        if java_graph.getVertexIdType() is not None:
            self.vertex_id_type = java_graph.getVertexIdType().toString()
        self.vertex_id_strategy = java_graph.getVertexIdStrategy().toString()
        self.edge_id_strategy = java_graph.getEdgeIdStrategy().toString()

    @property
    def pgx_instance(self) -> ServerInstance:
        """Get the server instance."""
        return ServerInstance(self._graph.getPgxInstance())

    @property
    def config(self) -> GraphConfig:
        """Get the GraphConfig object."""
        return GraphConfig(self._graph.getConfig())

    def get_meta_data(self) -> GraphMetaData:
        """Get the GraphMetaData object.

        :returns: A 'GraphMetaData' object of this graph.
        """
        return GraphMetaData(self._graph.getMetaData())

    def get_id(self) -> str:
        """Get the Graph id.

        :returns: A string representation of the id of this graph.
        """
        pgx_id = java_handler(self._graph.getId, [])
        return java_handler(pgx_id.toString, [])

    def get_pgx_id(self) -> PgxId:
        """Get the Graph id.

        :returns: The id of this graph.
        """
        pgx_id = java_handler(self._graph.getId, [])
        return PgxId(pgx_id)

    def get_vertex(self, vid: Union[str, int]) -> PgxVertex:
        """Get a vertex with a specified id.

        :param vid: Vertex id
        :returns: pgxVertex object
        """
        java_vertex = java_caster(self._graph.getVertex, (vid, self.vertex_id_type))
        return PgxVertex(self, java_vertex)

    def has_vertex(self, vid: int) -> bool:
        """Check if the vertex with id vid is in the graph.

        :param vid: vertex id
        """
        return java_handler(self._graph.hasVertex, [vid])

    def get_edge(self, eid: int) -> PgxEdge:
        """Get an edge with a specified id.

        :param eid: edge id
        """
        return PgxEdge(self, java_handler(self._graph.getEdge, [eid]))

    def has_edge(self, eid: int) -> bool:
        """Check if the edge with id vid is in the graph.

        :param eid: Edge id
        """
        return java_handler(self._graph.hasEdge, [eid])

    def get_random_vertex(self) -> PgxVertex:
        """Get a random vertex from the graph."""
        return PgxVertex(self, self._graph.getRandomVertex())

    def get_random_edge(self) -> PgxEdge:
        """Get a edge vertex from the graph."""
        return PgxEdge(self, self._graph.getRandomEdge())

    def has_vertex_labels(self) -> bool:
        """Return True if the graph has vertex labels, False if not."""
        return self._graph.hasVertexLabels()

    def get_vertex_labels(self) -> VertexLabels:
        """Get the vertex labels belonging to this graph."""
        vl = java_handler(self._graph.getVertexLabels, [])
        return VertexLabels(self, vl)

    def has_edge_label(self) -> bool:
        """Return True if the graph has edge labels, False if not."""
        return self._graph.hasEdgeLabel()

    def get_edge_label(self) -> EdgeLabel:
        """Get the edge labels belonging to this graph."""
        el = java_handler(self._graph.getEdgeLabel, [])
        return EdgeLabel(self, el)

    def get_vertex_properties(self) -> List[VertexProperty]:
        """Get the set of vertex properties belonging to this graph.

        This list might contain transient, private and published properties.
        """
        java_props = self._graph.getVertexProperties()
        props = []
        for prop in java_props:
            props.append(VertexProperty(self, prop))
        props.sort(key=lambda prop: prop.name)
        return props

    def get_edge_properties(self) -> List[EdgeProperty]:
        """Get the set of edge properties belonging to this graph.

        This list might contain transient, private and published properties.
        """
        java_props = self._graph.getEdgeProperties()
        props = []
        for prop in java_props:
            props.append(EdgeProperty(self, prop))
        props.sort(key=lambda prop: prop.name)
        return props

    def get_vertex_property(self, name: str) -> Optional[VertexProperty]:
        """Get a vertex property by name.

        :param name: Property name
        """
        prop = java_handler(self._graph.getVertexProperty, [name])
        if prop:
            return VertexProperty(self, prop)
        return None

    def get_edge_property(self, name: str) -> Optional[EdgeProperty]:
        """Get an edge property by name.

        :param name: Property name
        """
        prop = java_handler(self._graph.getEdgeProperty, [name])
        if prop:
            return EdgeProperty(self, prop)
        return None

    def create_scalar(self, data_type: str, name: Optional[str] = None) -> Scalar:
        """Create a new Scalar.

        :param data_type: Scalar type
        :param name:  Name of the scalar to be created
        """
        if data_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='data_type', opts=list(property_types.keys()))
            )
        scalar = java_handler(self._graph.createScalar, [property_types[data_type], name])
        return Scalar(self, scalar)

    def create_vector_scalar(self, data_type: str, name: Optional[str] = None) -> Scalar:
        """Create a new vertex property.

        :param data_type: Property type
        :param name:  Name of the scalar to be created
        """
        if data_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='data_type', opts=list(property_types.keys()))
            )
        scalar = java_handler(self._graph.createVectorScalar, [property_types[data_type], name])
        return Scalar(self, scalar)

    def create_vertex_property(self, data_type: str, name: Optional[str] = None) -> VertexProperty:
        """Create a new vertex property.

        :param data_type: Property type
        :param name:  Name of the property to be created
        """
        if data_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='data_type', opts=list(property_types.keys()))
            )
        prop = java_handler(self._graph.createVertexProperty, [property_types[data_type], name])
        return VertexProperty(self, prop)

    def create_vertex_vector_property(
        self, data_type: str, dim: int, name: Optional[str] = None
    ) -> VertexProperty:
        """Create a session-bound vertex vector property.

        :param data_type: Type of the vector property to create
        :param dim: Dimension of vector property to be created
        :param name: Name of vector property to be created
        """
        if data_type not in vector_types:
            raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(vector_types)))
        prop = java_handler(
            self._graph.createVertexVectorProperty, [property_types[data_type], dim, name]
        )
        return VertexProperty(self, prop)

    def get_or_create_vertex_vector_property(
        self, data_type: str, dim: int, name: Optional[str] = None
    ) -> VertexProperty:
        """Get or create a session-bound vertex vector property.

        :param data_type: Type of the vector property to create
        :param dim: Dimension of vector property to be created
        :param name: Name of vector property to be created
        """
        if data_type not in vector_types:
            raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(vector_types)))
        prop = java_handler(
            self._graph.getOrCreateVertexVectorProperty, [property_types[data_type], dim, name]
        )
        return VertexProperty(self, prop)

    def create_edge_property(self, data_type: str, name: Optional[str] = None) -> EdgeProperty:
        """Create a session-bound edge property.

        :param data_type: Type of the vector property to create
        :param name: Name of vector property to be created
        """
        if data_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='data_type', opts=list(property_types.keys()))
            )
        prop = java_handler(self._graph.createEdgeProperty, [property_types[data_type], name])
        return EdgeProperty(self, prop)

    def get_or_create_edge_vector_property(
        self, data_type: str, dim: int, name: Optional[str] = None
    ) -> EdgeProperty:
        """Get or create a session-bound edge property.

        :param data_type: Type of the vector property to create
        :param dim: Dimension of vector property to be created
        :param name: Name of vector property to be created
        """
        if data_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='data_type', opts=list(property_types.keys()))
            )
        prop = java_handler(
            self._graph.getOrCreateEdgeVectorProperty, [property_types[data_type], dim, name]
        )
        return EdgeProperty(self, prop)

    def create_edge_vector_property(
        self, data_type: str, dim: int, name: Optional[str] = None
    ) -> EdgeProperty:
        """Create a session-bound vector property.

        :param data_type: Type of the vector property to create
        :param dim: Dimension of vector property to be created
        :param name: Name of vector property to be created
        """
        if data_type not in vector_types:
            raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(vector_types)))
        prop = java_handler(
            self._graph.createEdgeVectorProperty, [property_types[data_type], dim, name]
        )
        return EdgeProperty(self, prop)

    def get_or_create_vertex_property(
        self, name: str, data_type: Optional[str] = None, dim: int = 0
    ) -> VertexProperty:
        """Get or create a vertex property.

        :param name: Property name
        :param data_type:  Property type
        :param dim: Dimension of vector property to be created
        """
        prop = self.get_vertex_property(name)
        if prop or data_type is None:
            return prop
        elif data_type in vector_types and dim > 0:
            return self.create_vertex_vector_property(data_type, dim, name)
        elif data_type in property_types:
            return self.create_vertex_property(data_type, name)
        raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(property_types.keys())))

    def get_or_create_edge_property(
        self, name: str, data_type: Optional[str] = None, dim: int = 0
    ) -> EdgeProperty:
        """Get or create an edge property.

        :param name: Property name
        :param data_type:  Property type
        :param dim: Dimension of vector property to be created
        """
        prop = self.get_edge_property(name)
        if prop or data_type is None:
            return prop
        elif data_type in vector_types and dim > 0:
            return self.create_edge_vector_property(data_type, dim, name)
        elif data_type and data_type in property_types:
            return self.create_edge_property(data_type, name)
        raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(property_types.keys())))

    def pick_random_vertex(self) -> PgxVertex:
        """Select a random vertex from the graph.

        :return: The PgxVertex object
        """
        return self.get_random_vertex()

    def create_components(
        self, components: Union[VertexProperty, str], num_components: int
    ) -> PgxPartition:
        """Create a Partition object holding a collection of vertex sets, one for each component.

        :param components: Vertex property mapping each vertex to its component
            ID. Note that only component IDs in the range of
            [0..numComponents-1] are allowed. The returned future will complete
            exceptionally with an IllegalArgumentException if an invalid
            component ID is encountered. Gaps are supported: certain IDs not
            being associated with any vertices will yield to empty components.
        :param num_components: How many different components the components
            property contains
        :return: The Partition object
        """
        if not isinstance(components, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='components', type=VertexProperty.__name__))
        java_partition = java_handler(
            self._graph.createComponents, [components._prop, num_components]
        )
        java_vertex_property = java_handler(java_partition.getComponentsProperty, [])
        property = VertexProperty(self, java_vertex_property)

        return PgxPartition(self, java_partition, property)

    def store(
        self,
        format: str,
        path: str,
        num_partitions: Optional[int] = None,
        vertex_properties: bool = True,
        edge_properties: bool = True,
        overwrite: bool = False,
    ) -> GraphConfig:
        """Store graph in a file.

        :param format: One of ['pgb', 'edge_list', 'two_tables', 'adj_list',
            'flat_file', 'graphml', 'pg', 'rdf', 'csv']
        :param path: Path to which graph will be stored
        :param num_partitions: The number of partitions that should be
            created, when exporting to multiple files
        :param vertex_properties: The collection of vertex properties to store
            together with the graph data. If not specified all the vertex
            properties are stored
        :param edge_properties: The collection of edge properties to store
            together with the graph data. If not specified all the vertex
            properties are stored
        :param overwrite: Overwrite if existing
        """
        if format not in format_types:
            raise ValueError(INVALID_OPTION.format(var='format', opts=list(format_types.keys())))
        else:
            format = format_types[format]
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        config = None
        if num_partitions or isinstance(num_partitions, int):
            config = java_handler(
                self._graph.store, [format, path, num_partitions, vp, ep, overwrite]
            )
        else:
            config = java_handler(self._graph.store, [format, path, vp, ep, overwrite])
        return GraphConfig(config)

    def close(self) -> None:
        """Destroy without waiting for completion."""
        return java_handler(self._graph.close, [])

    def destroy_vertex_property_if_exists(self, name: str) -> None:
        """Destroy a specific vertex property if it exists.

        :param name: Property name
        """
        return java_handler(self._graph.destroyVertexPropertyIfExists, [name])

    def destroy_edge_property_if_exists(self, name: str) -> None:
        """Destroy a specific edge property if it exists.

        :param name: Property name
        """
        return java_handler(self._graph.destroyEdgePropertyIfExists, [name])

    @property
    def is_fresh(self) -> bool:
        """Check whether an in-memory representation of a graph is fresh."""
        return self._graph.isFresh()

    def get_vertices(
        self, filter_expr: Optional[Union[str, VertexFilter]] = None, name: Optional[str] = None
    ) -> VertexSet:
        """Create a new vertex set containing vertices according to the given filter expression.

        :param filter_expr:  VertexFilter object with the filter expression
             if None all the vertices are returned
        :param name:  The name of the collection to be created.
             If None, a name will be generated.
        """
        if filter_expr is None:
            filter_expr = VertexFilter("true")
        elif not isinstance(filter_expr, VertexFilter):
            raise TypeError(ARG_MUST_BE.format(arg='filter_expr', type=VertexFilter.__name__))
        return VertexSet(self, java_handler(self._graph.getVertices, [filter_expr._filter, name]))

    def get_edges(
        self, filter_expr: Optional[Union[str, EdgeFilter]] = None, name: Optional[str] = None
    ) -> EdgeSet:
        """Create a new edge set containing vertices according to the given filter expression.

        :param filter_expr:  EdgeFilter object with the filter expression.
             If None all the vertices are returned.
        :param name:  the name of the collection to be created.
             If None, a name will be generated.
        """
        if filter_expr is None:
            filter_expr = EdgeFilter("true")
        elif not isinstance(filter_expr, EdgeFilter):
            raise TypeError(ARG_MUST_BE.format(arg='filter_expr', type=EdgeFilter.__name__))
        return EdgeSet(self, java_handler(self._graph.getEdges, [filter_expr._filter, name]))

    def create_vertex_set(self, name: Optional[str] = None) -> VertexSet:
        """Create a new vertex set.

        :param name:  Set name
        """
        return VertexSet(self, java_handler(self._graph.createVertexSet, [name]))

    def create_vertex_sequence(self, name: Optional[str] = None) -> VertexSequence:
        """Create a new vertex sequence.

        :param name:  Sequence name
        """
        return VertexSequence(self, java_handler(self._graph.createVertexSequence, [name]))

    def create_edge_set(self, name: Optional[str] = None) -> EdgeSet:
        """Create a new edge set.

        :param name:  Edge set name
        """
        return EdgeSet(self, java_handler(self._graph.createEdgeSet, [name]))

    def create_edge_sequence(self, name: Optional[str] = None) -> EdgeSequence:
        """Create a new edge sequence.

        :param name:  Sequence name
        """
        return EdgeSequence(self, java_handler(self._graph.createEdgeSequence, [name]))

    def create_map(self, key_type: str, val_type: str, name: Optional[str] = None) -> PgxMap:
        """Create a session-bound map.

        Possible types are:
        ['integer','long','double','boolean','string','vertex','edge',
        'local_date','time','timestamp','time_with_timezone','timestamp_with_timezone']

        :param key_type:  Property type of the keys that are going to be stored inside the map
        :param val_type:  Property type of the values that are going to be stored inside the map
        :param name:  Map name
        """
        if key_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='key_type', opts=list(property_types.keys()))
            )
        elif val_type not in property_types:
            raise ValueError(
                INVALID_OPTION.format(var='val_type', opts=list(property_types.keys()))
            )
        k = property_types[key_type]
        v = property_types[val_type]
        return PgxMap(self, java_handler(self._graph.createMap, [k, v, name]))

    def sort_by_degree(
        self,
        vertex_properties: Union[List[VertexProperty], bool] = True,
        edge_properties: Union[List[EdgeProperty], bool] = True,
        ascending: bool = True,
        in_degree: bool = True,
        in_place: bool = False,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Create a sorted version of a graph and all its properties.

        The returned graph is sorted such that the node numbering is ordered by
        the degree of the nodes. Note that the returned graph and properties
        are transient.

        :param vertex_properties: List of vertex properties belonging to graph
            specified to be kept in the new graph
        :param edge_properties: List of edge properties belonging to graph
            specified to be kept in the new graph
        :param ascending:  Sorting order
        :param in_degree:  If in_degree should be used for sorting. Otherwise use out degree.
        :param in_place:  If the sorting should be done in place or a new graph should be created
        :param name:  New graph name
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        ascending = sort_order[ascending]
        in_degree = degree_type[in_degree]
        in_place = mode[in_place]
        new_graph = java_handler(
            self._graph.sortByDegree, [vp, ep, ascending, in_degree, in_place, name]
        )
        return PgxGraph(self.session, new_graph)

    def transpose(
        self,
        vertex_properties: bool = True,
        edge_properties: bool = True,
        edge_label_mapping: Optional[Mapping[str, str]] = None,
        in_place: bool = False,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Create a transpose of this graph.

        A transpose of a directed graph is another directed graph on the same
        set of vertices with all of the edges reversed. If this graph contains
        an edge (u,v) then the return graph will contain an edge (v,u) and vice
        versa. If this graph is undirected (isDirected() returns false), this
        operation has no effect and will either return a copy or act as
        identity function depending on the mode parameter.

        :param vertex_properties:  List of vertex properties belonging to graph
            specified to be kept in the new graph
        :param edge_properties:  List of edge properties belonging to graph
            specified to be kept in the new graph
        :param edge_label_mapping:  Can be used to rename edge labels.
            For example, an edge (John,Mary) labeled "fatherOf" can be transformed
            to be labeled "hasFather" on the transpose graph's edge (Mary,John)
            by passing in a dict like object {"fatherOf":"hasFather"}.
        :param in_place:  If the transpose should be done in place or a new
            graph should be created
        :param name:  New graph name
        """
        if edge_label_mapping is None:
            edge_label_mapping = {}
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        edge_labels = None
        if len(edge_label_mapping) > 0:
            edge_labels = autoclass('java.util.HashMap')()
            for key in edge_label_mapping.keys():
                edge_labels.put(key, edge_label_mapping[key])
        in_place = mode[in_place]
        new_graph = java_handler(self._graph.transpose, [vp, ep, edge_labels, in_place, name])
        return PgxGraph(self.session, new_graph)

    def undirect(
        self,
        vertex_properties: Union[bool, List[VertexProperty]] = True,
        edge_properties: Union[bool, List[EdgeProperty]] = True,
        keep_multi_edges: bool = True,
        keep_self_edges: bool = True,
        keep_trivial_vertices: bool = True,
        in_place: bool = False,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """
        Create an undirected version of the graph.

        An undirected graph has some restrictions. Some algorithms are only supported on directed
        graphs or are not yet supported for undirected graphs. Further, PGX does not support
        storing undirected graphs nor reading from undirected formats. Since the edges do not have a
        direction anymore, the behavior of `pgxEdge.source()` or `pgxEdge.destination()` can be
        ambiguous. In order to provide deterministic results, PGX will always return the vertex
        with the smaller internal id as source and the other as destination vertex.

        :param vertex_properties: List of vertex properties belonging to
            graph specified to be kept in the new graph
        :param edge_properties: List of edge properties belonging to graph
            specified to be kept in the new graph
        :param keep_multi_edges: Defines if multi-edges should be kept in the
            result
        :param keep_self_edges: Defines if self-edges should be kept in the
            result
        :param keep_trivial_vertices: Defines if isolated nodes should be kept
            in the result
        :param in_place: If the operation should be done in place of if a new
            graph has to be created
        :param name: New graph name
        """
        if self.is_directed:
            vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
            keep_multi_edges = multi_edges[keep_multi_edges]
            keep_self_edges = self_edges[keep_self_edges]
            keep_trivial_vertices = trivial_vertices[keep_trivial_vertices]
            in_place = mode[in_place]
            new_graph = java_handler(
                self._graph.undirect,
                [vp, ep, keep_multi_edges, keep_self_edges, keep_trivial_vertices, in_place,
                    name],
            )
            return PgxGraph(self.session, new_graph)
        else:
            return self

    def undirect_with_strategy(self, mutation_strategy: MutationStrategy) -> "PgxGraph":
        """
        Create an undirected version of the graph using a custom mutation strategy.

        An undirected graph has some restrictions. Some algorithms are only supported on directed
        graphs or are not yet supported for undirected graphs. Further, PGX does not support
        storing undirected graphs nor reading from undirected formats. Since the edges do not have a
        direction anymore, the behavior of `pgxEdge.source()` or `pgxEdge.destination()` can be
        ambiguous. In order to provide deterministic results, PGX will always return the vertex
        with the smaller internal id as source and the other as destination vertex.

        :param mutation_strategy: Defines a custom strategy for dealing with multi-edges.
        """
        if self.is_directed:
            new_graph = java_handler(
                self._graph.undirect,
                [mutation_strategy._mutation_strategy]
            )
            return PgxGraph(self.session, new_graph)
        else:
            return self

    def simplify(
        self,
        vertex_properties: Union[bool, List[VertexProperty]] = True,
        edge_properties: Union[bool, List[EdgeProperty]] = True,
        keep_multi_edges: bool = False,
        keep_self_edges: bool = False,
        keep_trivial_vertices: bool = False,
        in_place: bool = False,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Create a simplified version of a graph.

        Note that the returned graph and properties are transient and therefore
        session bound. They can be explicitly destroyed and get automatically
        freed once the session dies.

        :param vertex_properties: List of vertex properties belonging to graph
            specified to be kept in the new graph
        :param edge_properties: List of edge properties belonging to graph
            specified to be kept in the new graph
        :param keep_multi_edges: Defines if multi-edges should be kept in the
            result
        :param keep_self_edges: Defines if self-edges should be kept in the
            result
        :param keep_trivial_vertices: Defines if isolated nodes should be kept
            in the result
        :param in_place: If the operation should be done in place of if a new
            graph has to be created
        :param name: New graph name. If None, a name will be generated.
            Only relevant if a new graph is to be created.
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        keep_multi_edges = multi_edges[keep_multi_edges]
        keep_self_edges = self_edges[keep_self_edges]
        keep_trivial_vertices = trivial_vertices[keep_trivial_vertices]
        in_place = mode[in_place]
        new_graph = java_handler(
            self._graph.simplify,
            [vp, ep, keep_multi_edges, keep_self_edges, keep_trivial_vertices, in_place, name],
        )
        return PgxGraph(self.session, new_graph)

    def simplify_with_strategy(self, mutation_strategy: MutationStrategy) -> "PgxGraph":
        """Create a simplified version of a graph using a custom mutation strategy.

        Note that the returned graph and properties are transient and therefore
        session bound. They can be explicitly destroyed and get automatically
        freed once the session dies.

        :param mutation_strategy: Defines a custom strategy for dealing with multi-edges.
        """
        new_graph = java_handler(self._graph.simplify, [mutation_strategy._mutation_strategy])
        return PgxGraph(self.session, new_graph)

    def bipartite_sub_graph_from_left_set(
        self,
        vset: Union[str, VertexSet],
        vertex_properties: Union[List[VertexProperty], bool] = True,
        edge_properties: bool = True,
        name: Optional[str] = None,
        is_left_name: Optional[str] = None,
    ) -> "BipartiteGraph":
        """Create a bipartite version of this graph with the given vertex set being the left set.

        :param vset: Vertex set representing the left side
        :param vertex_properties:  List of vertex properties belonging to graph
            specified to be kept in the new graph
        :param edge_properties:  List of edge properties belonging to graph
            specified to be kept in the new graph
        :param name:  name of the new graph. If None, a name will be generated.
        :param is_left_name:   Name of the boolean isLeft vertex property of
            the new graph. If None, a name will be generated.
        """
        if not isinstance(vset, VertexSet):
            raise TypeError(ARG_MUST_BE.format(arg='vset', type=VertexSet.__name__))
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        b_graph = java_handler(
            self._graph.bipartiteSubGraphFromLeftSet, [vp, ep, vset._collection, name, is_left_name]
        )
        return BipartiteGraph(self.session, b_graph)

    def bipartite_sub_graph_from_in_degree(
        self,
        vertex_properties: Union[List[VertexProperty], bool] = True,
        edge_properties: bool = True,
        name: Optional[str] = None,
        is_left_name: Optional[str] = None,
        in_place: bool = False,
    ) -> "BipartiteGraph":
        """Create a bipartite version of this graph with all vertices of in-degree = 0 being the
        left set.

        :param vertex_properties:   List of vertex properties belonging to
            graph specified to be kept in the new graph
        :param edge_properties:  List of edge properties belonging to graph
            specified to be kept in the new graph
        :param name:  New graph name
        :param is_left_name:  Name of the boolean isLeft vertex property of
            the new graph. If None, a name will be generated.
        :param in_place: Whether to create a new copy (False) or overwrite this
            graph (True)
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        b_graph = java_handler(
            self._graph.bipartiteSubGraphFromInDegree, [vp, ep, name, is_left_name, in_place]
        )
        return BipartiteGraph(self.session, b_graph)

    def is_bipartite(self, is_left: Union[VertexProperty, str]) -> int:
        """Check whether a given graph is a bipartite graph.

        A graph is considered a bipartite graph if all nodes can be divided in a 'left' and a
        'right' side where edges only go from nodes on the 'left' side to nodes on the 'right'
        side.

        :param is_left: Boolean vertex property that - if the method returns true -
            will contain for each node whether it is on the 'left' side of the
            bipartite graph. If the method returns False, the content is undefined.
        """
        if not isinstance(is_left, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='is_left', type=VertexProperty.__name__))
        return java_handler(self._graph.isBipartiteGraph, [is_left._prop])

    def sparsify(
        self,
        sparsification: float,
        vertex_properties: bool = True,
        edge_properties: bool = True,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Sparsify the given graph and returns a new graph with less edges.

        :param sparsification:  The sparsification coefficient. Must be between
            0.0 and 1.0..
        :param vertex_properties:   List of vertex properties belonging to
            graph specified to be kept in the new graph
        :param edge_properties:  List of edge properties belonging to graph
            specified to be kept in the new graph
        :param name:  Filtered graph name
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        new_graph = java_handler(self._graph.sparsify, [vp, ep, sparsification, name])
        return PgxGraph(self.session, new_graph)

    def filter(
        self,
        graph_filter: Union[str, VertexFilter, EdgeFilter],
        vertex_properties: bool = True,
        edge_properties: bool = True,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Create a subgraph of this graph.

        To create the subgraph, a given filter expression is used to determine
        which parts of the graph will be part of the subgraph.

        :param graph_filter:  Object representing a filter expression that is
            applied to create the subgraph
        :param vertex_properties:   List of vertex properties belonging to graph
            specified to be kept in the new graph
        :param edge_properties:  List of edge properties belonging to graph
            specified to be kept in the new graph
        :param name:  Filtered graph name
        """
        if not isinstance(graph_filter, GraphFilter):
            raise TypeError(ARG_MUST_BE.format(arg='graph_filter', type=GraphFilter.__name__))
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        new_graph = java_handler(self._graph.filter, [vp, ep, graph_filter._filter, name])
        return PgxGraph(self.session, new_graph)

    def clone(
        self,
        vertex_properties: bool = True,
        edge_properties: bool = True,
        name: Optional[str] = None,
    ) -> "PgxGraph":
        """Return a copy of this graph.

        :param vertex_properties: List of vertex properties belonging to graph
            specified to be cloned as well
        :param edge_properties:  List of edge properties belonging to graph
            specified to be cloned as well
        :param name:  Name of the new graph
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        all_filter = VertexFilter("true")
        new_graph = java_handler(self._graph.filter, [vp, ep, all_filter._filter, name])
        return PgxGraph(self.session, new_graph)

    def create_path(
        self,
        src: PgxVertex,
        dst: PgxVertex,
        cost: EdgeProperty,
        parent: VertexProperty,
        parent_edge: VertexProperty,
    ) -> PgxPath:
        """
        :param src: Source vertex of the path
        :param dst: Destination vertex of the path
        :param cost: Property holding the edge costs. If null, the resulting
            cost will equal the hop distance.
        :param parent: Property holding the parent vertices for each vertex of
            the shortest path. For example, if the shortest path is A -> B -> C,
            then parent[C] -> B and parent[B] -> A.
        :param parent_edge: Property holding the parent edges for each vertex of
            the shortest path
        :return: The PgxPath object
        """
        if not isinstance(src, PgxVertex):
            raise TypeError(ARG_MUST_BE.format(arg='src', type=PgxVertex.__name__))
        if not isinstance(dst, PgxVertex):
            raise TypeError(ARG_MUST_BE.format(arg='dst', type=PgxVertex.__name__))
        if not isinstance(cost, EdgeProperty):
            raise TypeError(ARG_MUST_BE.format(arg='cost', type=EdgeProperty.__name__))
        if not isinstance(parent, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='parent', type=VertexProperty.__name__))
        if not isinstance(parent_edge, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='parent_edge', type=VertexProperty.__name__))

        java_pgx_path = java_handler(
            self._graph.createPath,
            [src._prop, dst._prop, cost._prop, parent._prop, parent_edge._prop],
        )
        return PgxPath(self, java_pgx_path)

    def create_all_paths(
        self,
        src: Union[str, PgxVertex],
        cost: Optional[Union[str, EdgeProperty]],
        dist: Union[VertexProperty, str],
        parent: Union[VertexProperty, str],
        parent_edge: Union[VertexProperty, str],
    ) -> AllPaths:
        """
        Create an `AllPaths` object representing all the shortest paths from a single source
        to all the possible destinations (shortest regarding the given edge costs).

        :param src: Source vertex of the path
        :param cost: Property holding the edge costs. If None, the resulting
            cost will equal the hop distance
        :param dist: Property holding the distance to the source vertex for each vertex in the
            graph
        :param parent: Property holding the parent vertices of all the shortest paths
            For example, if the shortest path is A -> B -> C, then parent[C] -> B and
            parent[B] -> A
        :param parent_edge: Property holding the parent edges for each vertex of the shortest path
        :return: The `AllPaths` object
        """
        if not isinstance(src, PgxVertex):
            raise TypeError(ARG_MUST_BE.format(arg='src', type=PgxVertex.__name__))
        if cost is not None and not isinstance(cost, EdgeProperty):
            raise TypeError(ARG_MUST_BE.format(arg='cost', type=EdgeProperty.__name__))
        if not isinstance(dist, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='dist', type=VertexProperty.__name__))
        if not isinstance(parent, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='parent', type=VertexProperty.__name__))
        if not isinstance(parent_edge, VertexProperty):
            raise TypeError(ARG_MUST_BE.format(arg='parent_edge', type=VertexProperty.__name__))

        java_all_paths = java_handler(
            self._graph.createAllPaths,
            [
                src._vertex,
                None if cost is None else cost._prop,
                dist._prop,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return AllPaths(self._graph, java_all_paths)

    def query_pgql(self, query: str) -> PgqlResultSet:
        """Submit a pattern matching select only query.

        :param query:  Query string in PGQL
        :returns: PgqlResultSet with the result
        """
        query_res = java_handler(self._graph.queryPgql, [query])
        return PgqlResultSet(self, query_res)

    def rename(self, name: str) -> None:
        """Rename this graph.

        :param name: New name
        """
        java_handler(self._graph.rename, [name])
        self.name = name

    def publish(
        self,
        vertex_properties: Union[List[VertexProperty], bool] = None,
        edge_properties: Union[List[EdgeProperty], bool] = None
    ) -> None:
        """Publish the graph so it can be shared between sessions.

        This moves the graph name from the private into the public namespace.

        :param vertex_properties: List of vertex properties belonging to graph
            specified to be published as well
        :param edge_properties: List of edge properties belonging to graph
            specified by graph to be published as well
        """
        vp, ep = self._create_hash_sets(vertex_properties, edge_properties)
        java_handler(self._graph.publish, [vp, ep])

    @property
    def is_published(self) -> bool:
        """Check if this graph is published with snapshots."""
        return self._graph.isPublished()

    def combine_vertex_properties_into_vector_property(
        self, properties: List[Union[VertexProperty, str]], name: Optional[str] = None
    ) -> VertexProperty:
        """Take a list of scalar vertex properties of same type and create a new vertex vector
        property by combining them.

        The dimension of the vector property will be equals to the number of properties.

        :param properties:  List of scalar vertex properties
        :param name:  Name for the vector property. If not null, vector property
            will be named. If that results in a name conflict, the returned future
            will complete exceptionally.
        """
        props = autoclass('java.util.ArrayList')()
        for prop in properties:
            if not isinstance(prop, VertexProperty):
                raise TypeError(
                    ARG_MUST_BE.format(arg='props', type='list of ' + VertexProperty.__name__)
                )
            props.add(prop._prop)
        vprop = java_handler(self._graph.combineVertexPropertiesIntoVectorProperty, [props, name])
        return VertexProperty(self, vprop)

    def combine_edge_properties_into_vector_property(
        self, properties: List[Union[EdgeProperty, str]], name: Optional[str] = None
    ) -> EdgeProperty:
        """Take a list of scalar edge properties of same type and create a new edge vector
        property by combining them.

        The dimension of the vector property will be equals to the number of properties.

        :param properties: List of scalar edge properties
        :param name:  Name for the vector property. If not null, vector
            property will be named. If that results in a name conflict,
            the returned future will complete exceptionally.
        """
        props = autoclass('java.util.ArrayList')()
        for prop in properties:
            if not isinstance(prop, EdgeProperty):
                raise TypeError(
                    ARG_MUST_BE.format(arg='props', type='list of ' + EdgeProperty.__name__)
                )
            props.add(prop._prop)
        vprop = java_handler(self._graph.combineEdgePropertiesIntoVectorProperty, [props, name])
        return EdgeProperty(self, vprop)

    def get_collections(self) -> Dict[str, PgxCollection]:
        """Retrieve all currently allocated collections associated with the graph."""
        java_collections = java_handler(self._graph.getCollections, [])
        collections: Dict[str, PgxCollection] = {}
        for c in java_collections:
            item = java_collections[c]
            if isinstance(item, collection_types['vertex_sequence']):
                collections[c] = VertexSequence(self, item)
            elif isinstance(item, collection_types['vertex_set']):
                collections[c] = VertexSet(self, item)
            elif isinstance(item, collection_types['edge_sequence']):
                collections[c] = EdgeSequence(self, item)
            elif isinstance(item, collection_types['edge_set']):
                collections[c] = EdgeSet(self, item)
        return collections

    def _create_hash_sets(
        self,
        vertex_properties: Union[List[VertexProperty], bool],
        edge_properties: Union[List[EdgeProperty], bool],
    ) -> Tuple[Any, Any]:
        """
        :param vertex_properties: List of vertex properties belonging to graph
            specified to be published as well
        :param edge_properties: List of edge properties belonging to graph
            specified by graph to be published as well
        """
        vp = autoclass('java.util.HashSet')()
        if isinstance(vertex_properties, bool):
            vp = vertex_props[vertex_properties]
        elif isinstance(vertex_properties, Iterable):
            for prop in vertex_properties:
                if not isinstance(prop, VertexProperty):
                    raise TypeError(
                        ARG_MUST_BE.format(
                            arg='vertex_properties', type='list of ' + VertexProperty.__name__
                        )
                    )
                vp.add(prop._prop)
        ep = autoclass('java.util.HashSet')()
        if isinstance(edge_properties, bool):
            ep = edge_props[edge_properties]
        elif isinstance(edge_properties, Iterable):
            for prop in edge_properties:
                if not isinstance(prop, EdgeProperty):
                    raise TypeError(
                        ARG_MUST_BE.format(
                            arg='edge_properties', type='list of ' + EdgeProperty.__name__
                        )
                    )
                ep.add(prop._prop)
        return (vp, ep)

    def create_change_set(
        self,
        vertex_id_generation_strategy: str = 'user_ids',
        edge_id_generation_strategy: str = 'auto_generated',
    ) -> "GraphChangeSet":
        """Create a change set for updating the graph.

        Uses auto generated IDs for the edges.

        .. note:: This is currently not supported for undirected graphs.

        :return: an empty change set
        :rtype: GraphChangeSet
        """

        # NOTE: The import of GraphChangeSet needs to be deferred as otherwise a
        # circular import is generated when executing 'import pypgx'. This
        # would be the import dependency circle:
        # 'PgxGraph' -> 'GraphChangeSet' -> 'GraphBuilder' -> 'PgxGraph'
        from pypgx.api._graph_change_set import GraphChangeSet

        java_v_strategy = id_generation_strategies[vertex_id_generation_strategy]
        java_e_strategy = id_generation_strategies[edge_id_generation_strategy]
        java_change_set = java_handler(
            self._graph.createChangeSet, [java_v_strategy, java_e_strategy]
        )

        return GraphChangeSet(self.session, java_change_set, self.vertex_id_type)

    def prepare_pgql(self, pgql_query: str) -> "PreparedStatement":
        """Prepare a PGQL query.

        :param pgql_query: Query string in PGQL
        :return: A prepared statement object
        """
        from pypgx.api._prepared_statement import PreparedStatement

        java_prepared_statement = java_handler(self._graph.preparePgql, [pgql_query])
        return PreparedStatement(java_prepared_statement)

    def execute_pgql(self, pgql_query: str) -> Optional[PgqlResultSet]:
        """(BETA) Blocking version of cloneAndExecutePgqlAsync(String).

        Calls cloneAndExecutePgqlAsync(String) and waits for the returned PgxFuture to complete.

        throws InterruptedException if the caller thread gets interrupted while
        waiting for completion.

        throws ExecutionException if any exception occurred during asynchronous
        execution. The actual exception will be nested.

        :param pgql_query: Query string in PGQL
        :return: The query result set as PgqlResultSet object
        """
        java_pgql_result_set = java_handler(self._graph.executePgql, [pgql_query])
        graph = PgxGraph(self, self._graph)

        if java_pgql_result_set is None:
            return None
        return PgqlResultSet(graph, java_pgql_result_set)

    def explain_pgql(self, pgql_query: str) -> Operation:
        """Explain the execution plan of a pattern matching query.

        Note: Different PGX versions may return different execution plans.

        :param pgql_query: Query string in PGQL
        :return: The query plan
        """
        java_operation = java_handler(self._graph.explainPgql, [pgql_query])
        return Operation(java_operation)

    def clone_and_execute_pgql(self, pgql_query: str) -> "PgxGraph":
        """Create a deep copy of the graph, and execute on it the pgql query.

        :param pgql_query: Query string in PGQL
        :return: A cloned PgxGraph with the pgql query executed

        throws InterruptedException if the caller thread gets interrupted while waiting for
            completion.
        throws ExecutionException   if any exception occurred during asynchronous execution.
            The actual exception will be nested.
        """
        java_graph = java_handler(self._graph.cloneAndExecutePgql, [pgql_query])
        return PgxGraph(self, java_graph)

    def publish_with_snapshots(self) -> None:
        """Publish the graph and all its snapshots so they can be shared between sessions."""
        java_handler(self._graph.publishWithSnapshots, [])

    def is_published_with_snapshots(self) -> bool:
        """Check if this graph is published with snapshots.

        :return: True if this graph is published, false otherwise
        """
        return java_handler(self._graph.isPublishedWithSnapshots, [])

    def destroy(self) -> None:
        """Destroy the graph with all its properties.

        After this operation, neither the graph nor its properties can be used
        anymore within this session.

        .. note:: if you have multiple :class:`PgxGraph` objects referencing the same graph
            (e.g. because you called :meth:`PgxSession.get_graph` multiple times with the
            same argument), they will ALL become invalid after calling this method;
            therefore, subsequent operations on ANY of them will result in an exception.
        """
        java_handler(self._graph.destroy, [])

    def is_pinned(self) -> bool:
        """For a published graph, indicates if the graph is pinned. A pinned graph will stay
        published even if no session is using it.
        """
        return java_handler(self._graph.isPinned, [])

    def pin(self) -> None:
        """For a published graph, pin the graph so that it stays published even if no sessions uses
        it. This call pins the graph lineage, which ensures that at least the latest available
        snapshot stays published when no session uses the graph.
        """
        java_handler(self._graph.pin, [])

    def unpin(self) -> None:
        """For a published graph, unpin the graph so that if no snapshot of the graph is used by
        any session or pinned, the graph and all its snapshots can be removed.
        """
        java_handler(self._graph.unpin, [])

    def create_picking_strategy_builder(self) -> PickingStrategyBuilder:
        """Create a new `PickingStrategyBuilder` that can be used to build a new `PickingStrategy`
        to simplify this graph.
        """
        return PickingStrategyBuilder(java_handler(self._graph.createPickingStrategyBuilder, []))

    def create_merging_strategy_builder(self) -> MergingStrategyBuilder:
        """Create a new `MergingStrategyBuilder` that can be used to build a new `MutationStrategy`
        to simplify this graph.
        """
        return MergingStrategyBuilder(java_handler(self._graph.createMergingStrategyBuilder, []))

    def create_synchronizer(
        self,
        synchronizer_class: str = "oracle.pgx.api.FlashbackSynchronizer",
        connection: Any = None,
        invalid_change_policy: Optional[str] = None,
    ) -> Synchronizer:
        """Create a synchronizer object which can be used to keep this graph in sync with changes
        happening in its original data source. Only partitioned graphs with all providers loaded
        from Oracle Database are supported.

        Possible invalid_change_policy types are: ['ignore', 'ignore_and_log',
                                                   'ignore_and_log_once', 'error']

        :param synchronizer_class: string representing java class including package, currently
            'oracle.pgx.api.FlashbackSynchronize' is the only existent option
        :param connection: the connection object to the RDBMS (# TODO : GM-28504)
        :param invalid_change_policy: sets the `OnInvalidChange` parameter to the Synchronizer
            `ChangeSet`
        :return: a synchronizer
        """
        if isinstance(synchronizer_class, str):
            synchronizer_class = autoclass(synchronizer_class)
        if connection is not None and invalid_change_policy is not None:
            if invalid_change_policy not in on_invalid_change_types:
                raise ValueError(
                    INVALID_OPTION.format(
                        var='invalid_change_policy', opts=list(on_invalid_change_types.keys())
                    )
                )
            t = on_invalid_change_types[invalid_change_policy]
            return Synchronizer(
                java_handler(self._graph.createSynchronizer, [synchronizer_class, connection, t])
            )
        if connection is None and invalid_change_policy is not None:
            raise ValueError(
                "an invalid_change_policy can only be used if the connection parameter is set"
            )
        if connection is not None:
            return Synchronizer(
                java_handler(self._graph.createSynchronizer, [synchronizer_class, connection])
            )
        return Synchronizer(java_handler(self._graph.createSynchronizer, [synchronizer_class]))

    def alter_graph(self) -> GraphAlterationBuilder:
        """Create a graph alteration builder to define the graph schema alterations to perform on
        the graph.

        :return: an empty graph alteration builder
        """
        return GraphAlterationBuilder(java_handler(self._graph.alterGraph, []))

    def get_redaction_rules(
        self, authorization_type: str, name: str
    ) -> List[PgxRedactionRuleConfig]:
        """Get the redaction rules for an `authorization_type` name.

        Possible authorization types are: ['user', 'role']

        :param authorization_type: the authorization type of the rules to be returned
        :param name: the name of the user or role for which the rules should be returned
        :return: a list of redaction rules for the given name of type `authorization_type`
        """
        if authorization_type not in authorization_types:
            raise ValueError(
                INVALID_OPTION.format(
                    var='authorization_type', opts=list(authorization_types.keys())
                )
            )
        t = authorization_types[authorization_type]
        java_redaction_rules = java_handler(self._graph.getRedactionRules, [t, name])
        redaction_rules = []
        for rule in java_redaction_rules:
            redaction_rules.append(PgxRedactionRuleConfig(rule))
        return redaction_rules

    def add_redaction_rule(
        self, redaction_rule_config: PgxRedactionRuleConfig, authorization_type: str, *names: str
    ) -> None:
        """Add a redaction rule for `authorization_type` names.

        Possible authorization types are: ['user', 'role']

        :param authorization_type: the authorization type of the rule to be added
        :param names: the names of the users or roles for which the rule should be added
        """
        if authorization_type not in authorization_types:
            raise ValueError(
                INVALID_OPTION.format(
                    var='authorization_type', opts=list(authorization_types.keys())
                )
            )
        t = authorization_types[authorization_type]
        java_handler(
            self._graph.addRedactionRule, [redaction_rule_config._redaction_rule_config, t, *names]
        )

    def remove_redaction_rule(
        self, redaction_rule_config: PgxRedactionRuleConfig, authorization_type: str, *names: str
    ) -> None:
        """Remove a redaction rule for `authorization_type` names.

        Possible authorization types are: ['user', 'role']

        :param authorization_type: the authorization type of the rule to be removed
        :param names: the names of the users or roles for which the rule should be removed
        """
        if authorization_type not in authorization_types:
            raise ValueError(
                INVALID_OPTION.format(
                    var='authorization_type', opts=list(authorization_types.keys())
                )
            )
        t = authorization_types[authorization_type]
        java_handler(
            self._graph.removeRedactionRule,
            [redaction_rule_config._redaction_rule_config, t, *names],
        )

    def grant_permission(
        self, permission_entity: PermissionEntity, pgx_resource_permission: str
    ) -> None:
        """Grant a permission on this graph to the given entity.

        Possible `PGXResourcePermission` types are: ['none', 'read', 'write', 'export', 'manage']
        Possible `PermissionEntity` objects are: `PgxUser` and `PgxRole`.

        Cannont grant 'manage'.

        :param permission_entity: the entity the rule is granted to
        :param pgx_resource_permission: the permission type
        """
        java_permission_entity = permission_entity._permission_entity
        if pgx_resource_permission not in pgx_resource_permissions:
            raise ValueError(
                INVALID_OPTION.format(
                    var='pgx_resource_permission', opts=list(pgx_resource_permissions.keys())
                )
            )
        t = pgx_resource_permissions[pgx_resource_permission]
        java_handler(self._graph.grantPermission, [java_permission_entity, t])

    def revoke_permission(self, permission_entity: PermissionEntity) -> None:
        """Revoke all permissions on this graph from the given entity.

        Possible `PermissionEntity` objects are: `PgxUser` and `PgxRole`.

        :param permission_entity: the entity for which all permissions will be revoked
        """
        java_permission_entity = permission_entity._permission_entity
        java_handler(self._graph.revokePermission, [java_permission_entity])

    def __repr__(self) -> str:
        return "{}(name: {}, v: {}, e: {}, directed: {}, memory(Mb): {})".format(
            self.__class__.__name__,
            self.name,
            self.num_vertices,
            self.num_edges,
            self.is_directed,
            self.memory_mb,
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._graph.equals(other._graph)


class BipartiteGraph(PgxGraph):
    """A bipartite PgxGraph."""

    _java_class = 'oracle.pgx.api.BipartiteGraph'

    def __init__(self, session: "PgxSession", java_graph) -> None:
        super().__init__(session, java_graph)

    def get_is_left_property(self) -> VertexProperty:
        """Get the 'is Left' vertex property of the graph."""
        is_left_prop = java_handler(self._graph.getIsLeftProperty, [])
        return VertexProperty(self, is_left_prop)
