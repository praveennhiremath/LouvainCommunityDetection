#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from typing import Any, Union, TYPE_CHECKING

from jnius import autoclass
from pypgx._utils.error_messages import INVALID_OPTION

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api.frames._pgx_data_types import VectorType


java_types = {
    'integer': autoclass('java.lang.Integer'),
    'long': autoclass('java.lang.Long'),
    'boolean': autoclass('java.lang.Boolean'),
    'double': autoclass('java.lang.Double'),
    'float': autoclass('java.lang.Float'),
    'string': autoclass('java.lang.String'),
}

picking_functions = {}
label_merging_functions = {}
merging_functions = {}
authorization_types = {}
id_types = {}
id_generation_strategies = {}
id_strategies = {}
on_invalid_change_types = {}
property_types = {}
format_types = {}
filter_types = {}
source_types = {}
vertex_props = {}
edge_props = {}
pgx_entities = {}
pgx_resource_permissions = {}
collection_types = {}
direction_types = {}
time_units = {}

sort_order = {}
degree_type = {}
mode = {}
self_edges = {}
multi_edges = {}
trivial_vertices = {}

vector_types = ('integer', 'long', 'double', 'float')
col_types = (
    'vertex',
    'edge',
    'point2d',
    'date',
    'time',
    'timestamp',
    'time_with_timezone',
    'timestamp_with_timezone',
    'vertex_labels',
    'array',
    'boolean',
)

point2d = autoclass('oracle.pgql.lang.spatial.Point2D')
local_date = autoclass('java.time.LocalDate')
local_time = autoclass('java.time.LocalTime')
timestamp = autoclass('java.time.LocalDateTime')
time_with_timezone = autoclass('java.time.OffsetTime')
timestamp_with_timezone = autoclass('java.time.OffsetDateTime')
legacy_date = autoclass('java.util.Date')
java_set = autoclass('java.util.Set')
java_list = autoclass('java.util.List')
array_list = autoclass('java.util.ArrayList')


def _set_up_types() -> None:
    """Set up the types in this module."""

    # The reason this function exists is to avoid have having helper variables
    # exported (this function's local variables aren't exported by the module).

    picking_function = autoclass("oracle.pgx.common.mutations.PickingStrategyFunction")
    label_merging_function = autoclass("oracle.pgx.common.mutations.LabelMergingFunction")
    merging_function = autoclass("oracle.pgx.common.mutations.MergingFunction")
    authorization_type = autoclass('oracle.pgx.common.types.AuthorizationType')
    id_type = autoclass('oracle.pgx.common.types.IdType')
    id_generation_strategy = autoclass('oracle.pgx.config.IdGenerationStrategy')
    id_strategy = autoclass('oracle.pgx.common.types.IdStrategy')
    on_invalid_change = autoclass('oracle.pgx.config.OnInvalidChange')
    pgx_resource_permission = autoclass('oracle.pgx.common.auth.PgxResourcePermission')
    property_type = autoclass('oracle.pgx.common.types.PropertyType')
    direction_type = autoclass('oracle.pgx.common.types.Direction')
    format_type = autoclass('oracle.pgx.config.Format')
    source_type = autoclass('oracle.pgx.api.GraphSource')
    vertex_property = autoclass('oracle.pgx.api.VertexProperty')
    edge_property = autoclass('oracle.pgx.api.EdgeProperty')
    time_unit = autoclass('java.util.concurrent.TimeUnit')

    picking_functions['min'] = picking_function.MIN
    picking_functions['max'] = picking_function.MAX

    label_merging_functions['min'] = label_merging_function.MIN
    label_merging_functions['max'] = label_merging_function.MAX

    merging_functions['min'] = merging_function.MIN
    merging_functions['max'] = merging_function.MAX
    merging_functions['sum'] = merging_function.SUM

    authorization_types['user'] = authorization_type.USER
    authorization_types['role'] = authorization_type.ROLE

    id_types['integer'] = id_type.INTEGER
    id_types['long'] = id_type.LONG
    id_types['string'] = id_type.STRING

    id_generation_strategies['user_ids'] = id_generation_strategy.USER_IDS
    id_generation_strategies['auto_generated'] = id_generation_strategy.AUTO_GENERATED

    id_strategies['no_ids'] = id_strategy.NO_IDS
    id_strategies['keys_as_ids'] = id_strategy.KEYS_AS_IDS
    id_strategies['unstable_generated_ids'] = id_strategy.UNSTABLE_GENERATED_IDS
    id_strategies['partitioned_ids'] = id_strategy.PARTITIONED_IDS

    on_invalid_change_types['ignore'] = on_invalid_change.IGNORE
    on_invalid_change_types['ignore_and_log'] = on_invalid_change.IGNORE_AND_LOG
    on_invalid_change_types['ignore_and_log_once'] = on_invalid_change.IGNORE_AND_LOG_ONCE
    on_invalid_change_types['error'] = on_invalid_change.ERROR

    property_types['integer'] = property_type.INTEGER
    property_types['long'] = property_type.LONG
    property_types['double'] = property_type.DOUBLE
    property_types['boolean'] = property_type.BOOLEAN
    property_types['string'] = property_type.STRING
    property_types['vertex'] = property_type.VERTEX
    property_types['edge'] = property_type.EDGE
    property_types['local_date'] = property_type.LOCAL_DATE
    property_types['time'] = property_type.TIME
    property_types['timestamp'] = property_type.TIMESTAMP
    property_types['time_with_timezone'] = property_type.TIME_WITH_TIMEZONE
    property_types['timestamp_with_timezone'] = property_type.TIMESTAMP_WITH_TIMEZONE

    direction_types['outgoing'] = direction_type.OUTGOING
    direction_types['incoming'] = direction_type.INCOMING
    direction_types['both'] = direction_type.BOTH

    format_types['pgb'] = format_type.PGB
    format_types['edge_list'] = format_type.EDGE_LIST
    format_types['two_tables'] = format_type.TWO_TABLES
    format_types['adj_list'] = format_type.ADJ_LIST
    format_types['flat_file'] = format_type.FLAT_FILE
    format_types['graphml'] = format_type.GRAPHML
    format_types['pg'] = format_type.PG
    format_types['rdf'] = format_type.RDF
    format_types['csv'] = format_type.CSV

    source_types['pg_view'] = source_type.PG_VIEW

    filter_types['vertex'] = autoclass('oracle.pgx.api.filter.VertexFilter')
    filter_types['edge'] = autoclass('oracle.pgx.api.filter.EdgeFilter')

    pgx_entities['vertex'] = autoclass('oracle.pgx.api.PgxVertex')
    pgx_entities['edge'] = autoclass('oracle.pgx.api.PgxEdge')

    pgx_resource_permissions['none'] = pgx_resource_permission.NONE
    pgx_resource_permissions['read'] = pgx_resource_permission.READ
    pgx_resource_permissions['write'] = pgx_resource_permission.WRITE
    pgx_resource_permissions['export'] = pgx_resource_permission.EXPORT
    pgx_resource_permissions['manage'] = pgx_resource_permission.MANAGE

    collection_types['vertex_sequence'] = autoclass('oracle.pgx.api.VertexSequence')
    collection_types['vertex_set'] = autoclass('oracle.pgx.api.VertexSet')
    collection_types['edge_sequence'] = autoclass('oracle.pgx.api.EdgeSequence')
    collection_types['edge_set'] = autoclass('oracle.pgx.api.EdgeSet')

    java_sort_order = autoclass('oracle.pgx.api.PgxGraph$SortOrder')
    sort_order[True] = java_sort_order.ASCENDING
    sort_order[False] = java_sort_order.DESCENDING

    java_degree = autoclass('oracle.pgx.api.PgxGraph$Degree')
    degree_type[True] = java_degree.IN
    degree_type[False] = java_degree.OUT

    java_mode = autoclass('oracle.pgx.api.PgxGraph$Mode')
    mode[True] = java_mode.MUTATE_IN_PLACE
    mode[False] = java_mode.CREATE_COPY

    java_self_edges = autoclass('oracle.pgx.api.PgxGraph$SelfEdges')
    self_edges[True] = java_self_edges.KEEP_SELF_EDGES
    self_edges[False] = java_self_edges.REMOVE_SELF_EDGES

    java_multi_edges = autoclass('oracle.pgx.api.PgxGraph$MultiEdges')
    multi_edges[True] = java_multi_edges.KEEP_MULTI_EDGES
    multi_edges[False] = java_multi_edges.REMOVE_MULTI_EDGES

    java_trivial_vertices = autoclass('oracle.pgx.api.PgxGraph$TrivialVertices')
    trivial_vertices[True] = java_trivial_vertices.KEEP_TRIVIAL_VERTICES
    trivial_vertices[False] = java_trivial_vertices.REMOVE_TRIVIAL_VERTICES

    vertex_props[True] = vertex_property.ALL
    vertex_props[False] = vertex_property.NONE

    edge_props[True] = edge_property.ALL
    edge_props[False] = edge_property.NONE

    time_units['days'] = time_unit.DAYS
    time_units['hours'] = time_unit.HOURS
    time_units['microseconds'] = time_unit.MICROSECONDS
    time_units['milliseconds'] = time_unit.MILLISECONDS
    time_units['minutes'] = time_unit.MINUTES
    time_units['nanoseconds'] = time_unit.NANOSECONDS
    time_units['seconds'] = time_unit.SECONDS


_set_up_types()

ACTIVATION_FUNCTION = autoclass('oracle.pgx.config.mllib.ActivationFunction')
ACTIVATION_FUNCTIONS = {
    'LEAKY_RELU': ACTIVATION_FUNCTION.LEAKY_RELU,
    'RELU': ACTIVATION_FUNCTION.RELU,
    'LINEAR': ACTIVATION_FUNCTION.LINEAR,
    'TANH': ACTIVATION_FUNCTION.TANH,
}

WEIGHT_INIT_SCHEME = autoclass('oracle.pgx.config.mllib.WeightInitScheme')
WEIGHT_INIT_SCHEMES = {
    'ZEROS': WEIGHT_INIT_SCHEME.ZEROS,
    'ONES': WEIGHT_INIT_SCHEME.ONES,
    'XAVIER_UNIFORM': WEIGHT_INIT_SCHEME.XAVIER_UNIFORM,
    'HE': WEIGHT_INIT_SCHEME.HE,
    'XAVIER': WEIGHT_INIT_SCHEME.XAVIER,
}

SUPERVISED_LOSS_FUNCTIONS = {
    'SOFTMAX_CROSS_ENTROPY': 'SoftmaxCrossEntropyLoss',
    'SIGMOID_CROSS_ENTROPY': 'SigmoidCrossEntropyLoss',
}

UNSUPERVISED_LOSS_FUNCTION = autoclass(
    'oracle.pgx.config.mllib.UnsupervisedGraphWiseModelConfig$LossFunction'
)
UNSUPERVISED_LOSS_FUNCTIONS = {
    'SIGMOID_CROSS_ENTROPY': UNSUPERVISED_LOSS_FUNCTION.SIGMOID_CROSS_ENTROPY,
}

READOUT_FUNCTION = autoclass('oracle.pgx.config.mllib.GraphWiseDgiLayerConfig$ReadoutFunction')
READOUT_FUNCTIONS = {"MEAN": READOUT_FUNCTION.MEAN}

DISCRIMINATOR_FUNCTION = autoclass(
    'oracle.pgx.config.mllib.GraphWiseDgiLayerConfig$Discriminator'
)
DISCRIMINATOR_FUNCTIONS = {"BILINEAR": DISCRIMINATOR_FUNCTION.BILINEAR}
BATCH_GENERATOR = autoclass('oracle.pgx.config.mllib.batchgenerator.BatchGenerator')
BATCH_GENERATORS = {
    'STANDARD': autoclass('oracle.pgx.config.mllib.batchgenerator.StandardBatchGenerator'),
    'STRATIFIED_OVERSAMPLING': autoclass(
        'oracle.pgx.config.mllib.batchgenerator.StratifiedOversamplingBatchGenerator'
    ),
}


def get_data_type(datatype: Union["VectorType", str]) -> Any:
    # Import here to prevent a circular import error.
    from pypgx.api.frames._pgx_data_types import VectorType, DataTypes

    data_classes = DataTypes._data_type_instances
    if isinstance(datatype, VectorType):
        return datatype._vector
    elif datatype not in data_classes.keys():
        raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(data_classes.keys())))
    return data_classes[datatype]
