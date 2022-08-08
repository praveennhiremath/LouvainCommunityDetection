#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

"""Public API for the PGX client.

Classes found in the ``pypgx.api`` package and its subpackages should typically
not be directly instantiated by the user. Instead, they are returned by functions,
instance methods, and in some cases, class methods."""

import sys as _sys

from ._all_paths import AllPaths
from ._analyst import Analyst
from ._compiled_program import CompiledProgram
from ._graph_alteration_builder import GraphAlterationBuilder
from ._graph_builder import EdgeBuilder, GraphBuilder, VertexBuilder
from ._graph_change_set import EdgeModifier, GraphChangeSet, VertexModifier
from ._graph_config import GraphConfig
from ._graph_config_factory import GraphConfigFactory
from ._graph_meta_data import GraphMetaData
from ._matrix_factorization_model import MatrixFactorizationModel
from ._mutation_strategy_builder import (
    MutationStrategyBuilder,
    MergingStrategyBuilder,
    PickingStrategyBuilder
)
from ._namespace import (
    Namespace,
    NAMESPACE_PRIVATE,
    NAMESPACE_PUBLIC
)
from ._partition import PgxPartition
from ._pgql_result_set import PgqlResultSet
from ._pgx import Pgx
from ._pgx_collection import (
    EdgeCollection,
    EdgeSequence,
    EdgeSet,
    PgxCollection,
    ScalarCollection,
    ScalarSequence,
    ScalarSet,
    VertexCollection,
    VertexSequence,
    VertexSet,
)
from ._pgx_entity import PgxEdge, PgxEntity, PgxVertex
from ._pgx_graph import BipartiteGraph, PgxGraph
from ._pgx_map import PgxMap
from ._pgx_path import PgxPath
from ._pgx_session import PgxSession
from ._prepared_statement import PreparedStatement
from ._property import EdgeLabel, EdgeProperty, VertexProperty, VertexLabels
from ._scalar import Scalar
from ._server_instance import ServerInstance
from ._synchronizer import Synchronizer, FlashbackSynchronizer


__all__ = [name for name in dir() if not name.startswith('_')]
# Deprecated attributes imported below this line.

from pypgx.api.frames import (
    PgxCsvFrameReader,
    PgxCsvFrameStorer,
    PgxDbFrameReader,
    PgxDbFrameStorer,
    PgxFrame,
    PgxFrameBuilder,
    PgxFrameColumn,
    PgxGenericFrameReader,
    PgxGenericFrameStorer,
    PgxPgbFrameReader,
    PgxPgbFrameStorer,
)

from pypgx._utils.deprecation import DeprecatedAttribute as _DeprecatedAttribute, Module as _Module

_deprecations = {
    name: _DeprecatedAttribute(new_name="pypgx.api.frames." + name, since_version="21.4")
    for name in
    [
        "PgxCsvFrameReader",
        "PgxCsvFrameStorer",
        "PgxDbFrameReader",
        "PgxDbFrameStorer",
        "PgxFrame",
        "PgxFrameBuilder",
        "PgxFrameColumn",
        "PgxGenericFrameReader",
        "PgxGenericFrameStorer",
        "PgxPgbFrameReader",
        "PgxPgbFrameStorer",
    ]
}
_sys.modules[__name__] = _Module(_sys.modules[__name__], _deprecations)
