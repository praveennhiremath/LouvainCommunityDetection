#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import List, Optional, Union

from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import INVALID_OPTION
from pypgx._utils.item_converter import convert_to_java_list
from pypgx._utils.pgx_types import (
    label_merging_functions,
    merging_functions,
    mode,
    multi_edges,
    picking_functions,
    self_edges,
    trivial_vertices,
)
from pypgx.api._mutation_strategy import MutationStrategy
from pypgx.api._pgx_id import PgxId
from pypgx.api._property import EdgeProperty
from pypgx.api._property import VertexProperty


class MutationStrategyBuilder:
    """A class for defining a mutation strategy on a graph."""

    _java_class = 'oracle.pgx.api.MutationStrategyBuilder'

    def __init__(self, java_mutation_strategy_builder) -> None:
        self._mutation_strategy_builder = java_mutation_strategy_builder

    def set_new_graph_name(self, new_graph_name: Optional[str]) -> "MutationStrategyBuilder":
        """Set a new graph name. If None, a new graph name will be generated.

        :param new_graph_name: a new graph name
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setNewGraphName,
            [new_graph_name]
        )
        return self

    def set_copy_mode(self, copy_mode: bool) -> "MutationStrategyBuilder":
        """Define whether the mutation should occur on the original graph or on a copy.

        If set to True, the mutation will occur on the original graph without creating a new
        instance. If set to False, a new graph instance will be created. The default copy mode is
        False.

        :param copy_mode: whether to mutate the original graph or create a new one
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setCopyMode,
            [mode[copy_mode]]
        )
        return self

    def set_trivial_vertices(self, keep_trivial_vertices: bool) -> "MutationStrategyBuilder":
        """Define if isolated nodes should be kept in the result.

        By default (without calling this), isolated nodes will be kept.

        :param keep_trivial_vertices: whether to keep or remove trivial vertices in the result
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setTrivialVertices,
            [trivial_vertices[keep_trivial_vertices]]
        )
        return self

    def set_self_edges(self, keep_self_edges: bool) -> "MutationStrategyBuilder":
        """Define if self edges should be kept in the result.

        By default (without calling this), self edges will be removed.

        :param copy_self_edges: whether to keep or remove self edges in the result
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setSelfEdges,
            [self_edges[keep_self_edges]]
        )
        return self

    def set_multi_edges(self, keep_multi_edges: bool) -> "MutationStrategyBuilder":
        """Define if multi edges should be kept in the result.

        By default (without calling this), multi edges will be removed.

        :param copy_multi_edges: whether to keep or remove multi edges in the result
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setMultiEdges,
            [multi_edges[keep_multi_edges]]
        )
        return self

    def set_kept_vertex_properties(
        self,
        props_to_keep: List[VertexProperty]
    ) -> "MutationStrategyBuilder":
        """Set vertex properties that will be kept after the mutation.

        By default (without calling this), all vertex properties will be kept.

        :param props_to_keep: list of `VertexProperty` objects to keep
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setKeptVertexProperties,
            [convert_to_java_list([prop._prop for prop in props_to_keep])]
        )
        return self

    def set_kept_edge_properties(
        self,
        props_to_keep: List[EdgeProperty]
    ) -> "MutationStrategyBuilder":
        """Set edge properties that will be kept after the mutation.

        By default (without calling this), all edge properties will be kept.

        :param props_to_keep: list of `EdgeProperty` objects to keep
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setKeptEdgeProperties,
            [convert_to_java_list([prop._prop for prop in props_to_keep])]
        )
        return self

    def drop_vertex_property(self, vertex_property: VertexProperty) -> "MutationStrategyBuilder":
        """Set a vertex property that will be dropped after the mutation.

        By default (without calling this), all vertex properties will be kept.

        :param vertex_property: `VertexProperty` object to drop
        :returns: the `MutationStrategyBuilder` instance itself
        """
        java_vertex_prop = vertex_property._prop
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.dropVertexProperty,
            [java_vertex_prop]
        )
        return self

    def drop_edge_property(self, edge_property: EdgeProperty) -> "MutationStrategyBuilder":
        """Set an edge property that will be dropped after the mutation.

        By default (without calling this), all edge properties will be kept.

        :param edge_property: `EdgeProperty` object to drop
        :returns: the `MutationStrategyBuilder` instance itself
        """
        java_edge_prop = edge_property._prop
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.dropEdgeProperty,
            [java_edge_prop]
        )
        return self

    def drop_vertex_properties(
        self,
        vertex_properties: List[VertexProperty]
    ) -> "MutationStrategyBuilder":
        """Set vertex properties that will be dropped after the mutation.

        By default (without calling this), all edge properties will be kept.

        :param vertex_properties: list of `VertexProperty` objects to drop
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.dropVertexProperties,
            [convert_to_java_list([prop._prop for prop in vertex_properties])]
        )
        return self

    def drop_edge_properties(
        self,
        edge_properties: List[EdgeProperty]
    ) -> "MutationStrategyBuilder":
        """Set edge properties that will be dropped after the mutation.
         By default (without calling this) all edgeProperties will be kept.

        :param edge_properties: list of `EdgeProperty` objects to drop
        :returns: the `MutationStrategyBuilder` instance itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.dropEdgeProperties,
            [convert_to_java_list([prop._prop for prop in edge_properties])]
        )
        return self

    def build(self) -> MutationStrategy:
        """Build the `MutationSrategy` object with the chosen parameters.

        Parameters that were not set, are instantiated with
        default values.

        :returns: a `MutationStrategy` instance with the chosen parameters
        """
        java_mutation_strategy = java_handler(self._mutation_strategy_builder.build, [])
        return MutationStrategy(java_mutation_strategy)


class MergingStrategyBuilder(MutationStrategyBuilder):
    """A class for defining a merging strategy on a graph."""

    _java_class = 'oracle.pgx.api.MergingStrategyBuilder'

    def set_property_merging_strategy(
        self,
        prop: Union[str, PgxId, EdgeProperty],
        merging_function: str
    ) -> "MergingStrategyBuilder":
        """Define a merging function for the given edge property.

        All properties, where no `merging_function` was defined will be merged using the "max"
        function.

        This strategy can be used to merge the properties of multi-edges.
        PGX allows the user to define a `merging_function` for every property.

        :param prop: a property name, `PgxId`, or `EdgeProperty`
        :param merging_function: available functions are: "min", "max", and "sum"
        :returns: the `MergingStrategyBuilder` itself
        """
        if merging_function in merging_functions.keys():
            java_prop = prop._prop if isinstance(prop, EdgeProperty) else \
                prop._id if isinstance(prop, PgxId) else prop
            self._mutation_strategy_builder = java_handler(
                self._mutation_strategy_builder.setPropertyMergingStrategy,
                [java_prop, merging_functions[merging_function]]
            )
            return self
        else:
            raise ValueError(
                INVALID_OPTION.format(var='merging_function', opts="min, max, or sum")
            )

    def set_keep_user_defined_edge_keys(
        self,
        keep_user_defined_edge_keys: bool
    ) -> "MergingStrategyBuilder":
        """If set to True, the user-defined edge keys are kept as far as possible.

        If multiple edges A and B are merged into one edge, a new key is generated for this edge.

        :param keep_user_defined_edge_keys: whether to keep user-defined edge keys
        :returns: the `MergingStrategyBuilder` itself
        """
        self._mutation_strategy_builder = java_handler(
            self._mutation_strategy_builder.setKeepUserDefinedEdgeKeys,
            [keep_user_defined_edge_keys]
        )
        return self

    def set_label_merging_strategy(self, label_merging_function: str) -> "MergingStrategyBuilder":
        """Define a merging function for the edge labels.

        By default (without calling this), the labels will be merged using the "max" function

        :param label_merging_function: available functions are: "min" and "max"
        :returns: the `MergingStrategyBuilder` itself
        """
        if label_merging_function in label_merging_functions.keys():
            self._mutation_strategy_builder = java_handler(
                self._mutation_strategy_builder.setLabelMergingStrategy,
                [label_merging_functions[label_merging_function]]
            )
            return self
        else:
            raise ValueError(
                INVALID_OPTION.format(var='label_merging_function', opts="min or max")
            )


class PickingStrategyBuilder(MutationStrategyBuilder):
    """A class for defining a picking strategy on a graph."""

    _java_class = 'oracle.pgx.api.PickingStrategyBuilder'

    def set_pick_by_property(
        self,
        prop: Union[str, PgxId, EdgeProperty],
        picking_function: str
    ) -> "PickingStrategyBuilder":
        """If there are multiple edges between two vertices, the edge that satisfies the
        `picking_function` will be picked.

        For instance, pick the edge where property_1 is MAX.
        Note, if two or more properties could be chosen according to the `picking_function`,
        only one of them is picked.

        :param prop: a property name, `PgxId`, or `EdgeProperty`
        :param picking_function: available functions are: "min" and "max"
        :returns: the `PickingStrategyBuilder` itself
        """
        if picking_function in picking_functions.keys():
            java_prop = prop._prop if isinstance(prop, EdgeProperty) else \
                prop._id if isinstance(prop, PgxId) else prop
            self._mutation_strategy_builder = java_handler(
                self._mutation_strategy_builder.setPickByProperty,
                [java_prop, picking_functions[picking_function]]
            )
            return self
        else:
            raise ValueError(
                INVALID_OPTION.format(var='picking_strategy_function', opts="min or max")
            )

    def set_pick_by_label(self, picking_function: str) -> "PickingStrategyBuilder":
        """If there are multiple edges between two vertices, the edge whose label
        satisfies the `picking_function` will be picked.

        :param picking_function: available functions are: "min" and "max"
        :returns: the `PickingStrategyBuilder` itself
        """
        if picking_function in picking_functions.keys():
            self._mutation_strategy_builder = java_handler(
                self._mutation_strategy_builder.setPickByLabel,
                [picking_functions[picking_function]]
            )
            return self
        else:
            raise ValueError(
                INVALID_OPTION.format(var='picking_strategy_function', opts="min or max")
            )

    def set_pick_by_edge_id(self, picking_function: str) -> "PickingStrategyBuilder":
        """If there are multiple edges between two vertices, the edge that satisfies the
        `picking_function` will be picked.

        :param picking_function: available functions are: "min" and "max"
        :returns: the `PickingStrategyBuilder` itself
        """
        if picking_function in picking_functions.keys():
            self._mutation_strategy_builder = java_handler(
                self._mutation_strategy_builder.setPickByEdgeId,
                [picking_functions[picking_function]]
            )
            return self
        else:
            raise ValueError(
                INVALID_OPTION.format(var='picking_strategy_function', opts="min or max")
            )
