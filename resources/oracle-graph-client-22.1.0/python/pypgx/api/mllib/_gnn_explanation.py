#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from pypgx.api._pgx_graph import PgxGraph
from pypgx.api._property import VertexProperty
from pypgx._utils.error_handling import java_handler
from typing import Dict, List, Any


class GnnExplanation:
    """GnnExplanation object"""

    _java_class = 'oracle.pgx.api.mllib.GnnExplanation'

    def __init__(self, java_gnn_explanation) -> None:
        self._explanation = java_gnn_explanation

    def get_vertex_feature_importance(self) -> Dict[VertexProperty, float]:
        """Get the feature importances as a map from property to importance value.

        :returns: the feature importances.
        """
        java_importance_map = java_handler(self._explanation.getVertexFeatureImportance, [])
        importance_map = {}
        for java_prop in java_importance_map:
            importance_map[VertexProperty._from_java(java_prop)] = java_importance_map[java_prop]
        return importance_map

    def get_importance_graph(self) -> PgxGraph:
        """Get the importance Graph, that is, the computation graph with an additional vertex
        property indicating vertex importance. The additional importance property can be retrieved
        via get_vertex_importance_property.

        :returns: the importance graph
        :rtype: PgxGraph
        """
        from pypgx.api._pgx_session import PgxSession  # need to import here to avoid import loop

        java_graph = java_handler(self._explanation.getImportanceGraph, [])
        java_session = java_handler(java_graph.getSession, [])
        return PgxGraph(PgxSession(java_session), java_graph)

    def get_vertex_importance_property(self) -> VertexProperty:
        """Get the vertex property that contains the computed vertex importance.

        :returns: the vertex importance property
        """
        java_prop = java_handler(self._explanation.getVertexImportanceProperty, [])
        return VertexProperty._from_java(java_prop)

    def get_embedding(self) -> List[float]:
        """Get the inferred embedding of the specified vertex.

        :returns: the embedding
        """
        return java_handler(self._explanation.getEmbedding, [])


class SupervisedGnnExplanation(GnnExplanation):
    """SupervisedGnnExplanation object"""

    _java_class = 'oracle.pgx.api.mllib.SupervisedGnnExplanation'

    def __init__(self, java_supervised_gnn_explanation, bool_label: bool) -> None:
        super().__init__(java_supervised_gnn_explanation)
        self._explanation = java_supervised_gnn_explanation

        # java Boolean are not converted to python booleans by pyjnius => we have to cast sometimes
        # see GM-28290
        self._bool_label = bool_label

    def get_logits(self) -> List[float]:
        """Get the inferred logits of the specified vertex.

        :returns: the logits
        """
        return java_handler(self._explanation.getLogits, [])

    def get_label(self) -> Any:
        """Get the inferred label of the specified vertex.

        :returns: the label
        """
        label = java_handler(self._explanation.getLabel, [])
        if self._bool_label:  # edge case as pyjnius converts java Boolean to integers
            label = bool(label)
        return label
