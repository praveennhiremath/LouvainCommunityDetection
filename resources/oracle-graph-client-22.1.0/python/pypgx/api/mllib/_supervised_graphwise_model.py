#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from pypgx._utils.error_handling import java_handler

from pypgx.api.frames import PgxFrame
from pypgx.api.mllib import GraphWisePredictionLayerConfig
from pypgx.api.mllib._gnn_explanation import SupervisedGnnExplanation
from pypgx.api.mllib._graphwise_model import GraphWiseModel
from pypgx.api.mllib._model_utils import ModelStorer
from pypgx._utils.item_converter import (
    convert_python_to_java_vertex_list,
    convert_python_to_java_vertex,
)
from jnius import autoclass, cast
from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_graph import PgxGraph
from typing import List, Union, Dict, Iterable


class SupervisedGraphWiseModel(GraphWiseModel):
    """SupervisedGraphWise model object."""

    _java_class = 'oracle.pgx.api.mllib.SupervisedGraphWiseModel'

    def get_prediction_layer_configs(self) -> GraphWisePredictionLayerConfig:
        """Get the configuration objects for the prediction layers.

        :return: configuration of the prediction layer
        :rtype: GraphWisePredictionLayerConfig
        """
        if 'pred_layer_config' not in self.params:
            self.params['pred_layer_config'] = None
        return self.params['pred_layer_config']

    def get_loss_function(self) -> str:
        """Get the loss function name.

        :return: loss function name. Can be one of SOFTMAX_CROSS_ENTROPY, SIGMOID_CROSS_ENTROPY,
            DEVNET (case insensitive)
        :rtype: str
        """
        if 'loss_fn' not in self.params:
            self.params['loss_fn'] = None
        return self.params['loss_fn']

    def get_class_weights(self) -> Dict:
        """Get the class weights.

        :return: a dictionary mapping classes to their weights.
        :rtype: dict
        """
        if 'class_weights' not in self.params:
            self.params['class_weights'] = None
        return self.params['class_weights']

    def get_vertex_target_property_name(self) -> str:
        """Get the target property name

        :return: target property name
        :rtype: str
        """
        if 'vertex_target_property_name' not in self.params:
            self.params['vertex_target_property_name'] = self._model.getVertexTargetPropertyName()
        return self.params['vertex_target_property_name']

    def store(self, path: str, key: str, overwrite: bool = False) -> None:
        """Store the model in a file.

        :param path: Path where to store the model
        :type path: str
        :param key: Encryption key
        :type key: str
        :param overwrite: Whether or not to overwrite pre-existing file
        :type overwrite: bool

        :return: None
        """
        self.check_is_fitted()
        java_handler(self._model.store, [path, key, overwrite])

    def export(self) -> ModelStorer:
        """Return a ModelStorer object which can be used to save the model.

        :returns: ModelStorer object
        :rtype: ModelStorer
        """
        return ModelStorer(self)

    def fit(self, graph: PgxGraph) -> None:
        """Fit the model on a graph.

        :param graph: Graph to fit on
        :type graph: PgxGraph

        :return: None
        """
        java_handler(self._model.fit, [graph._graph])
        self._is_fitted = True
        self.loss = self._model.getTrainingLoss()

    def infer_embeddings(
        self, graph: PgxGraph, vertices: Union[Iterable[PgxVertex], Iterable[int]]
    ) -> PgxFrame:
        """Infer the embeddings for the specified vertices

        :param graph: the graph
        :type graph: PgxGraph
        :param vertices: the vertices to infer embeddings for. Can be a list of vertices or their
            IDs.

        :returns: PgxFrame containing the embeddings for each vertex
        :rtype: PgxFrame
        """
        self.check_is_fitted()
        vids = convert_python_to_java_vertex_list(graph, vertices)
        return PgxFrame(java_handler(self._model.inferEmbeddings, [graph._graph, vids]))

    def infer_logits(
        self, graph: PgxGraph, vertices: Union[Iterable[PgxVertex], Iterable[int]]
    ) -> PgxFrame:
        """Infer the prediction logits for the specified vertices

        :param graph: the graph
        :type graph: PgxGraph
        :param vertices: the vertices to infer logits for. Can be a list of vertices or their
            IDs.

        :returns: PgxFrame containing the logits for each vertex
        :rtype: PgxFrame
        """
        self.check_is_fitted()
        vids = convert_python_to_java_vertex_list(graph, vertices)
        return PgxFrame(java_handler(self._model.inferLogits, [graph._graph, vids]))

    def infer_labels(
        self, graph: PgxGraph, vertices: Union[Iterable[PgxVertex], Iterable[int]]
    ) -> PgxFrame:
        """Infer the labels for the specified vertices

        :param graph: the graph
        :type graph: PgxGraph
        :param vertices: the vertices to infer labels for. Can be a list of vertices or their
            IDs.

        :returns: PgxFrame containing the labels for each vertex
        :rtype: PgxFrame
        """
        self.check_is_fitted()
        vids = convert_python_to_java_vertex_list(graph, vertices)
        return PgxFrame(java_handler(self._model.inferLabels, [graph._graph, vids]))

    def evaluate_labels(
        self, graph: PgxGraph, vertices: Union[Iterable[PgxVertex], Iterable[int]]
    ) -> PgxFrame:
        """Evaluate (macro averaged) classification performance statistics for the specified
        vertices.

        :param graph: the graph
        :type graph: PgxGraph
        :param vertices: the vertices to evaluate on. Can be a list of vertices or their
            IDs.

        :returns: PgxFrame containing the metrics
        :rtype: PgxFrame
        """
        self.check_is_fitted()
        vids = convert_python_to_java_vertex_list(graph, vertices)
        return PgxFrame(java_handler(self._model.evaluateLabels, [graph._graph, vids]))

    def infer_and_get_explanation(
        self, graph: PgxGraph, vertex: Union[PgxVertex, int]
    ) -> SupervisedGnnExplanation:
        """Perform inference on the specified vertex and generate an explanation that contains
        scores of how important each property and each vertex in the computation graph is for the
        prediction.

        :param graph: the graph
        :type graph: PgxGraph
        :param vertex: the vertex or its ID
        :type vertex: PgxVertex or int

        :returns: explanation containing feature importance and vertex importance.
        :rtype: SupervisedGnnExplanation
        """
        self.check_is_fitted()
        java_vertex = convert_python_to_java_vertex(graph, vertex)

        # fix for edge case GM-27791
        # pyjnius converts java Booleans to integers. In order to cast them back to bools in
        # explanation.get_label, we need to pass the information whether the label should be bool
        java_bool_property_type = autoclass('oracle.pgx.common.types.PropertyType').BOOLEAN
        model_config = cast(
            'oracle.pgx.config.mllib.SupervisedGraphWiseModelConfig', self._model.getConfig()
        )
        bool_label = model_config.getLabelType() == java_bool_property_type

        return SupervisedGnnExplanation(
            java_handler(self._model.inferAndGetExplanation, [graph._graph, java_vertex]),
            bool_label=bool_label,
        )

    def _get_col_names(self, vec) -> List[str]:
        result_elements = vec.getPgqlResultElements()
        col_names = []
        for idx in range(result_elements.size()):
            col_names.append(result_elements.get(idx).getVarName())
        return col_names
