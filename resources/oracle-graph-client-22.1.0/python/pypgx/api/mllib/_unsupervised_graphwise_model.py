#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from pypgx._utils.error_handling import java_handler

from pypgx.api.frames import PgxFrame
from pypgx.api.mllib._gnn_explanation import GnnExplanation
from pypgx.api.mllib._graphwise_model import GraphWiseModel
from pypgx.api.mllib._graphwise_dgi_layer_config import GraphWiseDgiLayerConfig
from pypgx.api.mllib._model_utils import ModelStorer
from pypgx._utils.item_converter import (
    convert_python_to_java_vertex_list,
    convert_python_to_java_vertex,
)
from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_graph import PgxGraph
from typing import Union, Iterable


class UnsupervisedGraphWiseModel(GraphWiseModel):
    """UnsupervisedGraphWise model object."""

    _java_class = 'oracle.pgx.api.mllib.UnsupervisedGraphWiseModel'

    def get_dgi_layer_config(self) -> GraphWiseDgiLayerConfig:
        """Get the configuration object for the dgi layer.

        :return: configuration
        :rtype: GraphWiseDgiLayerConfig
        """
        if 'dgi_layer_config' not in self.params:
            java_dgi_layer_config = self._model.getDgiLayerConfigs()
            self.params['dgi_layer_config'] = GraphWiseDgiLayerConfig(java_dgi_layer_config, {})
        return self.params['dgi_layer_config']

    def get_loss_function(self) -> str:
        """Get the loss function name.

        :return: loss function name. Can only be SIGMOID_CROSS_ENTROPY (case insensitive)
        :rtype: str
        """
        if 'loss_fn' not in self.params:
            self.params['loss_fn'] = None
        return self.params['loss_fn']

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
        """Infer the embeddings for the specified vertices.

        :return: PgxFrame containing the embeddings for each vertex.
        :rtype: PgxFrame
        """
        self.check_is_fitted()
        vids = convert_python_to_java_vertex_list(graph, vertices)
        return PgxFrame(java_handler(self._model.inferEmbeddings, [graph._graph, vids]))

    def infer_and_get_explanation(
        self, graph: PgxGraph, vertex: Union[PgxVertex, int], num_clusters: int = 50
    ) -> GnnExplanation:
        """Perform inference on the specified vertex and generate an explanation that contains
        scores of how important each property and each vertex in the computation graph is for the
        embeddings position relative to embeddings of other vertices in the graph.

        :param graph: the graph
        :param vertex: the vertex
        :param num_clusters: the number of semantic vertex clusters expected in the graph,
            must be greater than 1
        :returns: explanation containing feature importance and vertex importance.
        """
        self.check_is_fitted()
        java_vertex = convert_python_to_java_vertex(graph, vertex)
        return GnnExplanation(
            java_handler(
                self._model.inferAndGetExplanation, [graph._graph, java_vertex, num_clusters]
            )
        )
