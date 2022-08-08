#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from pypgx._utils.error_handling import java_handler
from pypgx.api.mllib._graphwise_conv_layer_config import GraphWiseConvLayerConfig
from typing import List


class GraphWiseModelConfig:
    """Graphwise Model Configuration class"""

    _java_class = "oracle.pgx.config.mllib.GraphWiseModelConfig"

    def __init__(self, java_graphwise_model_config) -> None:
        self._config = java_graphwise_model_config
        self.shuffle = java_graphwise_model_config.isShuffle()
        self.input_feature_dim = java_graphwise_model_config.getInputFeatureDim()
        self.edge_input_feature_dim = java_graphwise_model_config.getEdgeInputFeatureDim()
        self.is_fitted = java_graphwise_model_config.isFitted()
        self.training_loss = java_graphwise_model_config.getTrainingLoss()
        self.batch_size = java_graphwise_model_config.getBatchSize()
        self.num_epochs = java_graphwise_model_config.getNumEpochs()
        self.learning_rate = java_graphwise_model_config.getLearningRate()
        self.embedding_dim = java_graphwise_model_config.getEmbeddingDim()
        self.seed = java_graphwise_model_config.getSeed()
        self.conv_layer_configs = self.get_conv_layer_configs()
        self.vertex_input_property_names = java_graphwise_model_config.getVertexInputPropertyNames()
        if self.vertex_input_property_names:
            self.vertex_input_property_names = self.vertex_input_property_names.toArray()
        self.edge_input_property_names = java_graphwise_model_config.getEdgeInputPropertyNames()
        if self.edge_input_property_names:
            self.edge_input_property_names = self.edge_input_property_names.toArray()
        self.standardize = java_graphwise_model_config.isStandardize()
        self.backend = java_graphwise_model_config.getBackend().name()

    def get_conv_layer_configs(self) -> List[GraphWiseConvLayerConfig]:
        """Return a list of conv layer configs"""
        java_conv_layer_configs = java_handler(self._config.getConvLayerConfigs, [])
        conv_layer_configs = []
        for config in java_conv_layer_configs:
            params = {
                "weight_init_scheme": config.getWeightInitScheme().name(),
                "activation_fn": config.getActivationFunction().name(),
                "num_sampled_neighbors": config.getNumSampledNeighbors(),
                "neighbor_weight_property_name": config.getNeighborWeightPropertyName(),
            }
            conv_layer_configs.append(GraphWiseConvLayerConfig(config, params))
        return conv_layer_configs

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size

        :param batch_size: batch size
        :type batch_size: int
        """
        java_handler(self._config.setBatchSize, [batch_size])
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs: int) -> None:
        """Set the number of epochs

        :param num_epochs: number of epochs
        :type num_epochs: int
        """
        java_handler(self._config.setNumEpochs, [num_epochs])
        self.num_epochs = num_epochs

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate

        :param learning_rate: initial learning rate
        :type learning_rate: int
        """
        java_handler(self._config.setLearningRate, [learning_rate])
        self.learning_rate = learning_rate

    def set_embedding_dim(self, embedding_dim: int) -> None:
        """Set the embedding dimension

        :param embedding_dim: embedding dimension
        :type embedding_dim: int
        """
        java_handler(self._config.setEmbeddingDim, [embedding_dim])
        self.embedding_dim = embedding_dim

    def set_seed(self, seed: int) -> None:
        """Set the seed

        :param seed: seed
        :type seed: int
        """
        java_handler(self._config.setSeed, [seed])
        self.seed = seed

    def set_fitted(self, fitted: bool) -> None:
        """Set the fitted flag

        :param fitted: fitted flag
        :type fitted: bool
        """
        java_handler(self._config.setFitted, [fitted])
        self.fitted = fitted

    def set_shuffle(self, shuffle: bool) -> None:
        """Set the shuffling flag

        :param shuffle: shuffling flag
        :type shuffle: bool
        """
        java_handler(self._config.setShuffle, [shuffle])
        self.shuffle = shuffle

    def set_training_loss(self, training_loss: float) -> None:
        """Set the training loss

        :param training_loss: training loss
        :type training_loss: float
        """
        java_handler(self._config.setTrainingLoss, [training_loss])
        self.training_loss = training_loss

    def set_input_feature_dim(self, input_feature_dim: int) -> None:
        """Set the input feature dimension

        :param input_feature_dim: input feature dimension
        :type input_feature_dim: int
        """
        java_handler(self._config.setInputFeatureDim, [input_feature_dim])
        self.input_feature_dim = input_feature_dim

    def set_edge_input_feature_dim(self, edge_input_feature_dim: int) -> None:
        """Set the edge input feature dimension

        :param edge_input_feature_dim: edge input feature dimension
        :type edge_input_feature_dim: int
        """
        java_handler(self._config.setEdgeInputFeatureDim, [edge_input_feature_dim])
        self.edge_input_feature_dim = edge_input_feature_dim

    def set_standarize(self, standardize: bool) -> None:
        """Set the standardize flag

        :param standardize: standardize flag
        :type standardize: bool
        """
        java_handler(self._config.setStandardize, [standardize])
        self.standardize = standardize
