#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import List

from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx.api.mllib._graphwise_conv_layer_config import GraphWiseConvLayerConfig
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import MODEL_NOT_FITTED
from pypgx.api.mllib._graphwise_model_config import GraphWiseModelConfig


class GraphWiseModel(PgxContextManager):
    """GraphWise model object.

    This is a base class for :class:`UnsupervisedGraphWiseModel` and
    :class:`SupervisedGraphWiseModel`.
    """

    def __init__(self, java_graphwise_model, params=None) -> None:
        if params is None:
            params = {}
        self._model = java_graphwise_model
        self.params = params
        self.update_is_fitted()

    def destroy(self) -> None:
        """Destroy this model object.

        :return: None
        """
        java_handler(self._model.destroy, [])

    def close(self) -> None:
        """Call :meth:`destroy`.

        :return: None
        """
        self.destroy()

    def update_is_fitted(self) -> None:
        """Determine whether the model is fitted.

        This updates the internal state.

        :return: None
        """
        # Determining whether the model has been fitted is relevant especially for
        # models that are being loaded from a file.
        self._is_fitted = self._model.isFitted()
        self.loss = self._model.getTrainingLoss()
        self.vertex_input_feature_dim = self._model.getInputFeatureDim()
        self.edge_input_feature_dim = self._model.getEdgeInputFeatureDim()

    def check_is_fitted(self) -> None:
        """Make sure the model is fitted.

        :return: None
        :raise: RuntimeError if the model is not fitted
        """
        self.update_is_fitted()
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)

    def get_num_epochs(self) -> int:
        """Get the number of epochs to train the model

        :return: number of epochs to train the model
        :rtype: int
        """
        if 'num_epochs' not in self.params:
            self.params['num_epochs'] = self._model.getNumEpochs()
        return self.params['num_epochs']

    def get_learning_rate(self) -> float:
        """Get the initial learning rate

        :return: initial learning rate
        :rtype: float
        """
        if 'learning_rate' not in self.params:
            self.params['learning_rate'] = self._model.getLearningRate()
        return self.params['learning_rate']

    def get_batch_size(self) -> int:
        """Get the batch size

        :return: batch size
        :rtype: int
        """
        if 'batch_size' not in self.params:
            self.params['batch_size'] = self._model.getBatchSize()
        return self.params['batch_size']

    def get_layer_size(self) -> int:
        """Get the dimension of the embeddings

        :return: embedding dimension
        :rtype: int
        """
        if 'layer_size' not in self.params:
            self.params['layer_size'] = self._model.getEmbeddingDim()
        return self.params['layer_size']

    def get_seed(self) -> int:
        """Get the random seed

        :return: random seed
        :rtype: int
        """
        if 'seed' not in self.params:
            self.params['seed'] = self._model.getSeed()
        return self.params['seed']

    def get_conv_layer_config(self) -> GraphWiseConvLayerConfig:
        """Get the configuration objects for the convolutional layers

        :return: configurations
        :rtype: GraphWiseConvLayerConfig
        """
        if 'conv_layer_config' not in self.params:
            java_conv_layer_configs = java_handler(self._model.getConvLayerConfigs, [])
            conv_layer_configs = []
            for config in java_conv_layer_configs:
                params = {
                    "weight_init_scheme": config.getWeightInitScheme().name(),
                    "activation_fn": config.getActivationFunction().name(),
                    "num_sampled_neighbors": config.getNumSampledNeighbors(),
                    "neighbor_weight_property_name": config.getNeighborWeightPropertyName(),
                }
                conv_layer_configs.append(GraphWiseConvLayerConfig(config, params))
            self.params['conv_layer_config'] = conv_layer_configs
        return self.params['conv_layer_config']

    def get_config(self) -> GraphWiseModelConfig:
        """Return the GraphWiseModelConfig object

        :return: the config
        :rtype: GraphWiseModelConfig
        """
        java_config = java_handler(self._model.getConfig, [])
        return GraphWiseModelConfig(java_config)

    def get_vertex_input_property_names(self) -> List[str]:
        """Get the vertices input feature names

        :return: vertices input feature names
        :rtype: list(str)
        """
        if 'vertex_input_property_names' not in self.params:
            self.params['vertex_input_property_names'] = None
        return self.params['vertex_input_property_names']

    def get_edge_input_property_names(self) -> List[str]:
        """Get the edges input feature names

        :return: edges input feature names
        :rtype: list(str)
        """
        if 'edge_input_property_names' not in self.params:
            self.params['edge_input_property_names'] = None
        return self.params['edge_input_property_names']

    def is_fitted(self) -> bool:
        """Check if the model is fitted

        :return: `True` if the model is fitted, `False` otherwise
        :rtype: bool
        """
        self.update_is_fitted()
        return self._is_fitted

    def get_training_loss(self) -> float:
        """Get the final training loss

        :return: training loss
        :rtype: float
        """
        self.check_is_fitted()
        return self.loss

    def get_vertex_input_feature_dim(self) -> int:
        """Get the input feature dimension, that is, the dimension of all the input vertex
        properties when concatenated

        :return: input feature dimension
        :rtype: int
        """
        self.check_is_fitted()
        return self.vertex_input_feature_dim

    def get_edge_input_feature_dim(self) -> int:
        """Get the edges input feature dimension, that is, the dimension of all the input edge
        properties when concatenated

        :return: edges input feature dimension
        :rtype: int
        """
        self.check_is_fitted()
        return self.edge_input_feature_dim

    def _get_col_names(self, vec):
        result_elements = vec.getPgqlResultElements()
        col_names = []
        for idx in range(result_elements.size()):
            col_names.append(result_elements.get(idx).getVarName())
        return col_names

    def __repr__(self) -> str:
        attributes = []
        self.update_is_fitted()
        attributes.append('fitted: %s' % self._is_fitted)
        if self._is_fitted:
            attributes.append('loss: %.5f' % self.loss)
        for param in self.params:
            if param != 'self':
                attributes.append('%s: %s' % (param, self.params[param]))
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attributes))

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._model.equals(other._model)
