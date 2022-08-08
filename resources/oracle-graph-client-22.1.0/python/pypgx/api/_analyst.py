#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import pypgx._utils.algorithms_metadata as alg_metadata
from jnius import autoclass
from pypgx.api._pgx_map import PgxMap
from pypgx.api._all_paths import AllPaths
from pypgx.api._matrix_factorization_model import MatrixFactorizationModel
from pypgx.api._partition import PgxPartition
from pypgx.api._pgx_collection import VertexSet, EdgeSet, VertexSequence, EdgeSequence
from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_path import PgxPath
from pypgx.api._property import EdgeProperty, VertexProperty
from pypgx.api._pgx_graph import BipartiteGraph, PgxGraph
from pypgx.api.filters import EdgeFilter, VertexFilter
from pypgx.api.mllib import (
    DeepWalkModel,
    Pg2vecModel,
    SupervisedGraphWiseModel,
    UnsupervisedGraphWiseModel,
    GraphWisePredictionLayerConfig,
    GraphWiseConvLayerConfig,
    GraphWiseDgiLayerConfig,
    PermutationCorruption,
    CorruptionFunction,
)
from pypgx.api.mllib._model_utils import ModelLoader
from pypgx.api.mllib._loss_function import _get_loss_function, LossFunction
from pypgx._utils.arguments_validator import validate_arguments
from pypgx._utils.error_handling import java_handler, java_caster, _cast_to_java
from pypgx._utils.error_messages import PROPERTY_NOT_FOUND
from pypgx._utils.error_messages import UNHASHABLE_TYPE
from pypgx._utils.pgx_types import (
    ACTIVATION_FUNCTIONS,
    WEIGHT_INIT_SCHEMES,
    UNSUPERVISED_LOSS_FUNCTIONS,
    DISCRIMINATOR_FUNCTIONS,
    READOUT_FUNCTIONS,
    BATCH_GENERATORS,
)
from pypgx._utils.pgx_types import java_types, property_types
from typing import Any, List, Mapping, Optional, Tuple, Union, Iterable, TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession


class Analyst:
    """The Analyst gives access to all built-in algorithms of PGX.

    Unlike some of the other classes inside this package, the Analyst is not stateless. It
    creates session-bound transient data to hold the result of algorithms and keeps track of them.
    """

    _java_class = 'oracle.pgx.api.Analyst'

    def __init__(self, session: "PgxSession", java_analyst) -> None:
        self._analyst = java_analyst
        self.session = session

    def __repr__(self) -> str:
        return "{}(session id: {})".format(self.__class__.__name__, self.session.id)

    def __str__(self) -> str:
        return repr(self)

    def close(self) -> None:
        """Destroy without waiting for completion."""
        java_handler(self._analyst.close, [])

    def destroy(self) -> None:
        """Destroy with waiting for completion."""
        java_handler(self._analyst.destroy, [])

    def pg2vec_builder(
        self,
        graphlet_id_property_name: str,
        vertex_property_names: List[str],
        min_word_frequency: int = 1,
        batch_size: int = 128,
        num_epochs: int = 5,
        layer_size: int = 200,
        learning_rate: float = 0.04,
        min_learning_rate: float = 0.0001,
        window_size: int = 4,
        walk_length: int = 8,
        walks_per_vertex: int = 5,
        graphlet_size_property_name: str = "graphletSize-Pg2vec",
        use_graphlet_size: bool = True,
        validation_fraction: float = 0.05,
        seed: Optional[int] = None,
    ) -> Pg2vecModel:
        """Build a pg2Vec model and return it.

        :param graphlet_id_property_name: Property name of the graphlet-id in the input graph
        :param vertex_property_names: Property names to consider for pg2vec model training
        :param min_word_frequency:  Minimum word frequency to consider before pruning
        :param batch_size:  Batch size for training the model
        :param num_epochs:  Number of epochs to train the model
        :param layer_size:  Number of dimensions for the output vectors
        :param learning_rate:  Initial learning rate
        :param min_learning_rate:  Minimum learning rate
        :param window_size:  Window size to consider while training the model
        :param walk_length:  Length of the walks
        :param walks_per_vertex:  Number of walks to consider per vertex
        :param graphlet_size_property_name: Property name for graphlet size
        :param use_graphlet_size:  Whether to use or not the graphlet size
        :param validation_fraction:  Fraction of training data on which to compute final loss
        :param seed:  Seed
        :returns: Built Pg2Vec Model
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.pg2vec_builder)

        properties = autoclass('java.util.ArrayList')()
        for p in vertex_property_names:
            if not isinstance(p, str):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=p))
            properties.add(p)

        builder = self._analyst.pg2vecModelBuilder()
        java_handler(builder.setGraphLetIdPropertyName, [graphlet_id_property_name])
        java_handler(builder.setVertexPropertyNames, [properties])
        java_handler(builder.setMinWordFrequency, [min_word_frequency])
        java_handler(builder.setBatchSize, [batch_size])
        java_handler(builder.setNumEpochs, [num_epochs])
        java_handler(builder.setLayerSize, [layer_size])
        java_handler(builder.setLearningRate, [learning_rate])
        java_handler(builder.setMinLearningRate, [min_learning_rate])
        java_handler(builder.setWindowSize, [window_size])
        java_handler(builder.setWalkLength, [walk_length])
        java_handler(builder.setWalksPerVertex, [walks_per_vertex])
        java_handler(builder.setUseGraphletSize, [use_graphlet_size])
        java_handler(builder.setGraphletSizePropertyName, [graphlet_size_property_name])
        java_handler(builder.setValidationFraction, [validation_fraction])
        if seed is not None:
            java_caster(builder.setSeed, (seed, 'long'))
        model = java_handler(builder.build, [])
        return Pg2vecModel(model)

    def load_pg2vec_model(self, path: str, key: Optional[str]) -> Pg2vecModel:
        """Load an encrypted pg2vec model.

        :param path: Path to model
        :param key: The decryption key, or None if no encryption was used
        :returns: Loaded model
        """
        model = java_handler(self._analyst.loadPg2vecModel, [path, key])
        return Pg2vecModel(model)

    def get_pg2vec_model_loader(self) -> ModelLoader:
        """Return a ModelLoader that can be used for loading a Pg2vecModel.

        :returns: ModelLoader
        """
        return ModelLoader(
            self,
            self._analyst.loadPg2vecModel,
            lambda x: Pg2vecModel(x),
            'oracle.pgx.api.mllib.Pg2vecModel',
        )

    def deepwalk_builder(
        self,
        min_word_frequency: int = 1,
        batch_size: int = 128,
        num_epochs: int = 2,
        layer_size: int = 200,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        window_size: int = 5,
        walk_length: int = 5,
        walks_per_vertex: int = 4,
        sample_rate: float = 0.00001,
        negative_sample: int = 10,
        validation_fraction: float = 0.05,
        seed: Optional[int] = None,
    ) -> DeepWalkModel:
        """Build a DeepWalk model and return it.

        :param min_word_frequency: Minimum word frequency to consider before pruning
        :param batch_size:  Batch size for training the model
        :param num_epochs:  Number of epochs to train the model
        :param layer_size:  Number of dimensions for the output vectors
        :param learning_rate:  Initial learning rate
        :param min_learning_rate:  Minimum learning rate
        :param window_size:  Window size to consider while training the model
        :param walk_length:  Length of the walks
        :param walks_per_vertex:  Number of walks to consider per vertex
        :param sample_rate:  Sample rate
        :param negative_sample:  Number of negative samples
        :param validation_fraction:  Fraction of training data on which to compute final loss
        :param seed:  Random seed for training the model
        :returns: Built DeepWalk model
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.deepwalk_builder)

        builder = self._analyst.deepWalkModelBuilder()
        java_handler(builder.setMinWordFrequency, [min_word_frequency])
        java_handler(builder.setBatchSize, [batch_size])
        java_handler(builder.setNumEpochs, [num_epochs])
        java_handler(builder.setLayerSize, [layer_size])
        java_handler(builder.setLearningRate, [learning_rate])
        java_handler(builder.setMinLearningRate, [min_learning_rate])
        java_handler(builder.setWindowSize, [window_size])
        java_handler(builder.setWalkLength, [walk_length])
        java_handler(builder.setWalksPerVertex, [walks_per_vertex])
        java_handler(builder.setSampleRate, [sample_rate])
        java_handler(builder.setNegativeSample, [negative_sample])
        java_handler(builder.setValidationFraction, [validation_fraction])
        if seed is not None:
            java_caster(builder.setSeed, (seed, 'long'))
        model = java_handler(builder.build, [])
        return DeepWalkModel(model)

    def load_deepwalk_model(self, path: str, key: Optional[str]) -> DeepWalkModel:
        """Load an encrypted DeepWalk model.

        :param path: Path to model
        :param key: The decryption key, or None if no encryption was used
        :returns: Loaded model
        """
        model = java_handler(self._analyst.loadDeepWalkModel, [path, key])
        return DeepWalkModel(model)

    def get_deepwalk_model_loader(self) -> ModelLoader:
        """Return a ModelLoader that can be used for loading a DeepWalkModel.

        :returns: ModelLoader
        """
        return ModelLoader(
            self,
            self._analyst.loadDeepWalkModel,
            lambda x: DeepWalkModel(x),
            'oracle.pgx.api.mllib.DeepWalkModel',
        )

    def supervised_graphwise_builder(
        self,
        vertex_target_property_name: str,
        vertex_input_property_names: List[str] = [],
        edge_input_property_names: List[str] = [],
        loss_fn: Union[LossFunction, str] = 'SOFTMAX_CROSS_ENTROPY',
        batch_gen: str = 'STANDARD',
        batch_gen_params: List[Any] = [],
        pred_layer_config: Optional[Iterable[GraphWisePredictionLayerConfig]] = None,
        conv_layer_config: Optional[Iterable[GraphWiseConvLayerConfig]] = None,
        batch_size: int = 128,
        num_epochs: int = 3,
        learning_rate: float = 0.01,
        layer_size: int = 128,
        class_weights: Optional[Union[Mapping[str, float], Mapping[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> SupervisedGraphWiseModel:
        """Build a SupervisedGraphWise model and return it.

        :param vertex_target_property_name: Target property name
        :param vertex_input_property_names: Vertices Input feature names
        :param edge_input_property_names: Edges Input feature names
        :param loss_fn: Loss function. Supported: String ('SOFTMAX_CROSS_ENTROPY',
            'SIGMOID_CROSS_ENTROPY') or LossFunction object
        :param batch_gen: Batch generator. Supported: 'STANDARD', 'STRATIFIED_OVERSAMPLING'
        :param batch_gen_params: List of parameters passed to the batch generator
        :param pred_layer_config: Prediction layer configuration as list of PredLayerConfig,
            or default if None
        :param conv_layer_config: Conv layer configuration as list of ConvLayerConfig,
            or default if None
        :param batch_size:  Batch size for training the model
        :param num_epochs:  Number of epochs to train the model
        :param learning_rate: Learning rate
        :param layer_size: Number of dimensions for the output vectors
        :param class_weights: Class weights to be used in the loss function.
            The loss for the corresponding class will be multiplied by the factor given in this map.
            If null, uniform class weights will be used.
        :param seed: Seed
        :returns: Built SupervisedGraphWise model
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.supervised_graphwise_builder)

        # create default config, useful when printing the model to see the config
        if pred_layer_config is None:
            pred_layer_config = [self.graphwise_pred_layer_config()]
            arguments['pred_layer_config'] = pred_layer_config
        if conv_layer_config is None:
            conv_layer_config = [self.graphwise_conv_layer_config()]
            arguments['conv_layer_config'] = conv_layer_config

        # convert vertices input properties to Java ArrayList<String>
        vertex_input_properties = autoclass('java.util.ArrayList')()
        for p in vertex_input_property_names:
            if not isinstance(p, str):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=p))
            vertex_input_properties.add(p)
        # convert edges input properties to Java ArrayList<String>
        edge_input_properties = autoclass('java.util.ArrayList')()
        for p in edge_input_property_names:
            if not isinstance(p, str):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=p))
            edge_input_properties.add(p)

        builder = self._analyst.supervisedGraphWiseModelBuilder()

        # create a list of the Java objects of the pred layer configs
        pred_layer_configs = []
        for layer_config in pred_layer_config:
            if not isinstance(layer_config, GraphWisePredictionLayerConfig):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=layer_config))
            pred_layer_configs.append(layer_config._config)
        java_handler(builder.setPredictionLayerConfigs, pred_layer_configs)

        # create a list of the Java objects of the conv layer configs
        conv_layer_configs = []
        for layer_config in conv_layer_config:
            if not isinstance(layer_config, GraphWiseConvLayerConfig):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=layer_config))
            conv_layer_configs.append(layer_config._config)
        java_handler(builder.setConvLayerConfigs, conv_layer_configs)

        # make the loss fn uppercase, so that we perform case insensitive match
        if isinstance(loss_fn, str):
            loss_fn = _get_loss_function(loss_fn)
        loss_fn_java_obj = autoclass(loss_fn._java_class)(*loss_fn._java_arg_list)
        java_handler(builder.setLossFunction, [loss_fn_java_obj])

        batch_gen = batch_gen.upper()
        if batch_gen not in BATCH_GENERATORS.keys():
            raise ValueError(
                'Batch generator (%s) must be of the following types: %s'
                % (batch_gen, ', '.join(BATCH_GENERATORS.keys()))
            )
        java_handler(builder.setBatchGenerator, [BATCH_GENERATORS[batch_gen](*batch_gen_params)])

        # set the remaining parameters
        java_handler(builder.setBatchSize, [batch_size])
        java_handler(builder.setNumEpochs, [num_epochs])
        java_handler(builder.setLearningRate, [learning_rate])
        java_handler(builder.setEmbeddingDim, [layer_size])
        java_handler(builder.setVertexInputPropertyNames, [vertex_input_properties])
        if edge_input_property_names:
            java_handler(builder.setEdgeInputPropertyNames, [edge_input_properties])
        java_handler(builder.setVertexTargetPropertyName, [vertex_target_property_name])

        # convert the class weights to Map<?, Float> where the type of the key depends
        # on the type on the Python side
        if class_weights is not None:
            types = set()
            for _class in class_weights:
                types.add(type(_class))
            if len(types) > 1:
                raise ValueError('Keys in class weights have different types')

            class_type = list(types)[0]
            type_to_class = {
                type(1): java_types['integer'],
                type(True): java_types['boolean'],
                type(1.0): java_types['float'],
                type(""): java_types['string'],
            }
            if class_type not in type_to_class:
                raise ValueError(
                    'Class weight (%s) not supported. Only %s are supported'
                    % (class_type, ', '.join(map(str, type_to_class.keys())))
                )

            java_class_weights = autoclass('java.util.HashMap')()
            for _class, _weight in class_weights.items():
                java_class = _cast_to_java(_class, type_to_class[class_type])
                java_weight = _cast_to_java(_weight, java_types['float'])
                java_handler(java_class_weights.put, [java_class, java_weight])
            java_handler(builder.setClassWeights, [java_class_weights])

        if seed is not None:
            java_caster(builder.setSeed, (seed, 'integer'))
        model = java_handler(builder.build, [])
        return SupervisedGraphWiseModel(model, arguments)

    def load_supervised_graphwise_model(
        self, path: str, key: Optional[str]
    ) -> SupervisedGraphWiseModel:
        """Load an encrypted SupervisedGraphWise model.

        :param path: Path to model
        :param key: The decryption key, or None if no encryption was used
        :returns: Loaded model
        """
        model = java_handler(self._analyst.loadSupervisedGraphWiseModel, [path, key])
        return SupervisedGraphWiseModel(model)

    def get_supervised_graphwise_model_loader(self) -> ModelLoader:
        """Return a ModelLoader that can be used for loading a SupervisedGraphWiseModel.

        :returns: ModelLoader
        """
        return ModelLoader(
            self,
            self._analyst.loadSupervisedGraphWiseModel,
            lambda x: SupervisedGraphWiseModel(x),
            'oracle.pgx.api.mllib.SupervisedGraphWiseModel',
        )

    def unsupervised_graphwise_builder(
        self,
        vertex_input_property_names: List[str] = [],
        edge_input_property_names: List[str] = [],
        loss_fn: str = 'SIGMOID_CROSS_ENTROPY',
        conv_layer_config: Optional[Iterable[GraphWiseConvLayerConfig]] = None,
        batch_size: int = 128,
        num_epochs: int = 3,
        learning_rate: float = 0.001,
        layer_size: int = 128,
        seed: Optional[int] = None,
        dgi_layer_config: Optional[GraphWiseDgiLayerConfig] = None,
    ) -> UnsupervisedGraphWiseModel:
        """Build a UnsupervisedGraphWise model and return it.

        :param vertex_input_property_names: Vertices Input feature names
        :param edge_input_property_names: Edges Input feature names
        :param loss_fn: Loss function. Supported: SIGMOID_CROSS_ENTROPY
        :param conv_layer_config: Conv layer configuration as list of ConvLayerConfig,
            or default if None
        :param batch_size:  Batch size for training the model
        :param num_epochs:  Number of epochs to train the model
        :param learning_rate: Learning rate
        :param layer_size: Number of dimensions for the output vectors
        :param seed: Seed
        :param dgi_layer_config: Dgi layer configuration as DgiLayerConfig object,
            or default if None
        :returns: Built UnsupervisedGraphWise model
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.unsupervised_graphwise_builder)

        # create default config, useful when printing the model to see the config

        if dgi_layer_config is None:
            dgi_layer_config = self.graphwise_dgi_layer_config()
            arguments['dgi_layer_config'] = dgi_layer_config

        if conv_layer_config is None:
            conv_layer_config = [self.graphwise_conv_layer_config()]
            arguments['conv_layer_config'] = conv_layer_config

        # convert vertices input properties to Java ArrayList<String>
        input_properties = autoclass('java.util.ArrayList')()
        for p in vertex_input_property_names:
            if not isinstance(p, str):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=p))
            input_properties.add(p)

        # convert edges input properties to Java ArrayList<String>
        edge_input_properties = autoclass('java.util.ArrayList')()
        for p in edge_input_property_names:
            if not isinstance(p, str):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=p))
            edge_input_properties.add(p)

        builder = self._analyst.unsupervisedGraphWiseModelBuilder()

        # Set the dgi layer config
        if not isinstance(dgi_layer_config, GraphWiseDgiLayerConfig):
            raise TypeError(PROPERTY_NOT_FOUND.format(prop=dgi_layer_config))
        java_handler(builder.setDgiLayerConfig, [dgi_layer_config._config])

        # create a list of the Java objects of the conv layer configs
        conv_layer_configs = []
        for layer_config in conv_layer_config:
            if not isinstance(layer_config, GraphWiseConvLayerConfig):
                raise TypeError(PROPERTY_NOT_FOUND.format(prop=layer_config))
            conv_layer_configs.append(layer_config._config)
        java_handler(builder.setConvLayerConfigs, conv_layer_configs)

        # make the loss fn uppercase, so that we perform case insensitive match
        loss_fn = loss_fn.upper()
        if loss_fn not in UNSUPERVISED_LOSS_FUNCTIONS.keys():
            raise ValueError(
                'Loss function (%s) must be of the following types: %s'
                % (loss_fn, ', '.join(UNSUPERVISED_LOSS_FUNCTIONS.keys()))
            )
        java_handler(builder.setLossFunction, [UNSUPERVISED_LOSS_FUNCTIONS[loss_fn]])

        # set the remaining parameters
        java_handler(builder.setBatchSize, [batch_size])
        java_handler(builder.setNumEpochs, [num_epochs])
        java_handler(builder.setLearningRate, [learning_rate])
        java_handler(builder.setEmbeddingDim, [layer_size])
        java_handler(builder.setVertexInputPropertyNames, [input_properties])
        if edge_input_property_names:
            java_handler(builder.setEdgeInputPropertyNames, [edge_input_properties])

        if seed is not None:
            java_caster(builder.setSeed, (seed, 'integer'))
        model = java_handler(builder.build, [])
        return UnsupervisedGraphWiseModel(model, arguments)

    def load_unsupervised_graphwise_model(self, path: str, key: str) -> UnsupervisedGraphWiseModel:
        """Load an encrypted UnsupervisedGraphWise model.

        :param path: Path to model
        :param key: The decryption key, or null if no encryption was used
        :returns: Loaded model
        """
        model = java_handler(self._analyst.loadUnsupervisedGraphWiseModel, [path, key])
        return UnsupervisedGraphWiseModel(model)

    def get_unsupervised_graphwise_model_loader(self) -> ModelLoader:
        """Return a ModelLoader that can be used for loading a UnsupervisedGraphWiseModel.

        :returns: ModelLoader
        """
        return ModelLoader(
            self,
            self._analyst.loadUnsupervisedGraphWiseModel,
            lambda x: UnsupervisedGraphWiseModel(x),
            'oracle.pgx.api.mllib.UnsupervisedGraphWiseModel',
        )

    def graphwise_pred_layer_config(
        self,
        hidden_dim: Optional[int] = None,
        activation_fn: str = 'ReLU',
        weight_init_scheme: str = 'XAVIER_UNIFORM',
    ) -> GraphWisePredictionLayerConfig:
        """Build a GraphWise prediction layer configuration and return it.

        :param hidden_dim: Hidden dimension. If this is the last layer, this setting
            will be ignored and replaced by the number of classes.
        :param activation_fn: Activation function.
            Supported functions: RELU, LEAKY_RELU, TANH, LINEAR.
            If this is the last layer, this setting will be ignored and replaced by
            the activation function of the loss function, e.g softmax or sigmoid.
        :param weight_init_scheme: Initialization scheme for the weights in the layer.
            Supportes schemes: XAVIER, XAVIER_UNIFORM, ONES, ZEROS.
            Note that biases are always initialized with zeros.
        :returns: Built GraphWisePredictionLayerConfig
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.graphwise_pred_layer_config)

        builder = self._analyst.graphWisePredictionLayerConfigBuilder()

        if hidden_dim is not None:
            java_handler(builder.setHiddenDimension, [hidden_dim])

        # make the activation fn uppercase, so that we perform case insensitive match
        activation_fn = activation_fn.upper()
        if activation_fn not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError(
                'Activation function (%s) must be of the following types: %s'
                % (activation_fn, ', '.join(ACTIVATION_FUNCTIONS.keys()))
            )
        java_handler(builder.setActivationFunction, [ACTIVATION_FUNCTIONS[activation_fn]])

        # make the weight init scheme uppercase, so that we perform case insensitive match
        weight_init_scheme = weight_init_scheme.upper()
        if weight_init_scheme not in WEIGHT_INIT_SCHEMES.keys():
            raise ValueError(
                'Weight init scheme (%s) must be of the following types: %s'
                % (weight_init_scheme, ', '.join(WEIGHT_INIT_SCHEMES.keys()))
            )
        java_handler(builder.setWeightInitScheme, [WEIGHT_INIT_SCHEMES[weight_init_scheme]])

        config = java_handler(builder.build, [])
        return GraphWisePredictionLayerConfig(config, arguments)

    def graphwise_conv_layer_config(
        self,
        num_sampled_neighbors: int = 10,
        neighbor_weight_property_name: Optional[str] = None,
        activation_fn: str = 'ReLU',
        weight_init_scheme: str = 'XAVIER_UNIFORM',
    ) -> GraphWiseConvLayerConfig:
        """Build a GraphWise conv layer configuration and return it.

        :param num_sampled_neighbors: Number of neighbors to sample
        :param neighbor_weight_property_name: Neighbor weight property name.
        :param activation_fn: Activation function.
            Supported functions: RELU, LEAKY_RELU, TANH, LINEAR.
            If this is the last layer, this setting will be ignored and replaced by
            the activation function of the loss function, e.g softmax or sigmoid.
        :param weight_init_scheme: Initialization scheme for the weights in the layer.
            Supported schemes: XAVIER, XAVIER_UNIFORM, ONES, ZEROS.
            Note that biases are always initialized with zeros.
        :returns: Built GraphWiseConvLayerConfig
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.graphwise_conv_layer_config)

        builder = self._analyst.graphWiseConvLayerConfigBuilder()

        java_handler(builder.setNumSampledNeighbors, [num_sampled_neighbors])
        if neighbor_weight_property_name is not None:
            java_handler(builder.setWeightedAggregationProperty, [neighbor_weight_property_name])

        # make the activation function uppercase, so that we perform case insensitive match
        activation_fn = activation_fn.upper()
        if activation_fn not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError(
                'Activation function (%s) must be of the following types: %s'
                % (activation_fn, ', '.join(ACTIVATION_FUNCTIONS.keys()))
            )
        java_handler(builder.setActivationFunction, [ACTIVATION_FUNCTIONS[activation_fn]])

        # make the weight init scheme uppercase, so that we perform case insensitive match
        weight_init_scheme = weight_init_scheme.upper()
        if weight_init_scheme not in WEIGHT_INIT_SCHEMES.keys():
            raise ValueError(
                'Weight init scheme (%s) must be of the following types: %s'
                % (weight_init_scheme, ', '.join(WEIGHT_INIT_SCHEMES.keys()))
            )
        java_handler(builder.setWeightInitScheme, [WEIGHT_INIT_SCHEMES[weight_init_scheme]])

        config = java_handler(builder.build, [])
        return GraphWiseConvLayerConfig(config, arguments)

    def graphwise_dgi_layer_config(
        self,
        corruption_function: Optional[CorruptionFunction] = None,
        readout_function: str = "MEAN",
        discriminator: str = "BILINEAR",
    ) -> GraphWiseDgiLayerConfig:
        """Build a GraphWise DGI layer configuration and return it.

        :param corruption_function(CorruptionFunction): Corruption Function to use
        :param readout_function(str): Neighbor weight property name.
            Supported functions: MEAN
        :param discriminator(str): discriminator function.
            Supported functions: BILINEAR
        :returns: GraphWiseDgiLayerConfig object
        """
        arguments = locals()

        if corruption_function is None:
            java_permutation_corruption = autoclass(
                "oracle.pgx.config.mllib.corruption.PermutationCorruption"
            )()
            corruption_function = PermutationCorruption(java_permutation_corruption)
            arguments['corruption_function'] = corruption_function

        validate_arguments(arguments, alg_metadata.graphwise_dgi_layer_config)

        builder = self._analyst.graphWiseDgiLayerConfigBuilder()

        java_handler(builder.setCorruptionFunction, [corruption_function._corruption_function])

        # make the readout function uppercase, so that we perform case insensitive match
        readout_function = readout_function.upper()
        if readout_function not in READOUT_FUNCTIONS.keys():
            raise ValueError(
                'Readout function (%s) must be of the following types: %s'
                % (readout_function, ', '.join(READOUT_FUNCTIONS.keys()))
            )
        java_handler(builder.setReadoutFunction, [READOUT_FUNCTIONS[readout_function]])

        # make the discriminator uppercase, so that we perform case insensitive match
        discriminator = discriminator.upper()
        if discriminator not in DISCRIMINATOR_FUNCTIONS.keys():
            raise ValueError(
                'Discriminator (%s) must be of the following types: %s'
                % (discriminator, ', '.join(DISCRIMINATOR_FUNCTIONS.keys()))
            )
        java_handler(builder.setDiscriminator, [DISCRIMINATOR_FUNCTIONS[discriminator]])

        config = java_handler(builder.build, [])
        return GraphWiseDgiLayerConfig(config, arguments)

    def pagerank(
        self,
        graph: PgxGraph,
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        norm: bool = False,
        rank: Union[VertexProperty, str] = "pagerank",
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of
            the error values of all vertices becomes smaller than this value.
        :param damping: Damping factor
        :param max_iter: Maximum number of iterations that will be performed
        :param norm: Determine whether the algorithm will take into account dangling vertices
            for the ranking scores.
        :param rank: Vertex property holding the PageRank value for each vertex, or name for a new
            property
        :returns: Vertex property holding the PageRank value for each vertex
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.pagerank)

        if isinstance(rank, str):
            rank = graph.create_vertex_property("double", rank)

        java_handler(
            self._analyst.pagerank, [graph._graph, tol, damping, max_iter, norm, rank._prop]
        )
        return rank

    def pagerank_approximate(
        self,
        graph: PgxGraph,
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        rank: Union[VertexProperty, str] = "approx_pagerank",
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the
            error values of all vertices becomes smaller than this value.
        :param damping: Damping factor
        :param max_iter: Maximum number of iterations that will be performed
        :param rank: Vertex property holding the PageRank value for each vertex
        :returns: Vertex property holding the PageRank value for each vertex
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.pagerank_approximate)

        if isinstance(rank, str):
            rank = graph.create_vertex_property("double", rank)

        java_handler(
            self._analyst.pagerankApproximate, [graph._graph, tol, damping, max_iter, rank._prop]
        )
        return rank

    def weighted_pagerank(
        self,
        graph: PgxGraph,
        weight: EdgeProperty,
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        norm: bool = False,
        rank: Union[VertexProperty, str] = "weighted_pagerank",
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param weight: Edge property holding the weight of each edge in the graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the
            error values of all vertices becomes smaller than this value.
        :param damping: Damping factor
        :param max_iter: Maximum number of iterations that will be performed
        :param rank: Vertex property holding the PageRank value for each vertex
        :returns: Vertex property holding the computed the peageRank value
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.weighted_pagerank)

        if isinstance(rank, str):
            rank = graph.create_vertex_property("double", rank)

        java_handler(
            self._analyst.weightedPagerank,
            [graph._graph, tol, damping, max_iter, norm, weight._prop, rank._prop],
        )
        return rank

    def personalized_pagerank(
        self,
        graph: PgxGraph,
        v: Union[VertexSet, PgxVertex],
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        norm: bool = False,
        rank: Union[VertexProperty, str] = "personalized_pagerank",
    ) -> VertexProperty:
        """Personalized PageRank for a vertex of interest.

        Compares and spots out important vertices in a graph.

        :param graph: Input graph
        :param v: The chosen vertex from the graph for personalization
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of
            the error values of all vertices becomes smaller than this value.
        :param damping: Damping factor
        :param max_iter: Maximum number of iterations that will be performed
        :param norm: Boolean flag to determine whether
            the algorithm will take into account dangling vertices for the ranking scores.
        :param rank: Vertex property holding the PageRank value for each vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.personalized_pagerank)

        if isinstance(rank, str):
            rank = graph.create_vertex_property("double", rank)

        if isinstance(v, PgxVertex):
            java_handler(
                self._analyst.personalizedPagerank,
                [graph._graph, v._vertex, tol, damping, max_iter, norm, rank._prop],
            )
        if isinstance(v, VertexSet):
            java_handler(
                self._analyst.personalizedPagerank,
                [graph._graph, v._collection, tol, damping, max_iter, norm, rank._prop],
            )
        return rank

    def personalized_weighted_pagerank(
        self,
        graph: PgxGraph,
        v: Union[VertexSet, PgxVertex],
        weight: EdgeProperty,
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        norm: bool = False,
        rank: Union[VertexProperty, str] = "personalized_weighted_pagerank",
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param v: The chosen vertex from the graph for personalization
        :param weight: Edge property holding the weight of each edge in the graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of
            the error values of all vertices becomes smaller than this value.
        :param damping: Damping factor
        :param max_iter: Maximum number of iterations that will be performed
        :param norm: Boolean flag to determine whether the algorithm will take into account
            dangling vertices for the ranking scores
        :param rank: Vertex property holding the PageRank value for each vertex
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.personalized_weighted_pagerank)

        if isinstance(rank, str):
            rank = graph.create_vertex_property("double", rank)

        if isinstance(v, PgxVertex):
            java_handler(
                self._analyst.personalizedWeightedPagerank,
                [graph._graph, v._vertex, tol, damping, max_iter, norm, weight._prop, rank._prop],
            )
        if isinstance(v, VertexSet):
            java_handler(
                self._analyst.personalizedWeightedPagerank,
                [
                    graph._graph,
                    v._collection,
                    tol,
                    damping,
                    max_iter,
                    norm,
                    weight._prop,
                    rank._prop,
                ],
            )
        return rank

    def vertex_betweenness_centrality(
        self, graph: PgxGraph, bc: Union[VertexProperty, str] = "betweenness"
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param bc: Vertex property holding the betweenness centrality value for each vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.vertex_betweenness_centrality)

        if isinstance(bc, str):
            bc = graph.create_vertex_property("double", bc)

        java_handler(self._analyst.vertexBetweennessCentrality, [graph._graph, bc._prop])
        return bc

    def approximate_vertex_betweenness_centrality(
        self,
        graph: PgxGraph,
        seeds: Union[VertexSet, int],
        bc: Union[VertexProperty, str] = "approx_betweenness",
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param seeds: The (unique) chosen nodes to be used to compute the approximated betweenness
            centrality coefficients
        :param bc: Vertex property holding the betweenness centrality value for each vertex
        :returns: Vertex property holding the computed scores
        """
        args = locals()
        validate_arguments(args, alg_metadata.approximate_vertex_betweenness_centrality)

        if isinstance(bc, str):
            bc = graph.create_vertex_property("double", bc)

        if isinstance(seeds, VertexSet):
            arguments = [graph._graph, bc._prop]
            for s in seeds:
                arguments.append(s._vertex)
            java_handler(self._analyst.approximateVertexBetweennessCentralityFromSeeds, arguments)
        elif isinstance(seeds, int):
            java_handler(
                self._analyst.approximateVertexBetweennessCentrality,
                [graph._graph, seeds, bc._prop],
            )
        return bc

    def closeness_centrality(
        self, graph: PgxGraph, cc: Union[VertexProperty, str] = "closeness"
    ) -> VertexProperty:
        """
        :param graph: Input graph
        :param cc: Vertex property holding the closeness centrality
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.closeness_centrality)

        if isinstance(cc, str):
            cc = graph.create_vertex_property("double", cc)

        java_handler(self._analyst.closenessCentralityUnitLength, [graph._graph, cc._prop])
        return cc

    def weighted_closeness_centrality(
        self,
        graph: PgxGraph,
        weight: EdgeProperty,
        cc: Union[VertexProperty, str] = "weighted_closeness",
    ) -> VertexProperty:
        """Measure the centrality of the vertices based on weighted distances, allowing to find
        well-connected vertices.

        :param graph: Input graph
        :param weight: Edge property holding the weight of each edge in the graph
        :param cc: (Out argument) vertex property holding the closeness centrality value for each
            vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.weighted_closeness_centrality)

        if isinstance(cc, str):
            cc = graph.create_vertex_property("double", cc)

        java_handler(
            self._analyst.closenessCentralityDoubleLength, [graph._graph, weight._prop, cc._prop]
        )
        return cc

    def hits(
        self,
        graph: PgxGraph,
        max_iter: int = 100,
        auth: Union[VertexProperty, str] = "authorities",
        hubs: Union[VertexProperty, str] = "hubs",
    ) -> Tuple[VertexProperty, VertexProperty]:
        """Hyperlink-Induced Topic Search (HITS) assigns ranking scores to the vertices,
        aimed to assess the quality of information and references in linked structures.

        :param graph: Input graph
        :param max_iter: Number of iterations that will be performed
        :param auth: Vertex property holding the authority score for each vertex
        :param hubs: Vertex property holding the hub score for each vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.hits)

        if auth is None or isinstance(auth, str):
            auth = graph.create_vertex_property("double", auth)

        if hubs is None or isinstance(hubs, str):
            hubs = graph.create_vertex_property("double", hubs)

        java_handler(self._analyst.hits, [graph._graph, max_iter, auth._prop, hubs._prop])
        return (auth, hubs)

    def eigenvector_centrality(
        self,
        graph: PgxGraph,
        tol: float = 0.001,
        max_iter: int = 100,
        l2_norm: bool = False,
        in_edges: bool = False,
        ec: Union[VertexProperty, str] = "eigenvector",
    ) -> VertexProperty:
        """Eigenvector centrality gets the centrality of the vertices in an intrincated way using
        neighbors, allowing to find well-connected vertices.

        :param graph: Input graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the
            error values of all vertices becomes smaller than this value.
        :param max_iter: Maximum iteration number
        :param l2_norm: Boolean flag to determine whether the algorithm will use the l2 norm
            (Euclidean norm) or the l1 norm (absolute value) to normalize the centrality scores
        :param in_edges: Boolean flag to determine whether the algorithm will use the incoming
            or the outgoing edges in the graph for the computations
        :param ec: Vertex property holding the resulting score for each vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.eigenvector_centrality)

        if isinstance(ec, str):
            ec = graph.create_vertex_property("double", ec)

        java_handler(
            self._analyst.eigenvectorCentrality,
            [graph._graph, max_iter, tol, l2_norm, in_edges, ec._prop],
        )
        return ec

    def out_degree_centrality(
        self, graph: PgxGraph, dc: Union[VertexProperty, str] = "out_degree"
    ) -> VertexProperty:
        """Measure the out-degree centrality of the vertices based on its degree.

        This lets you see how a vertex influences its neighborhood.

        :param graph: Input graph
        :param dc: Vertex property holding the degree centrality value for each vertex in the graph.
            Can be a string or a VertexProperty object.
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.out_degree_centrality)

        if isinstance(dc, str):
            dc = graph.create_vertex_property("integer", dc)

        java_handler(self._analyst.outDegreeCentrality, [graph._graph, dc._prop])
        return dc

    def in_degree_centrality(
        self, graph: PgxGraph, dc: Union[VertexProperty, str] = "in_degree"
    ) -> VertexProperty:
        """Measure the in-degree centrality of the vertices based on its degree.

        This lets you see how a vertex influences its neighborhood.

        :param graph: Input graph
        :param dc: Vertex property holding the degree centrality value for each vertex in the graph.
            Can be a string or a VertexProperty object.
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.in_degree_centrality)

        if isinstance(dc, str):
            dc = graph.create_vertex_property("integer", dc)

        java_handler(self._analyst.inDegreeCentrality, [graph._graph, dc._prop])
        return dc

    def degree_centrality(
        self, graph: PgxGraph, dc: Union[VertexProperty, str] = "degree"
    ) -> VertexProperty:
        """Measure the centrality of the vertices based on its degree.

        This lets you see how a vertex influences its neighborhood.

        :param graph: Input graph
        :param dc: Vertex property holding the degree centrality value for each vertex in the graph.
            Can be a string or a VertexProperty object.
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.degree_centrality)

        if isinstance(dc, str):
            dc = graph.create_vertex_property("integer", dc)

        java_handler(self._analyst.degreeCentrality, [graph._graph, dc._prop])
        return dc

    def adamic_adar_counting(
        self, graph: PgxGraph, aa: Union[EdgeProperty, str] = "adamic_adar"
    ) -> EdgeProperty:
        """Adamic-adar counting compares the amount of neighbors shared between vertices,
        this measure can be used with communities.

        :param graph: Input graph
        :param aa: Edge property holding the Adamic-Adar index for each edge in the graph.
            Can be a string or an EdgeProperty object.
        :returns: Edge property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.adamic_adar_counting)

        if isinstance(aa, str):
            aa = graph.create_edge_property("double", aa)

        java_handler(self._analyst.adamicAdarCounting, [graph._graph, aa._prop])
        return aa

    def communities_label_propagation(
        self,
        graph: PgxGraph,
        max_iter: int = 100,
        label: Union[VertexProperty, str] = "label_propagation",
    ) -> PgxPartition:
        """Label propagation can find communities in a graph relatively fast.

        :param graph: Input graph
        :param max_iter: Maximum number of iterations that will be performed
        :param label: Vertex property holding the degree centrality value for each vertex in the
            graph. Can be a string or a VertexProperty object
        :returns: Partition holding the node collections corresponding to the communities found
            by the algorithm
        :param label:   Vertex property holding the degree centrality value for each vertex in the
                        graph. Can be a string or a VertexProperty object.
        :returns:   Partition holding the node collections corresponding to the communities found
                    by the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.communities_label_propagation)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(
            self._analyst.communitiesLabelPropagation, [graph._graph, max_iter, label._prop]
        )
        return PgxPartition(graph, java_partition, label)

    def communities_conductance_minimization(
        self,
        graph: PgxGraph,
        max_iter: int = 100,
        label: Union[VertexProperty, str] = "conductance_minimization",
    ) -> PgxPartition:
        """Soman and Narang can find communities in a graph taking weighted edges into account.

        :param graph: Input graph
        :param max_iter: Maximum number of iterations that will be performed
        :param label: Vertex property holding the degree centrality value for each vertex in the
            graph. Can be a string or a VertexProperty object.
        :returns: Partition holding the node collections corresponding to the communities found
            by the algorithm
        :param label:   Vertex property holding the degree centrality value for each vertex in the
                        graph. Can be a string or a VertexProperty object.
        :returns:   Partition holding the node collections corresponding to the communities found
                    by the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.communities_conductance_minimization)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(
            self._analyst.communitiesConductanceMinimization, [graph._graph, max_iter, label._prop]
        )
        return PgxPartition(graph, java_partition, label)

    def communities_infomap(
        self,
        graph: PgxGraph,
        rank: VertexProperty,
        weight: EdgeProperty,
        tau: float = 0.15,
        tol: float = 0.0001,
        max_iter: int = 100,
        label: Union[VertexProperty, str] = "infomap",
    ) -> PgxPartition:
        """Infomap can find high quality communities in a graph.

        :param graph: Input graph
        :param rank: Vertex property holding the normalized PageRank value for each vertex
        :param weight: Ridge property holding the weight of each edge in the graph
        :param tau: Damping factor
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the error
            values of all vertices becomes smaller than this value.
        :param max_iter: Maximum iteration number
        :param label: Vertex property holding the degree centrality value for each vertex in the
            graph. Can be a string or a VertexProperty object.
        :returns: Partition holding the node collections corresponding to the communities found
            by the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.communities_infomap)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(
            self._analyst.communitiesInfomap,
            [graph._graph, rank._prop, weight._prop, tau, tol, max_iter, label._prop],
        )
        return PgxPartition(graph, java_partition, label)

    def louvain(
        self,
        graph: PgxGraph,
        weight: EdgeProperty,
        max_iter: int = 100,
        nbr_pass: int = 1,
        tol: float = 0.0001,
        community: Union[VertexProperty, str] = "community",
    ) -> PgxPartition:
        """Louvain to detect communities in a graph

        :param graph: Input graph.
        :param weight: Weights of the edges of the graph.
        :param max_iter: Maximum number of iterations that will be performed during each pass.
        :param nbr_pass: Number of passes that will be performed.
        :param tol: maximum tolerated error value, the algorithm will stop once the graph's
            total modularity gain becomes smaller than this value.
        :param community: Vertex property holding the community ID assigned to each vertex
        :returns: Community IDs vertex property
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.louvain)

        if isinstance(community, str):
            community = graph.create_vertex_property("long", community)

        java_partition = java_handler(
            self._analyst.louvain,
            [graph._graph, weight._prop, max_iter, nbr_pass, tol, community._prop],
        )
        return PgxPartition(graph, java_partition, community)

    def conductance(self, graph: PgxGraph, partition: PgxPartition, partition_idx: int) -> float:
        """Conductance assesses the quality of a partition in a graph.

        :param graph: Input graph
        :param partition: Partition of the graph with the corresponding node collections
        :param partition_idx: Number of the component to be used for computing its conductance
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.conductance)

        cond = java_handler(
            self._analyst.conductance, [graph._graph, partition._partition, partition_idx]
        )
        return cond.get()

    def partition_conductance(
        self, graph: PgxGraph, partition: PgxPartition
    ) -> Tuple[float, float]:
        """Partition conductance assesses the quality of many partitions in a graph.

        :param graph: Input graph
        :param partition: Partition of the graph with the corresponding node collections
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.partition_conductance)

        pair = java_handler(
            self._analyst.partitionConductance, [graph._graph, partition._partition]
        )
        return (pair.getFirst().get(), pair.getSecond().get())

    def partition_modularity(self, graph: PgxGraph, partition: PgxPartition) -> float:
        """Modularity summarizes information about the quality of components in a graph.

        :param graph: Input graph
        :param partition: Partition of the graph with the corresponding node collections
        :returns: Scalar (double) to store the conductance value of the given cut
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.partition_modularity)

        modularity = java_handler(
            self._analyst.partitionModularity, [graph._graph, partition._partition]
        )
        return modularity.get()

    def scc_kosaraju(
        self, graph: PgxGraph, label: Union[VertexProperty, str] = "scc_kosaraju"
    ) -> PgxPartition:
        """Kosaraju finds strongly connected components in a graph.

        :param graph: Input graph
        :param label: Vertex property holding the degree centrality value for each vertex in the
            graph. Can be a string or a VertexProperty object.
        :returns: Partition holding the node collections corresponding to the components found by
            the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.scc_kosaraju)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(self._analyst.sccKosaraju, [graph._graph, label._prop])
        return PgxPartition(graph, java_partition, label)

    def scc_tarjan(
        self, graph: PgxGraph, label: Union[VertexProperty, str] = "scc_tarjan"
    ) -> PgxPartition:
        """Tarjan finds strongly connected components in a graph.

        :param graph: Input graph
        :param label: Vertex property holding the degree centrality value for each vertex in the
            graph. Can be a string or a VertexProperty object.
        :returns: Partition holding the node collections corresponding to the components found by
            the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.scc_tarjan)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(self._analyst.sccTarjan, [graph._graph, label._prop])
        return PgxPartition(graph, java_partition, label)

    def wcc(self, graph: PgxGraph, label: Union[VertexProperty, str] = "wcc") -> PgxPartition:
        """Identify weakly connected components.

        This can be useful for clustering graph data.

        :param graph: Input graph
        :param label: Vertex property holding the value for each vertex in the graph.
            Can be a string or a VertexProperty object.
        :returns: Partition holding the node collections corresponding to the components found by
            the algorithm.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.wcc)

        if isinstance(label, str):
            label = graph.create_vertex_property("long", label)

        java_partition = java_handler(self._analyst.wcc, [graph._graph, label._prop])
        return PgxPartition(graph, java_partition, label)

    def salsa(
        self,
        bipartite_graph: BipartiteGraph,
        tol: float = 0.001,
        max_iter: int = 100,
        rank: Union[VertexProperty, str] = "salsa",
    ) -> VertexProperty:
        """Stochastic Approach for Link-Structure Analysis (SALSA) computes ranking scores.

        It assesses the quality of information and references in linked structures.

        :param bipartite_graph: Bipartite graph
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the error
            values of all vertices becomes smaller than this value.
        :param max_iter: Maximum number of iterations that will be performed
        :param rank: Vertex property holding the value for each vertex in the graph
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.salsa)

        if isinstance(rank, str):
            rank = bipartite_graph.create_vertex_property("double", rank)

        java_handler(self._analyst.salsa, [bipartite_graph._graph, tol, max_iter, rank._prop])
        return rank

    def personalized_salsa(
        self,
        bipartite_graph: BipartiteGraph,
        v: Union[VertexSet, PgxVertex],
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        rank: Union[VertexProperty, str] = "personalized_salsa",
    ) -> VertexProperty:
        """Personalized SALSA for a vertex of interest.

        Assesses the quality of information and references in linked structures.

        :param bipartite_graph: Bipartite graph
        :param v: The chosen vertex from the graph for personalization
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the
            error values of all vertices becomes smaller than this value.
        :param damping: Damping factor to modulate the degree of personalization of the scores by
            the algorithm
        :param max_iter: Maximum number of iterations that will be performed
        :param rank: (Out argument) vertex property holding the normalized authority/hub
            ranking score for each vertex
        :returns: Vertex property holding the computed scores
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.personalized_salsa)

        if isinstance(rank, str):
            rank = bipartite_graph.create_vertex_property("double", rank)

        if isinstance(v, PgxVertex):
            java_handler(
                self._analyst.personalizedSalsa,
                [bipartite_graph._graph, v._vertex, damping, max_iter, tol, rank._prop],
            )
        if isinstance(v, VertexSet):
            java_handler(
                self._analyst.personalizedSalsa,
                [bipartite_graph._graph, v._collection, damping, max_iter, tol, rank._prop],
            )
        return rank

    def whom_to_follow(
        self,
        graph: PgxGraph,
        v: PgxVertex,
        top_k: int = 100,
        size_circle_of_trust: int = 500,
        tol: float = 0.001,
        damping: float = 0.85,
        max_iter: int = 100,
        salsa_tol: float = 0.001,
        salsa_max_iter: int = 100,
        hubs: Optional[Union[VertexSequence, str]] = None,
        auth: Optional[Union[VertexSequence, str]] = None,
    ) -> Tuple[VertexSequence, VertexSequence]:
        """Whom-to-follow (WTF) is a recommendation algorithm.

        It returns two vertex sequences: one of similar users (hubs) and a second one with users
        to follow (auth).

        :param graph: Input graph
        :param v: The chosen vertex from the graph for personalization of the recommendations
        :param top_k: The maximum number of recommendations that will be returned
        :param size_circle_of_trust: The maximum size of the circle of trust
        :param tol: Maximum tolerated error value. The algorithm will stop once the sum of the error
            values of all vertices becomes smaller than this value.
        :param damping: Damping factor for the Pagerank stage
        :param max_iter: Maximum number of iterations that will be performed for the Pagerank stage
        :param salsa_tol: Maximum tolerated error value for the SALSA stage
        :param salsa_max_iter: Maximum number of iterations that will be performed for the SALSA
            stage
        :param hubs: (Out argument) vertex sequence holding the top rated hub vertices (similar
            users) for the recommendations
        :param auth: (Out argument) vertex sequence holding the top rated authority vertices
            (users to follow) for the recommendations
        :returns: Vertex properties holding hubs and auth
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.whom_to_follow)

        if hubs is None or isinstance(hubs, str):
            hubs = graph.create_vertex_sequence(hubs)

        if auth is None or isinstance(auth, str):
            auth = graph.create_vertex_sequence(auth)

        java_handler(
            self._analyst.whomToFollow,
            [
                graph._graph,
                v._vertex,
                top_k,
                size_circle_of_trust,
                max_iter,
                tol,
                damping,
                salsa_max_iter,
                salsa_tol,
                hubs._collection,
                auth._collection,
            ],
        )
        return (hubs, auth)

    def matrix_factorization_gradient_descent(
        self,
        bipartite_graph: BipartiteGraph,
        weight: EdgeProperty,
        learning_rate: float = 0.001,
        change_per_step: float = 1.0,
        lbd: float = 0.15,
        max_iter: int = 100,
        vector_length: int = 10,
        features: Union[VertexProperty, str] = "features",
    ) -> MatrixFactorizationModel:
        """
        :param bipartite_graph: Input graph
            between 1 and 5, the result will become inaccurate.
        :param learning_rate: Learning rate for the optimization process
        :param change_per_step: Parameter used to modulate the learning rate during the
            optimization process
        :param lbd: Penalization parameter to avoid over-fitting during optimization process
        :param max_iter: Maximum number of iterations that will be performed
        :param vector_length: Size of the feature vectors to be generated for the factorization
        :param features: Vertex property holding the generated feature vectors for each vertex.
            This function accepts names and VertexProperty objects.
        :returns: Matrix factorization model holding the feature vectors found by the algorithm
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.matrix_factorization_gradient_descent)

        if isinstance(features, str):
            features = bipartite_graph.create_vertex_vector_property(
                "double", vector_length, features
            )

        mfm = java_handler(
            self._analyst.matrixFactorizationGradientDescent,
            [
                bipartite_graph._graph,
                weight._prop,
                learning_rate,
                change_per_step,
                lbd,
                max_iter,
                vector_length,
                features._prop,
            ],
        )
        return MatrixFactorizationModel(bipartite_graph, mfm, features)

    def fattest_path(
        self,
        graph: PgxGraph,
        root: PgxVertex,
        capacity: EdgeProperty,
        distance: Union[VertexProperty, str] = "fattest_path_distance",
        parent: Union[VertexProperty, str] = "fattest_path_parent",
        parent_edge: Union[VertexProperty, str] = "fattest_path_parent_edge",
    ) -> AllPaths:
        """Fattest path is a fast algorithm for finding a shortest path adding constraints for
        flowing related matters.

        :param graph: Input graph
        :param root: Fattest path is a fast algorithm for finding a shortest path adding constraints
            for flowing related matters
        :param capacity: Edge property holding the capacity of each edge in the graph
        :param distance: Vertex property holding the capacity value of the fattest path up to the
            current vertex
        :param parent: Vertex property holding the parent vertex of the each vertex in the
            fattest path
        :param parent_edge: Vertex property holding the edge ID linking the current vertex
            in the path with the previous vertex in the path
        :returns: AllPaths object holding the information of the possible fattest paths from the
            source node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.fattest_path)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("double", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        paths = java_handler(
            self._analyst.fattestPath,
            [
                graph._graph,
                root._vertex,
                capacity._prop,
                distance._prop,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return AllPaths(graph, paths)

    def shortest_path_dijkstra(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        weight: EdgeProperty,
        parent: Union[VertexProperty, str] = "dijkstra_parent",
        parent_edge: Union[VertexProperty, str] = "dijkstra_parent_edge",
    ) -> PgxPath:
        """
        :param graph: Input graph
        :param src: Source node
        :param dst: Destination node
        :param weight: Edge property holding the (positive) weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :returns: PgxPath holding the information of the shortest path, if it exists
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_dijkstra)

        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        path = java_handler(
            self._analyst.shortestPathDijkstra,
            [graph._graph, src._vertex, dst._vertex, weight._prop, parent._prop, parent_edge._prop],
        )
        return PgxPath(graph, path)

    def shortest_path_filtered_dijkstra(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        weight: EdgeProperty,
        filter_expression: EdgeFilter,
        parent: Union[VertexProperty, str] = "dijkstra_parent",
        parent_edge: Union[VertexProperty, str] = "dijkstra_parent_edge",
    ) -> PgxPath:
        """
        :param graph: Input graph
        :param src: Source node
        :param dst: Destination node
        :param weight: Edge property holding the (positive) weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :param filter_expression: GraphFilter object for filtering
        :returns: PgxPath holding the information of the shortest path, if it exists
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_filtered_dijkstra)

        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        path = java_handler(
            self._analyst.shortestPathFilteredDijkstra,
            [
                graph._graph,
                src._vertex,
                dst._vertex,
                weight._prop,
                filter_expression._filter,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return PgxPath(graph, path)

    def shortest_path_bidirectional_dijkstra(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        weight: EdgeProperty,
        parent: Union[VertexProperty, str] = "bidirectional_dijkstra_parent",
        parent_edge: Union[VertexProperty, str] = "bidirectional_dijkstra_parent_edge",
    ) -> PgxPath:
        """Bidirectional dijkstra is a fast algorithm for finding a shortest path in a graph.

        :param graph: Input graph
        :param src: Source node
        :param dst: Destination node
        :param weight: Edge property holding the (positive) weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :returns: PgxPath holding the information of the shortest path, if it exists
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_bidirectional_dijkstra)

        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        path = java_handler(
            self._analyst.shortestPathDijkstraBidirectional,
            [graph._graph, src._vertex, dst._vertex, weight._prop, parent._prop, parent_edge._prop],
        )
        return PgxPath(graph, path)

    def shortest_path_filtered_bidirectional_dijkstra(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        weight: EdgeProperty,
        filter_expression: EdgeFilter,
        parent: Union[VertexProperty, str] = "bidirectional_dijkstra_parent",
        parent_edge: Union[VertexProperty, str] = "bidirectional_dijkstra_parent_edge",
    ) -> PgxPath:
        """
        :param graph: Input graph
        :param src: Source node
        :param dst: Destination node
        :param weight: Edge property holding the (positive) weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :param filter_expression: graphFilter object for filtering
        :returns: PgxPath holding the information of the shortest path, if it exists
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_filtered_bidirectional_dijkstra)

        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        path = java_handler(
            self._analyst.shortestPathFilteredDijkstraBidirectional,
            [
                graph._graph,
                src._vertex,
                dst._vertex,
                weight._prop,
                filter_expression._filter,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return PgxPath(graph, path)

    def shortest_path_bellman_ford(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        weight: EdgeProperty,
        distance: Union[VertexProperty, str] = "bellman_ford_distance",
        parent: Union[VertexProperty, str] = "bellman_ford_parent",
        parent_edge: Union[VertexProperty, str] = "bellman_ford_parent_edge",
    ) -> AllPaths:
        """Bellman-Ford finds multiple shortest paths at the same time.

        :param graph: Input graph
        :param src: Source node
        :param distance: (Out argument) vertex property holding the distance to the source vertex
            for each vertex in the graph
        :param weight: Edge property holding the weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :returns: AllPaths holding the information of the possible shortest paths from the source
            node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_bellman_ford)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("double", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        paths = java_handler(
            self._analyst.shortestPathBellmanFord,
            [
                graph._graph,
                src._vertex,
                weight._prop,
                distance._prop,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return AllPaths(graph, paths)

    def shortest_path_bellman_ford_reversed(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        weight: EdgeProperty,
        distance: Union[VertexProperty, str] = "bellman_ford_distance",
        parent: Union[VertexProperty, str] = "bellman_ford_parent",
        parent_edge: Union[VertexProperty, str] = "bellman_ford_parent_edge",
    ) -> AllPaths:
        """Reversed Bellman-Ford finds multiple shortest paths at the same time.

        :param graph: Input graph
        :param src: Source node
        :param distance: (Out argument) vertex property holding the distance to the source
            vertex for each vertex in the graph
        :param weight: Edge property holding the weight of each edge in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path.
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path.
        :returns: AllPaths holding the information of the possible shortest paths from the source
            node.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_bellman_ford_reversed)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("double", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        paths = java_handler(
            self._analyst.shortestPathBellmanFordReverse,
            [
                graph._graph,
                src._vertex,
                weight._prop,
                distance._prop,
                parent._prop,
                parent_edge._prop,
            ],
        )
        return AllPaths(graph, paths)

    def shortest_path_hop_distance(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        distance: Union[VertexProperty, str] = "hop_dist_distance",
        parent: Union[VertexProperty, str] = "hop_dist_parent",
        parent_edge: Union[VertexProperty, str] = "hop_dist_edge",
    ) -> AllPaths:
        """Hop distance can give a relatively fast insight on the distances in a graph.

        :param graph: Input graph
        :param src: Source node
        :param distance: Out argument) vertex property holding the distance to the source vertex
            for each vertex in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :returns: AllPaths holding the information of the possible shortest paths from the source
            node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_hop_distance)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("double", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        paths = java_handler(
            self._analyst.shortestPathHopDist,
            [graph._graph, src._vertex, distance._prop, parent._prop, parent_edge._prop],
        )
        return AllPaths(graph, paths)

    def shortest_path_hop_distance_reversed(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        distance: Union[VertexProperty, str] = "hop_dist_distance",
        parent: Union[VertexProperty, str] = "hop_dist_parent",
        parent_edge: Union[VertexProperty, str] = "hop_dist_edge",
    ) -> AllPaths:
        """Backwards hop distance can give a relatively fast insight on the distances in a graph.

        :param graph: Input graph
        :param src: Source node
        :param distance: Out argument) vertex property holding the distance to the source vertex
            for each vertex in the graph
        :param parent: (Out argument) vertex property holding the parent vertex of the each
            vertex in the shortest path
        :param parent_edge: (Out argument) vertex property holding the edge ID linking the
            current vertex in the path with the previous vertex in the path
        :returns: AllPaths holding the information of the possible shortest paths from the source
            node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.shortest_path_hop_distance_reversed)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("double", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)
        if isinstance(parent_edge, str):
            parent_edge = graph.create_vertex_property("edge", parent_edge)

        paths = java_handler(
            self._analyst.shortestPathHopDistReverse,
            [graph._graph, src._vertex, distance._prop, parent._prop, parent_edge._prop],
        )
        return AllPaths(graph, paths)

    def count_triangles(self, graph: PgxGraph, sort_vertices_by_degree: bool) -> int:
        """Triangle counting gives an overview of the amount of connections between vertices in
        neighborhoods.

        :param graph: Input graph
        :param sort_vertices_by_degree: Boolean flag for sorting the nodes by their degree as
            preprocessing step
        :returns: The total number of triangles found
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.count_triangles)

        return java_handler(self._analyst.countTriangles, [graph._graph, sort_vertices_by_degree])

    def k_core(
        self,
        graph: PgxGraph,
        min_core: int = 0,
        max_core: int = 2147483647,
        kcore: Union[VertexProperty, str] = "kcore",
    ) -> Tuple[int, VertexProperty]:
        """k-core decomposes a graph into layers revealing subgraphs with particular properties.

        :param graph: Input graph
        :param min_core: Minimum k-core value
        :param max_core: Maximum k-core value
        :param kcore: Vertex property holding the result value

        :returns: Pair holding the maximum core found and a node property with the largest k-core
            value for each node.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.k_core)

        if isinstance(kcore, str):
            kcore = graph.create_vertex_property("long", kcore)

        max_k_core = java_handler(graph._graph.createScalar, [property_types["long"]])

        java_handler(
            self._analyst.kcore, [graph._graph, min_core, int(max_core), max_k_core, kcore._prop]
        )
        return (max_k_core.get(), kcore)

    def diameter(
        self, graph: PgxGraph, eccentricity: Union[VertexProperty, str] = "eccentricity"
    ) -> Tuple[int, VertexProperty]:
        """Diameter/radius gives an overview of the distances in a graph.

        :param graph: Input graph
        :param eccentricity: (Out argument) vertex property holding the eccentricity value for
            each vertex
        :returns: Pair holding the diameter of the graph and a node property with eccentricity
            value for each node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.diameter)

        if isinstance(eccentricity, str):
            eccentricity = graph.create_vertex_property("integer", eccentricity)

        diameter = java_handler(graph._graph.createScalar, [property_types["integer"]])

        java_handler(self._analyst.diameter, [graph._graph, diameter, eccentricity._prop])
        return (diameter.get(), eccentricity)

    def radius(
        self, graph: PgxGraph, eccentricity: Union[VertexProperty, str] = "eccentricity"
    ) -> Tuple[int, VertexProperty]:
        """Radius gives an overview of the distances in a graph. it is computed as the minimum
        graph eccentricity.

        :param graph: Input graph
        :param eccentricity: (Out argument) vertex property holding the eccentricity value for
            each vertex
        :returns: Pair holding the radius of the graph and a node property with eccentricity
            value for each node
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.radius)

        if isinstance(eccentricity, str):
            eccentricity = graph.create_vertex_property("integer", eccentricity)

        radius = java_handler(graph._graph.createScalar, [property_types["integer"]])

        java_handler(self._analyst.radius, [graph._graph, radius, eccentricity._prop])
        return (radius.get(), eccentricity)

    def periphery(
        self, graph: PgxGraph, periphery: Optional[Union[VertexSet, str]] = None
    ) -> VertexSet:
        """Periphery/center gives an overview of the extreme distances and the corresponding
        vertices in a graph.

        :param graph: Input graph
        :param periphery: (Out argument) vertex set holding the vertices from the periphery or
            center of the graph
        :returns: Vertex set holding the vertices from the periphery or center of the graph
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.periphery)

        if periphery is None or isinstance(periphery, str):
            periphery = graph.create_vertex_set(periphery)

        java_handler(self._analyst.periphery, [graph._graph, periphery._collection])
        return periphery

    def center(self, graph: PgxGraph, center: Optional[Union[VertexSet, str]] = None) -> VertexSet:
        """Periphery/center gives an overview of the extreme distances and the corresponding
        vertices in a graph.

        The center is comprised by the set of vertices with eccentricity equal to the radius of
        the graph.

        :param graph: Input graph
        :param center: (Out argument) vertex set holding the vertices from the periphery or
            center of the graph
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.center)

        if center is None or isinstance(center, str):
            center = graph.create_vertex_set(center)

        java_handler(self._analyst.center, [graph._graph, center._collection])
        return center

    def local_clustering_coefficient(
        self, graph: PgxGraph, lcc: Union[VertexProperty, str] = "lcc"
    ) -> VertexProperty:
        """LCC gives information about potential clustering options in a graph.

        :param graph: Input graph
        :param lcc: Vertex property holding the lcc value for each vertex
        :returns: Vertex property holding the lcc value for each vertex
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.local_clustering_coefficient)

        if isinstance(lcc, str):
            lcc = graph.create_vertex_property("double", lcc)

        java_handler(self._analyst.localClusteringCoefficient, [graph._graph, lcc._prop])
        return lcc

    def find_cycle(
        self,
        graph: PgxGraph,
        src: Optional[PgxVertex] = None,
        vertex_seq: Optional[Union[VertexSequence, str]] = None,
        edge_seq: Optional[Union[EdgeSequence, str]] = None,
    ) -> PgxPath:
        """Find cycle looks for any loop in the graph.

        :param graph: Input graph
        :param src: Source vertex for the search
        :param vertex_seq: (Out argument) vertex sequence holding the vertices in the cycle
        :param edge_seq: (Out argument) edge sequence holding the edges in the cycle
        :returns: PgxPath representing the cycle as path, if exists.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.find_cycle)

        if vertex_seq is None or isinstance(vertex_seq, str):
            vertex_seq = graph.create_vertex_sequence(vertex_seq)

        if edge_seq is None or isinstance(edge_seq, str):
            edge_seq = graph.create_edge_sequence(edge_seq)

        cycle = None
        if src is None:
            cycle = java_handler(
                self._analyst.findCycle,
                [graph._graph, vertex_seq._collection, edge_seq._collection],
            )
        if isinstance(src, PgxVertex):
            cycle = java_handler(
                self._analyst.findCycle,
                [graph._graph, src._vertex, vertex_seq._collection, edge_seq._collection],
            )
        return PgxPath(graph, cycle)

    def reachability(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        max_hops: int,
        ignore_edge_direction: bool,
    ) -> int:
        """Reachability is a fast way to check if two vertices are reachable from each other.

        :param graph: Input graph
        :param src: Source vertex for the search
        :param dst: Destination vertex for the search
        :param max_hops: Maximum hop distance between the source and destination vertices
        :param ignore_edge_direction: Boolean flag for ignoring the direction of the edges during
            the search
        :returns: The number of hops between the vertices. It will return -1 if the vertices are
            not connected or are not reachable given the condition of the maximum hop distance
            allowed.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.reachability)

        return java_handler(
            self._analyst.reachability,
            [graph._graph, src._vertex, dst._vertex, max_hops, ignore_edge_direction],
        )

    def topological_sort(
        self, graph: PgxGraph, topo_sort: Union[VertexProperty, str] = "topo_sort"
    ) -> VertexProperty:
        """Topological sort gives an order of visit for vertices in directed acyclic graphs.

        :param graph: Input graph
        :param topo_sort: (Out argument) vertex property holding the topological order of each
            vertex
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.topological_sort)

        if isinstance(topo_sort, str):
            topo_sort = graph.create_vertex_property("integer", topo_sort)

        java_handler(self._analyst.topologicalSort, [graph._graph, topo_sort._prop])
        return topo_sort

    def topological_schedule(
        self, graph: PgxGraph, vs: VertexSet, topo_sched: Union[VertexProperty, str] = "topo_sched"
    ) -> VertexProperty:
        """Topological schedule gives an order of visit for the reachable vertices from the source.

        :param graph: Input graph
        :param vs: Set of vertices to be used as the starting points for the scheduling order
        :param topo_sched: (Out argument) vertex property holding the scheduled order of each
            vertex
        :returns: Vertex property holding the scheduled order of each vertex.
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.topological_schedule)

        if isinstance(topo_sched, str):
            topo_sched = graph.create_vertex_property("integer", topo_sched)

        java_handler(
            self._analyst.topologicalSchedule, [graph._graph, vs._collection, topo_sched._prop]
        )
        return topo_sched

    def out_degree_distribution(
        self, graph: PgxGraph, dist_map: Optional[Union[PgxMap, str]] = None
    ) -> PgxMap:
        """
        :param graph: Input graph
        :param dist_map: (Out argument) map holding a histogram of the vertex degrees in the graph
        :returns: Map holding a histogram of the vertex degrees in the graph
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.out_degree_distribution)

        if dist_map is None or isinstance(dist_map, str):
            dist_map = graph.create_map('integer', 'long', dist_map)

        java_handler(self._analyst.outDegreeDistribution, [graph._graph, dist_map._map])
        return dist_map

    def in_degree_distribution(
        self, graph: PgxGraph, dist_map: Optional[Union[PgxMap, str]] = None
    ) -> PgxMap:
        """Calculate the in-degree distribution.

        In-degree distribution gives information about the incoming flows in a graph.

        :param graph: Input graph
        :param dist_map: (Out argument) map holding a histogram of the vertex degrees in the graph
        :returns: Map holding a histogram of the vertex degrees in the graph
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.in_degree_distribution)

        if dist_map is None or isinstance(dist_map, str):
            dist_map = graph.create_map("integer", "long", dist_map)

        java_handler(self._analyst.inDegreeDistribution, [graph._graph, dist_map._map])
        return dist_map

    def prim(
        self, graph: PgxGraph, weight: EdgeProperty, mst: Union[EdgeProperty, str] = "mst"
    ) -> EdgeProperty:
        """Prim reveals tree structures with shortest paths in a graph.

        :param graph: Input graph
        :param weight: Edge property holding the weight of each edge in the graph
        :param mst: Edge property holding the edges belonging to the minimum spanning tree of
            the graph
        :returns: Edge property holding the edges belonging to the minimum spanning tree
            of the graph (i.e. all the edges with in_mst=true)
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.prim)

        if isinstance(mst, str):
            mst = graph.create_edge_property("boolean", mst)

        java_handler(self._analyst.prim, [graph._graph, weight._prop, mst._prop])
        return mst

    def filtered_bfs(
        self,
        graph: PgxGraph,
        root: PgxVertex,
        navigator: VertexFilter,
        init_with_inf: bool = True,
        max_depth: int = 2147483647,
        distance: Union[VertexProperty, str] = "distance",
        parent: Union[VertexProperty, str] = "parent",
    ) -> Tuple[VertexProperty, VertexProperty]:
        """Breadth-first search with an option to filter edges during the traversal of the graph.

        :param graph: Input graph
        :param root: The source vertex from the graph for the path.
        :param navigator: Navigator expression to be evaluated on the vertices during the graph
            traversal
        :param init_with_inf: Boolean flag to set the initial distance values of the vertices.
            If set to true, it will initialize the distances as INF, and -1 otherwise.
        :param max_depth: Maximum depth limit for the BFS traversal
        :param distance: Vertex property holding the hop distance for each reachable vertex in
            the graph
        :param parent: Vertex property holding the parent vertex of the each reachable vertex in
            the path
        :returns: Distance and parent vertex properties
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.filtered_bfs)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("integer", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)

        java_handler(
            self._analyst.filteredBfs,
            [
                graph._graph,
                root._vertex,
                navigator._filter,
                init_with_inf,
                distance._prop,
                parent._prop,
            ],
        )
        return (distance, parent)

    def filtered_dfs(
        self,
        graph: PgxGraph,
        root: PgxVertex,
        navigator: VertexFilter,
        init_with_inf: bool = True,
        max_depth: int = 2147483647,
        distance: Union[VertexProperty, str] = "distance",
        parent: Union[VertexProperty, str] = "parent",
    ) -> Tuple[VertexProperty, VertexProperty]:
        """Depth-first search with an option to filter edges during the traversal of the graph.

        :param graph: Input graph
        :param root: The source vertex from the graph for the path
        :param navigator: Navigator expression to be evaluated on the vertices during the graph
            traversal
        :param init_with_inf: Boolean flag to set the initial distance values of the vertices.
            If set to true, it will initialize the distances as INF, and -1 otherwise.
        :param max_depth: Maximum search depth
        :param distance: Vertex property holding the hop distance for each reachable vertex in
            the graph
        :param parent: Vertex property holding the parent vertex of the each reachable vertex in
            the path
        :returns: Distance and parent vertex properties
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.filtered_dfs)

        if isinstance(distance, str):
            distance = graph.create_vertex_property("integer", distance)
        if isinstance(parent, str):
            parent = graph.create_vertex_property("vertex", parent)

        java_handler(
            self._analyst.filteredDfs,
            [
                graph._graph,
                root._vertex,
                navigator._filter,
                init_with_inf,
                distance._prop,
                parent._prop,
            ],
        )
        return distance, parent

    def all_reachable_vertices_edges(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        k: int,
        filter: Optional[EdgeFilter] = None,
    ) -> Tuple[VertexSet, EdgeSet, PgxMap]:
        """Find all the vertices and edges on a path between the src and target of length smaller
        or equal to k.

        :param graph: Input graph
        :param src: The source vertex
        :param dst: The destination vertex
        :param k: The dimension of the distances property; i.e. number of high-degree vertices.
        :param filter: The filter to be used on edges when searching for a path
        :return: The vertices on the path, the edges on the path and a map containing the
            distances from the source vertex for each vertex on the path
        """
        arguments = locals()
        validate_arguments(arguments, alg_metadata.all_reachable_vertices_edges)

        java_args = [graph._graph, src._vertex, dst._vertex, k]
        if filter is None:
            java_triple = java_handler(self._analyst.allReachableVerticesEdges, java_args)
        else:
            java_args.append(filter._filter)
            java_triple = java_handler(self._analyst.allReachableVerticesEdgesFiltered, java_args)
        return (
            VertexSet(graph, java_triple.left),
            EdgeSet(graph, java_triple.middle),
            PgxMap(graph, java_triple.right),
        )

    def compute_high_degree_vertices(
        self,
        graph: PgxGraph,
        k: int,
        high_degree_vertex_mapping: Optional[Union[PgxMap, str]] = None,
        high_degree_vertices: Optional[Union[VertexSet, str]] = None,
    ) -> Tuple[PgxMap, VertexSet]:
        """Compute the k vertices with the highest degrees in the graph.

        :param graph: Input graph
        :param k: Number of high-degree vertices to be computed
        :param high_degree_vertex_mapping: (out argument) map with the top k high-degree vertices
            and their indices
        :param high_degree_vertices: (out argument) the high-degree vertices
        :return: a map with the top k high-degree vertices and their indices and a vertex
            set containing the same vertices
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.compute_high_degree_vertices)

        if high_degree_vertex_mapping is None or isinstance(high_degree_vertex_mapping, str):
            high_degree_vertex_mapping = graph.create_map(
                "integer", "vertex", high_degree_vertex_mapping
            )

        if high_degree_vertices is None or isinstance(high_degree_vertices, str):
            high_degree_vertices = graph.create_vertex_set(high_degree_vertices)

        java_handler(
            self._analyst.computeHighDegreeVertices,
            [graph._graph, k, high_degree_vertex_mapping._map, high_degree_vertices._collection],
        )
        return high_degree_vertex_mapping, high_degree_vertices

    def create_distance_index(
        self,
        graph: PgxGraph,
        high_degree_vertex_mapping: PgxMap,
        high_degree_vertices: VertexSet,
        index: Optional[Union[VertexProperty, str]] = None,
    ) -> VertexProperty:
        """Compute an index with distances to each high-degree vertex

        :param graph: Input graph
        :param high_degree_vertex_mapping: a map with the top k high-degree vertices and their
            indices and a vertex
        :param high_degree_vertices: the high-degree vertices
        :param index: (out-argument) the index containing the distances to each high-degree
            vertex for all vertices
        :return: the index containing the distances to each high-degree vertex for all vertices
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.create_distance_index)

        if index is None or isinstance(index, str):
            dim = len(high_degree_vertices)
            index = graph.create_vertex_vector_property("integer", dim, index)

        java_handler(
            self._analyst.createDistanceIndex,
            [
                graph._graph,
                high_degree_vertex_mapping._map,
                high_degree_vertices._collection,
                index._prop,
            ],
        )
        return index

    def bipartite_check(
        self, graph: PgxGraph, is_left: Union[VertexProperty, str] = "is_left"
    ) -> VertexProperty:
        """Verify whether a graph is bipartite.

        :param graph: Input graph
        :param is_left: (out-argument) vertex property holding the side of each
            vertex in a bipartite graph (true for left, false for right).
        :return: vertex property holding the side of each
            vertex in a bipartite graph (true for left, false for right).
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.bipartite_check)

        if is_left is None or isinstance(is_left, str):
            is_left = graph.create_vertex_property("boolean", is_left)

        java_handler(self._analyst.bipartiteCheck, [graph._graph, is_left._prop])
        return is_left

    def enumerate_simple_paths(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        k: int,
        vertices_on_path: VertexSet,
        edges_on_path: EdgeSet,
        dist: PgxMap,
    ) -> Tuple[List[int], VertexSet, EdgeSet]:
        """Enumerate simple paths between the source and destination vertex.

        :param graph: Input graph
        :param src: The source vertex
        :param dst: The destination vertex
        :param k: maximum number of iterations
        :param vertices_on_path: VertexSet containing all vertices to be considered while
            enumerating paths
        :param edges_on_path: EdgeSet containing all edges to be consider while enumerating paths
        :param dist: map containing the hop-distance from the source vertex to each vertex that is
            to be considered while enumerating the paths
        :return: Triple containing containing the path lengths, a vertex-sequence
            containing the vertices on the paths and edge-sequence containing the edges on the
            paths
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.enumerate_simple_paths)

        java_triple = java_handler(
            self._analyst.enumerateSimplePaths,
            [
                graph._graph,
                src._vertex,
                dst._vertex,
                k,
                vertices_on_path._collection,
                edges_on_path._collection,
                dist._map,
            ],
        )

        path_lengths = list(java_triple.left)
        path_vertices = VertexSet(graph, java_triple.middle)
        path_edges = EdgeSet(graph, java_triple.right)
        return path_lengths, path_vertices, path_edges

    def limited_shortest_path_hop_dist(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        max_hops: int,
        high_degree_vertex_mapping: PgxMap,
        high_degree_vertices: VertexSet,
        index: VertexProperty,
        path_vertices: Optional[Union[VertexSequence, str]] = None,
        path_edges: Optional[Union[EdgeSequence, str]] = None,
    ) -> Tuple[VertexSequence, EdgeSequence]:
        """Compute the shortest path between the source and destination vertex.

        The algorithm only considers paths up to a length of k.

        :param graph: Input graph
        :param src: The source vertex
        :param dst: The destination vertex
        :param max_hops: The maximum number of edges to follow when trying to find a path
        :param high_degree_vertex_mapping: Map with the top k high-degree vertices
            and their indices
        :param high_degree_vertices: The high-degree vertices
        :param index: Index containing distances to high-degree vertices
        :param path_vertices: (out-argument) will contain the vertices on the found path
            or will be empty if there is none
        :param path_edges: (out-argument) will contain the vertices on the found path or
            will be empty if there is none
        :return: A tuple containing the vertices in the shortest path from src to dst and the
            edges on the path. Both will be empty if there is no path within maxHops steps
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.limited_shortest_path_hop_dist)

        if path_vertices is None or isinstance(path_vertices, str):
            path_vertices = graph.create_vertex_sequence()

        if path_edges is None or isinstance(path_edges, str):
            path_edges = graph.create_edge_sequence()

        pair = java_handler(
            self._analyst.limitedShortestPathHopDist,
            [
                graph._graph,
                src._vertex,
                dst._vertex,
                max_hops,
                high_degree_vertex_mapping._map,
                high_degree_vertices._collection,
                index._prop,
                path_vertices._collection,
                path_edges._collection,
            ],
        )
        return VertexSequence(graph, pair.getFirst()), EdgeSequence(graph, pair.getSecond())

    def limited_shortest_path_hop_dist_filtered(
        self,
        graph: PgxGraph,
        src: PgxVertex,
        dst: PgxVertex,
        max_hops: int,
        high_degree_vertex_mapping: PgxMap,
        high_degree_vertices: VertexSet,
        index: VertexProperty,
        filter: EdgeFilter,
        path_vertices: Optional[Union[VertexSequence, str]] = None,
        path_edges: Optional[Union[EdgeSequence, str]] = None,
    ) -> Tuple[VertexSequence, EdgeSequence]:
        """Compute the shortest path between the source and destination vertex.

        The algorithm only considers paths up to a length of k.

        :param graph: Input graph
        :param src: The source vertex
        :param dst: The destination vertex
        :param max_hops: The maximum number of edges to follow when trying to find a path
        :param high_degree_vertex_mapping: Map with the top k high-degree vertices
            and their indices
        :param high_degree_vertices: The high-degree vertices
        :param index: Index containing distances to high-degree vertices
        :param filter: Filter to be evaluated on the edges when searching for a path
        :param path_vertices: (out-argument) will contain the vertices on the found path
            or will be empty if there is none
        :param path_edges: (out-argument) will contain the vertices on the found path or
            will be empty if there is none
        :return: A tuple containing the vertices in the shortest path from src to dst and the
            edges on the path. Both will be empty if there is no path within maxHops steps
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.limited_shortest_path_hop_dist_filtered)

        if path_vertices is None or isinstance(path_vertices, str):
            path_vertices = graph.create_vertex_sequence(path_vertices)

        if path_edges is None or isinstance(path_edges, str):
            path_edges = graph.create_edge_sequence(path_edges)

        pair = java_handler(
            self._analyst.limitedShortestPathHopDistFiltered,
            [
                graph._graph,
                src._vertex,
                dst._vertex,
                max_hops,
                high_degree_vertex_mapping._map,
                high_degree_vertices._collection,
                index._prop,
                filter._filter,
                path_vertices._collection,
                path_edges._collection,
            ],
        )
        return VertexSequence(graph, pair.getFirst()), EdgeSequence(graph, pair.getSecond())

    def random_walk_with_restart(
        self,
        graph: PgxGraph,
        source: PgxVertex,
        length: int,
        reset_prob: float,
        visit_count: Optional[PgxMap] = None,
    ) -> PgxMap:
        """Perform a random walk over the graph.

        The walk will start at the given source vertex and will randomly visit neighboring vertices
        in the graph, with a probability equal to the value of reset_probability of going back to
        the starting point.  The random walk will also go back to the starting point every time it
        reaches a vertex with no outgoing edges. The algorithm will stop once it reaches the
        specified walk length.

        :param graph: Input graph
        :param source: Starting point of the random walk
        :param length: Length (number of steps) of the random walk
        :param reset_prob: Probability value for resetting the random walk
        :param visit_count: (out argument) map holding the number of visits during the random walk
            for each vertex in the graph
        :return: map holding the number of visits during the random walk for each vertex in the
            graph
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.random_walk_with_restart)

        if visit_count is None:
            visit_count = graph.create_map("vertex", "integer")

        java_map = java_handler(
            self._analyst.randomWalkWithRestart,
            [graph._graph, source._vertex, length, reset_prob, visit_count._map],
        )
        return PgxMap(graph, java_map)

    def matrix_factorization_recommendations(
        self,
        bipartite_graph: BipartiteGraph,
        user: PgxVertex,
        vector_length: int,
        feature: VertexProperty,
        estimated_rating: Optional[Union[VertexProperty, str]] = None,
    ) -> VertexProperty:
        """Complement for Matrix Factorization.

        The generated feature vectors will be used for making predictions in cases where the given
        user vertex has not been related to a particular item from the item set. Similarly to the
        recommendations from matrix factorization, this algorithm will perform dot products between
        the given user vertex and the rest of vertices in the graph, giving a score of 0 to the
        items that are already related to the user and to the products with other user vertices,
        hence returning the results of the dot products for the unrelated item vertices. The scores
        from those dot products can be interpreted as the predicted scores for the unrelated items
        given a particular user vertex.

        :param bipartite_graph: Bipartite input graph
        :param user: Vertex from the left (user) side of the graph
        :param vector_length: size of the feature vectors
        :param feature: vertex property holding the feature vectors for each vertex
        :param estimated_rating: (out argument) vertex property holding the estimated rating score
            for each vertex
        :return: vertex property holding the estimated rating score for each vertex
        """

        arguments = locals()
        validate_arguments(arguments, alg_metadata.matrix_factorization_recommendations)

        if estimated_rating is None or isinstance(estimated_rating, str):
            estimated_rating = bipartite_graph.create_vertex_property("double", estimated_rating)

        java_handler(
            self._analyst.matrixFactorizationRecommendations,
            [
                bipartite_graph._graph,
                user._vertex,
                vector_length,
                feature._prop,
                estimated_rating._prop,
            ],
        )
        return estimated_rating

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
