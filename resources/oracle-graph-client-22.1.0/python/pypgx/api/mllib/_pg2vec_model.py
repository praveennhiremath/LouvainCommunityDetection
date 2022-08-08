#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import collections.abc

from jnius import autoclass, JavaException
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import MODEL_NOT_FITTED, VERTEX_ID_OR_COLLECTION_OF_IDS

from pypgx.api.frames import PgxFrame
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx.api.mllib._model_utils import ModelStorer
from pypgx._utils.pyjnius_helper import PyjniusHelper
from pypgx.api._pgx_graph import PgxGraph
from typing import List, Optional, Union, Iterable


class Pg2vecModel(PgxContextManager):
    """Pg2Vec model object."""

    _java_class = 'oracle.pgx.api.mllib.Pg2vecModel'

    def __init__(self, java_pg2vec_model) -> None:
        self._model = java_pg2vec_model
        self.graphlet_id_property_name = java_pg2vec_model.getGraphLetIdPropertyName()
        self.vertex_property_names = java_pg2vec_model.getVertexPropertyNames()
        self.min_word_frequency = java_pg2vec_model.getMinWordFrequency()
        self.batch_size = java_pg2vec_model.getBatchSize()
        self.num_epochs = java_pg2vec_model.getNumEpochs()
        self.layer_size = java_pg2vec_model.getLayerSize()
        self.learning_rate = java_pg2vec_model.getLearningRate()
        self.min_learning_rate = java_pg2vec_model.getMinLearningRate()
        self.window_size = java_pg2vec_model.getWindowSize()
        self.walk_length = java_pg2vec_model.getWalkLength()
        self.walks_per_vertex = java_pg2vec_model.getWalksPerVertex()
        self.use_graphlet_size = java_pg2vec_model.getUseGraphletSize()
        self.validation_fraction = java_pg2vec_model.getValidationFraction()
        self.graphlet_size_property_name = java_pg2vec_model.getGraphletSizePropertyName()
        self.graph: Optional[PgxGraph] = None

        # Determining whether the model has been fitted is relevant especially for
        # models that are being loaded from a file.
        try:
            self.loss = java_pg2vec_model.getLoss()
        except JavaException:
            self.loss = None
        self._is_fitted = self.loss is not None

        try:
            self.seed = java_pg2vec_model.getSeed()
        except JavaException:
            self.seed = None

    def store(self, path: str, key: Optional[str], overwrite: bool = False) -> None:
        """Store the model in a file.

        :param path: Path where to store the model
        :param key: Encryption key
        :param overwrite: Whether or not to overwrite pre-existing file
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        java_handler(self._model.store, [path, key, overwrite])

    def export(self) -> ModelStorer:
        """Return a ModelStore object which can be used to save the model.

        :returns: ModelStore object
        """
        return ModelStorer(self)

    def fit(self, graph: PgxGraph) -> None:
        """Fit the model on a graph.

        :param graph: Graph to fit on
        """
        java_handler(self._model.fit, [graph._graph])
        self._is_fitted = True
        self.graph = graph
        self.loss = self._model.getLoss()

    @property
    def trained_graphlet_vectors(self) -> PgxFrame:
        """Get the trained graphlet vectors for the current pg2vec model.

        :returns: PgxFrame containing the trained graphlet vectors
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        return PgxFrame(java_handler(self._model.getTrainedGraphletVectors, []))

    def infer_graphlet_vector(self, graph: PgxGraph) -> PgxFrame:
        """
        :param graph: graphlet for which to infer a vector
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        return PgxFrame(java_handler(self._model.inferGraphletVector, [graph._graph]))

    def infer_graphlet_vector_batched(self, graph: PgxGraph) -> PgxFrame:
        """
        :param graph: graphlets (as a single graph but different graphlet-id) for which to infer
            vectors
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        return PgxFrame(java_handler(self._model.inferGraphletVectorBatched, [graph._graph]))

    def compute_similars(
        self, graphlet_id: Union[Iterable[Union[int, str]], int, str], k: int
    ) -> PgxFrame:
        """Compute the top-k similar graphlets for a list of input graphlets.

        :param graphlet_id: graphletIds or iterable of graphletIds
        :param k: number of similars to return
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        if isinstance(graphlet_id, (int, str)):
            # Pass on `graphlet_id` and `k` directly to Java.
            return self._compute_similars(str(graphlet_id), k)
        if isinstance(graphlet_id, collections.abc.Iterable):
            # Convert `graphlet_id` from a Python iterable to a Java ArrayList before passing it on.
            ids = autoclass('java.util.ArrayList')()
            for i in graphlet_id:
                if not isinstance(i, (int, str)):
                    raise TypeError(VERTEX_ID_OR_COLLECTION_OF_IDS.format(var='graphlet_id'))
                ids.add(str(i))
            return self._compute_similars_list(ids, k)
        raise TypeError(VERTEX_ID_OR_COLLECTION_OF_IDS.format(var='graphlet_id'))

    def _compute_similars(self, v, k: int) -> PgxFrame:
        return PgxFrame(java_handler(self._model.computeSimilars, [v, k]))

    def _compute_similars_list(self, v, k: int) -> PgxFrame:
        return PgxFrame(java_handler(PyjniusHelper.computeSimilarsList, [self._model, v, k]))

    def destroy(self) -> None:
        """Destroy this model object."""
        java_handler(self._model.destroy, [])

    def close(self) -> None:
        """Call destroy"""
        self.destroy()

    def _get_col_names(self, vec) -> List[str]:
        result_elements = vec.getPgqlResultElements()
        col_names = []
        for idx in range(result_elements.size()):
            col_names.append(result_elements.get(idx).getVarName())
        return col_names

    def __repr__(self) -> str:
        if self.graph is not None:
            return "{}(graph: {}, loss: {}, vector dimension: {})".format(
                self.__class__.__name__, self.graph.name, self.loss, self.layer_size
            )
        else:
            return self.__class__.__name__

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._model.equals(other._model)
