#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from collections.abc import Iterable

from jnius import autoclass, JavaException
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import MODEL_NOT_FITTED, VERTEX_ID_OR_COLLECTION_OF_IDS

from pypgx.api.frames import PgxFrame
from pypgx.api.mllib._model_utils import ModelStorer
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.pyjnius_helper import PyjniusHelper
from pypgx.api._pgx_graph import PgxGraph
from typing import List, Optional, Union


class DeepWalkModel(PgxContextManager):
    """DeepWalk model object."""

    _java_class = 'oracle.pgx.api.mllib.DeepWalkModel'

    def __init__(self, java_deepwalk_model) -> None:
        self._model = java_deepwalk_model
        self.min_word_frequency = java_deepwalk_model.getMinWordFrequency()
        self.batch_size = java_deepwalk_model.getBatchSize()
        self.num_epochs = java_deepwalk_model.getNumEpochs()
        self.layer_size = java_deepwalk_model.getLayerSize()
        self.learning_rate = java_deepwalk_model.getLearningRate()
        self.min_learning_rate = java_deepwalk_model.getMinLearningRate()
        self.window_size = java_deepwalk_model.getWindowSize()
        self.walk_length = java_deepwalk_model.getWalkLength()
        self.walks_per_vertex = java_deepwalk_model.getWalksPerVertex()
        self.sample_rate = java_deepwalk_model.getSampleRate()
        self.negative_sample = java_deepwalk_model.getNegativeSample()
        self.validation_fraction = java_deepwalk_model.getValidationFraction()
        self.graph: Optional[PgxGraph] = None

        # Determining whether the model has been fitted is relevant especially for
        # models that are being loaded from a file.
        try:
            self.loss = java_deepwalk_model.getLoss()
        except JavaException:
            self.loss = None
        try:
            self.seed = java_deepwalk_model.getSeed()
        except JavaException:
            self.seed = None

        self._is_fitted = self.loss is not None

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
        """Return a ModelStorer object which can be used to save the model.

        :returns: ModelStorer object
        :rtype: ModelStorer
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
    def trained_vectors(self) -> PgxFrame:
        """Get the trained vertex vectors for the current DeepWalk model.

        :returns: PgxFrame object with the trained vertex vectors
        :rtype: PgxFrame
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        return PgxFrame(java_handler(self._model.getTrainedVertexVectors, []))

    def compute_similars(self, v: Union[int, str, List[int], List[str]], k: int) -> PgxFrame:
        """Compute the top-k similar vertices for a given vertex.

        :param v: id of the vertex or list of vertex ids for which to compute the similar vertices
        :param k: number of similar vertices to return
        """
        if not self._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        if isinstance(v, (int, str)):
            # Pass on `v` and `k` directly to Java.
            v = str(v)
            return self._compute_similars(v, k)
        if isinstance(v, Iterable):
            # Convert `v` from a Python iterable to a Java ArrayList before passing it on.
            vids = autoclass('java.util.ArrayList')()
            for i in v:
                if not isinstance(i, (int, str)):
                    raise TypeError(VERTEX_ID_OR_COLLECTION_OF_IDS.format(var='v'))
                vids.add(str(i))
            return self._compute_similars_list(vids, k)
        raise TypeError(VERTEX_ID_OR_COLLECTION_OF_IDS.format(var='v'))

    def _compute_similars(self, v, k: int) -> PgxFrame:
        return PgxFrame(java_handler(self._model.computeSimilars, [v, k]))

    def _compute_similars_list(self, v, k: int):
        return PgxFrame(java_handler(PyjniusHelper.computeSimilarsList, [self._model, v, k]))

    def destroy(self) -> None:
        """Destroy this model object."""
        java_handler(self._model.destroy, [])

    def close(self) -> None:
        """Call destroy"""
        self.destroy()

    def _get_col_names(self, vec):
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
