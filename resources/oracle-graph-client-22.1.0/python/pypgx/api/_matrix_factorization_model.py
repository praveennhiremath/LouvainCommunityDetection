#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import Union, TYPE_CHECKING

from pypgx.api._pgx_entity import PgxVertex
from pypgx._utils.error_handling import java_handler

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class MatrixFactorizationModel:
    """Object that holds the state for repeatedly returning estimated ratings."""

    _java_class = 'oracle.pgx.api.MatrixFactorizationModel'

    def __init__(self, graph: "PgxGraph", java_mfm, features) -> None:
        self._mfm = java_mfm
        self.root_mean_square_error = java_mfm.getRootMeanSquareError()
        self.features = features
        self.graph = graph

    def get_estimated_ratings(self, v: Union[PgxVertex, str, int]) -> float:
        """Return estimated ratings for a specific vertex.

        :param v: The vertex to get estimated ratings for.
        :returns: The VertexProperty containing the estimated ratings.
        """
        if not isinstance(v, PgxVertex):
            v = self.graph.get_vertex(v)

        prop = java_handler(self._mfm.getEstimatedRatings, [v._vertex])
        rating = prop.get(v.id)
        prop.destroy()
        return rating

    def __repr__(self) -> str:
        return "{}(graph: {}, rmse: {}, features dimension: {})".format(
            self.__class__.__name__,
            self.graph.name,
            self.root_mean_square_error,
            self.features.dimension,
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(
            (str(self), str(self.graph.name), str(self.features), str(self.root_mean_square_error))
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._mfm.equals(other._mfm)
