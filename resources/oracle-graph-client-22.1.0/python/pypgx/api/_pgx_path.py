#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from itertools import islice, zip_longest
from pypgx.api._pgx_entity import PgxEdge, PgxVertex
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler
from typing import List, Optional, Tuple, Iterator, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class PgxPath(PgxContextManager):
    """A path from a source to a destination vertex in a :class:`PgxGraph`."""

    _java_class = 'oracle.pgx.api.PgxPath'

    def __init__(self, graph: "PgxGraph", java_path) -> None:
        self._path = java_path
        self.exists = java_path.exists()
        src = java_path.getSource()
        self.source = PgxVertex(graph, src) if src is not None else None
        dst = java_path.getDestination()
        self.destination = PgxVertex(graph, dst) if dst is not None else None
        self.cost = java_path.getPathLengthWithCost()
        self.hops = java_path.getPathLengthWithHop()
        self.graph = graph

    @property
    def vertices(self) -> List[PgxVertex]:
        """Return a list of vertices in the path."""
        it = self._path.getVertices().iterator()
        return list(PgxVertex(self.graph, item) for item in islice(it, 0, None))

    @property
    def edges(self) -> List[PgxEdge]:
        """Return a list of edges in the path."""
        it = self._path.getEdges().iterator()
        return list(PgxEdge(self.graph, item) for item in islice(it, 0, None))

    @property
    def path(self) -> List[Tuple[PgxVertex, Optional[PgxEdge]]]:
        """Return path as a list of (vertex,edge) tuples."""
        return list(zip_longest(self.vertices, self.edges))

    def destroy(self) -> None:
        """Destroy this path."""
        java_handler(self._path.destroy, [])

    def __len__(self) -> int:
        return self.hops

    def __iter__(self) -> Iterator[Tuple[PgxVertex, Optional[PgxEdge]]]:
        return islice(self.path, 0, None)

    def __getitem__(
        self, idx: Union[slice, int]
    ) -> Union[List[Tuple[PgxVertex, Optional[PgxEdge]]], Tuple[PgxVertex, Optional[PgxEdge]]]:
        if isinstance(idx, slice):
            return list(islice(self, idx.start, idx.stop, idx.step))
        else:
            return list(islice(self, idx, idx + 1))[0]

    def __repr__(self) -> str:
        return "{}(graph: {}, src: {}, dst: {}, num. edges: {} cost: {})".format(
            self.__class__.__name__,
            self.graph.name,
            self.source,
            self.destination,
            self.hops,
            self.cost,
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(
            (
                str(self),
                str(self.graph.name),
                str(self.source),
                str(self.destination),
                str(self.cost),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._path.equals(other._path)
