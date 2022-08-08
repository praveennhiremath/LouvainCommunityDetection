#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from typing import TYPE_CHECKING

from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_path import PgxPath
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import ARG_MUST_BE

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class AllPaths(PgxContextManager):
    """The paths from one source vertex to all other vertices."""

    _java_class = 'oracle.pgx.api.AllPaths'

    def __init__(self, graph: "PgxGraph", java_all_paths) -> None:
        self._all_paths = java_all_paths
        self.source = PgxVertex(graph, java_all_paths.getSource())
        self.graph = graph

    def get_path(self, destination: PgxVertex) -> PgxPath:
        """Get the path.

        :param destination: The destination node
        :returns: The path result to the destination node
        """
        if not isinstance(destination, PgxVertex):
            raise TypeError(ARG_MUST_BE.format(arg='destination', type=PgxVertex.__name__))
        java_path = java_handler(self._all_paths.getPath, [destination._vertex])
        return PgxPath(self.graph, java_path)

    def destroy(self) -> None:
        """Destroy this object."""
        java_handler(self._all_paths.destroy, [])

    def __getitem__(self, destination: PgxVertex) -> PgxPath:
        return self.get_path(destination)

    def __repr__(self) -> str:
        return "{}(source: {})".format(self.__class__.__name__, self.source)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((str(self), str(self.graph.name), str(self.source)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._all_paths.equals(other._all_paths)
