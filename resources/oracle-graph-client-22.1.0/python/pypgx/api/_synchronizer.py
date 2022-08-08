#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from typing import TYPE_CHECKING

from pypgx._utils.error_handling import java_handler

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class Synchronizer:
    """A class for synchronizing changes in an external data source with a PGX graph."""

    _java_class = 'oracle.pgx.api.Synchronizer'

    def __init__(self, java_synchronizer) -> None:
        self._synchronizer = java_synchronizer

    def apply(self):
        """Apply the changes to the underlying PGX graph."""
        raise NotImplementedError()

    def fetch(self):
        """Fetch the changes from the external data source. You can call this multiple times
        to accumulate deltas. The deltas reset once you call `apply()`."""
        raise NotImplementedError()

    def get_graph_delta(self):
        """Get the description of the delta between current snapshot and the fetched changes.
        Can be used to make a decision for when to apply the delta."""
        raise NotImplementedError()

    def sync(self) -> "PgxGraph":
        """Synchronize changes from the external data source and return the new snapshot
        of the graph with the fetched changes applied."""
        from pypgx.api._pgx_graph import PgxGraph

        return PgxGraph(self, java_handler(self._synchronizer.sync, []))


class FlashbackSynchronizer(Synchronizer):
    """Synchronizes a PGX graph with an Oracle Database using Flashback queries."""

    _java_class = 'oracle.pgx.api.FlashbackSynchronizer'

    def __init__(self, java_flashback_synchronizer) -> None:
        self._flashback_synchronizer = java_flashback_synchronizer

    def apply(self):
        """Apply the changes to the underlying PGX graph."""
        return java_handler(self._flashback_synchronizer.apply, [])

    def fetch(self):
        """Fetch the changes from the external data source. You can call this multiple times
        to accumulate deltas. The deltas reset once you call `apply()`."""
        return java_handler(self._flashback_synchronizer.fetch, [])

    def get_graph_delta(self):
        """Synchronize changes from the external data source and return the new snapshot
        of the graph with the fetched changes applied."""
        return java_handler(self._flashback_synchronizer.getGraphDelta, [])
