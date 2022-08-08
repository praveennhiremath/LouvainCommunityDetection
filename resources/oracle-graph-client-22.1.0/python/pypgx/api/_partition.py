#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from itertools import islice
from typing import Union, Any, Iterator

from pypgx.api._pgx_collection import VertexSet
from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import INDEX_OUT_OF_BOUNDS


class PgxPartition(PgxContextManager):
    """A vertex partition of a graph. Each partition is a set of vertices."""

    _java_class = 'oracle.pgx.api.Partition'

    def __init__(self, graph, java_partition, property) -> None:
        self._partition = java_partition
        self.size = java_partition.size()
        self.property = property
        self.graph = graph

    def get_partition_by_vertex(self, v: Union[PgxVertex, int, str]) -> VertexSet:
        """Get the partition a particular vertex belongs to.

        :param v: The vertex
        :returns: The set of vertices representing the partition the given vertex belongs to
        """
        if isinstance(v, (int, str)):
            vertex = self.graph.get_vertex(v)
        else:
            vertex = v
        java_collection = java_handler(self._partition.getPartitionByVertex, [vertex._vertex])
        return VertexSet(self.graph, java_collection)

    def get_partition_by_index(self, idx: int) -> VertexSet:
        """Get a partition by index.

        :param idx: The index. Must be between 0 and size() - 1.
        :returns: The set of vertices representing the partition
        """
        if idx >= self.size:
            raise RuntimeError(INDEX_OUT_OF_BOUNDS.format(idx='idx', max_idx=self.size - 1))
        java_collection = java_handler(self._partition.getPartitionByIndex, [idx])
        return VertexSet(self.graph, java_collection)

    def get_partition_index_of_vertex(self, v: Union[PgxVertex, int, str]) -> Any:
        """Get a partition by index.

        :param v: The index. Must be between 0 and size() - 1.
        :returns: The set of vertices representing the partition
        """
        if isinstance(v, PgxVertex):
            v = v.id
        return java_handler(self._partition.getPartitionIndexOfVertex, [v])

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[VertexSet]:
        it = java_handler(self._partition.iterator, [])
        return (VertexSet(self.graph, item) for item in islice(it, 0, None))

    def __getitem__(self, idx: int) -> Union[list, VertexSet]:
        if isinstance(idx, slice):
            return list(item for item in islice(self, idx.start, idx.stop, idx.step))
        else:
            return self.get_partition_by_index(idx)

    def destroy(self) -> None:
        """Destroy the partition object."""
        java_handler(self._partition.destroy, [])

    def __repr__(self) -> str:
        return "{}(graph: {}, components: {})".format(
            self.__class__.__name__, self.graph.name, self.size
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((str(self), str(self.graph.name), str(self.size)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._partition.equals(other._partition)
