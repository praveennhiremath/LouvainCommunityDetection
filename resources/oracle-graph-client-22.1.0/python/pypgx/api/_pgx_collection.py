#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import collections.abc
from itertools import islice

from jnius import autoclass

from pypgx.api._pgx_entity import PgxEdge, PgxVertex
from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import UNHASHABLE_TYPE
from pypgx._utils.item_converter import convert_to_java_type
from pypgx.api._pgx_id import PgxId
from typing import Iterable, Iterator, List, Optional, Union, TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph
    from pypgx.api._pgx_map import PgxMap


class PgxCollection(PgxContextManager):
    """Superclass for Pgx collections."""

    _java_class = 'oracle.pgx.api.PgxCollection'

    def __init__(self, graph: Optional["PgxGraph"], java_collection) -> None:
        self._collection = java_collection
        self.name = java_collection.getName()
        self.content_type = java_collection.getContentType().toString()
        self.collection_type = java_collection.getCollectionType().toString()
        if java_collection.getIdType():
            self.id_type = java_collection.getIdType().toString()
        else:
            self.id_type = None
        self.is_mutable = java_collection.isMutable()
        self.graph = graph

    def clear(self) -> None:
        """Clear an existing collection.

        :return: None
        """
        return java_handler(self._collection.clear, [])

    def clone(self, name: Optional[str] = None) -> "PgxCollection":
        """Clone and rename existing collection.

        :param name:  New name of the collection. If none, the old name is not changed.
        """
        cloned_coll = java_handler(self._collection.clone, [name])
        return self.__class__(self.graph, cloned_coll)

    def to_mutable(self, name: Optional[str] = None) -> "PgxCollection":
        """Create a mutable copy of an existing collection.

        :param name: New name of the collection. If none, the old name is not changed.
        """
        mutable_coll = java_handler(self._collection.toMutable, [name])
        return self.__class__(self.graph, mutable_coll)

    @property
    def size(self) -> int:
        """Get the number of elements in this collection."""
        return self._collection.size()

    def _create_ids_array(self, collection, entity_type):

        array_ids = autoclass('java.util.ArrayList')()
        for item in collection:
            array_ids.add(item.id if isinstance(item, entity_type) else item)
        return array_ids

    def destroy(self) -> None:
        """Request destruction of this object.

        After this method returns, the behavior of any method of this class becomes undefined.

        :return: None
        """
        java_handler(self._collection.destroy, [])

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "{}(name: {}, graph: {}, size: {})".format(
            self.__class__.__name__, self.name, self.graph.name, self.size
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((str(self), str(self.graph.name)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._collection.equals(other._collection)

    def get_id(self) -> str:
        """Return the string representation of an internal identifier for this collection.
        Only meant for internal usage.

        :returns: a string representation of the internal identifier of this collection
        """
        java_pgx_id = java_handler(self._collection.getId, [])
        return java_handler(java_pgx_id.toString, [])

    def get_pgx_id(self) -> PgxId:
        """Return an internal identifier for this collection.
        Only meant for internal usage.

        :returns: the internal identifier of this collection
        """
        java_pgx_id = java_handler(self._collection.getId, [])
        return PgxId(java_pgx_id)

    def close(self) -> None:
        """Request destruction of this object. After this method returns, the behavior of any method
        of this class becomes undefined.
        """
        java_handler(self._collection.close, [])

    def add_all_elements(self, source: Iterable[Union[PgxEdge, PgxVertex]]) -> None:
        """Add elements to an existing collection.

        :param source: Elements to add
        """
        java_elements = autoclass('java.util.ArrayList')()
        for element in source:
            java_elements.add(convert_to_java_type(element))
        java_handler(self._collection.addAllElements, [java_elements])

    def remove_all_elements(self, source: Iterable[Union[PgxEdge, PgxVertex]]) -> None:
        """Remove elements from an existing collection.

        :param source: Elements to remove
        """
        java_elements = autoclass('java.util.ArrayList')()
        for element in source:
            java_elements.add(convert_to_java_type(element))
        java_handler(self._collection.removeAllElements, [java_elements])

    def contains(self, element):
        # noqa: D102
        raise NotImplementedError

    def add_all(self, elements):
        # noqa: D102
        raise NotImplementedError

    def remove(self, element):
        # noqa: D102
        raise NotImplementedError

    def remove_all(self, elements):
        # noqa: D102
        raise NotImplementedError


class VertexCollection(PgxCollection):
    """A collection of vertices."""

    _java_class = 'oracle.pgx.api.VertexCollection'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)

    def contains(self, v: Union[PgxVertex, int, str]) -> bool:
        """Check if the collection contains vertex v.

        :param v: PgxVertex object or id
        """
        if not isinstance(v, PgxVertex):
            v = self.graph.get_vertex(v)
        return java_handler(self._collection.contains, [v._vertex])

    def add(self, v: Union[PgxVertex, int, str, Iterable[Union[PgxVertex, int, str]]]) -> None:
        """Add one or multiple vertices to the collection.

        :param v: Vertex or vertex id. Can also be an iterable of vertices/Vetrices ids
        """
        if isinstance(v, collections.abc.Iterable):
            return self.add_all(v)
        elif not isinstance(v, PgxVertex):
            v = self.graph.get_vertex(v)
        java_handler(self._collection.add, [v._vertex])

    def add_all(self, vertices: Iterable[Union[PgxVertex, int, str]]) -> None:
        """Add multiple vertices to the collection.

        :param vertices: Iterable of vertices/Vertices ids
        """
        vids = self._create_ids_array(vertices, PgxVertex)
        java_handler(self._collection.addAllById, [vids])

    def remove(self, v: Union[PgxVertex, int, str, Iterable[Union[PgxVertex, int, str]]]) -> None:
        """Remove one or multiple vertices from the collection.

        :param v: Vertex or vertex id. Can also be an iterable of vertices/Vetrices ids.
        """
        if isinstance(v, collections.abc.Iterable):
            self.remove_all(v)
        else:
            if not isinstance(v, PgxVertex):
                v = self.graph.get_vertex(v)
            java_handler(self._collection.remove, [v._vertex])

    def remove_all(self, vertices: Iterable[Union[PgxVertex, int, str]]):
        """Remove multiple vertices from the collection.

        :param vertices: Iterable of vertices/Vetrices ids
        """
        vids = self._create_ids_array(vertices, PgxVertex)
        java_handler(self._collection.removeAllById, [vids])

    def __iter__(self) -> Iterator[PgxVertex]:
        it = self._collection.iterator()
        return (PgxVertex(self.graph, item) for item in islice(it, 0, self.size))

    def __getitem__(self, idx: Union[slice, int]) -> Union[List[PgxVertex], PgxVertex]:
        it = self._collection.iterator()
        if isinstance(idx, slice):
            return list(
                PgxVertex(self.graph, item) for item in islice(it, idx.start, idx.stop, idx.step)
            )
        else:
            return list(PgxVertex(self.graph, item) for item in islice(it, idx, idx + 1))[0]

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class EdgeCollection(PgxCollection):
    """A collection of edges."""

    _java_class = 'oracle.pgx.api.EdgeCollection'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)

    def contains(self, e: Union[PgxEdge, int]) -> bool:
        """Check if the collection contains edge e.

        :param e: PgxEdge object or id:
        :returns: Boolean
        """
        if not isinstance(e, PgxEdge):
            e = self.graph.get_edge(e)
        return java_handler(self._collection.contains, [e._edge])

    def add(self, e: Union[PgxEdge, int, Iterable[Union[PgxEdge, int]]]):
        """Add one or multiple edges to the collection.

        :param e: Edge or edge id. Can also be an iterable of edge/edge ids.
        """
        if isinstance(e, collections.abc.Iterable):
            return self.add_all(e)
        elif not isinstance(e, PgxEdge):
            e = self.graph.get_edge(e)
        java_handler(self._collection.add, [e._edge])

    def add_all(self, edges: Iterable[Union[PgxEdge, int]]) -> None:
        """Add multiple vertices to the collection.

        :param edges: Iterable of edges/edges ids
        """
        eids = self._create_ids_array(edges, PgxEdge)
        java_handler(self._collection.addAllById, [eids])

    def remove(self, e: Union[PgxEdge, int, Iterable[Union[PgxEdge, int]]]):
        """Remove one or multiple edges from the collection.

        :param e: Edges or edges id. Can also be an iterable of edges/edges ids.
        """
        if isinstance(e, collections.abc.Iterable):
            return self.remove_all(e)
        elif not isinstance(e, PgxEdge):
            e = self.graph.get_edge(e)
        java_handler(self._collection.remove, [e._edge])

    def remove_all(self, edges: Iterable[Union[PgxEdge, int]]):
        """Remove multiple edges from the collection.

        :param edges: Iterable of edges/edges ids
        """
        eids = self._create_ids_array(edges, PgxEdge)
        java_handler(self._collection.removeAllById, [eids])

    def __iter__(self) -> Iterator[PgxEdge]:
        it = self._collection.iterator()
        return (PgxEdge(self.graph, item) for item in islice(it, 0, self.size))

    def __getitem__(self, idx: Union[slice, int]) -> Union[List[PgxEdge], PgxEdge]:
        it = self._collection.iterator()
        if isinstance(idx, slice):
            return list(
                PgxEdge(self.graph, item) for item in islice(it, idx.start, idx.stop, idx.step)
            )
        else:
            return list(PgxEdge(self.graph, item) for item in islice(it, idx, idx + 1))[0]

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class VertexSet(VertexCollection):
    """An unordered set of vertices (no duplicates)."""

    _java_class = 'oracle.pgx.api.VertexSet'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)

    def extract_top_k_from_map(self, pgx_map: "PgxMap", k: int) -> None:
        """Extract the top k keys from the given map and puts them into this collection.

        :param pgx_map: the map to extract the keys from
        :param k:   how many keys to extract
        """
        java_pgx_map = pgx_map._map
        java_handler(self._collection.extractTopKFromMap, [java_pgx_map, k])


class VertexSequence(VertexCollection):
    """An ordered sequence of vertices which may contain duplicates."""

    _java_class = 'oracle.pgx.api.VertexSequence'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)


class EdgeSet(EdgeCollection):
    """An unordered set of edges (no duplicates)."""

    _java_class = 'oracle.pgx.api.EdgeSet'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)


class EdgeSequence(EdgeCollection):
    """An ordered sequence of edges which may contain duplicates."""

    _java_class = 'oracle.pgx.api.EdgeSequence'

    def __init__(self, graph: "PgxGraph", java_collection) -> None:
        super().__init__(graph, java_collection)


class ScalarCollection(PgxCollection):
    """A collection of scalars."""

    _java_class = 'oracle.pgx.api.ScalarCollection'

    def __init__(self, java_scalar_collection) -> None:
        super().__init__(None, java_scalar_collection)


class ScalarSequence(ScalarCollection):
    """An ordered sequence of scalars which may contain duplicates."""

    _java_class = 'oracle.pgx.api.ScalarSequence'


class ScalarSet(ScalarCollection):
    """An unordered set of scalars that does not contain duplicates."""

    _java_class = 'oracle.pgx.api.ScalarSet'
