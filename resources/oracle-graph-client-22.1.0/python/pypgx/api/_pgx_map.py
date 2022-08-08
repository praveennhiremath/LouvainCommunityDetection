#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from itertools import islice
from typing import Any, Iterator, Optional, TYPE_CHECKING

from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler, java_caster
from pypgx._utils.item_converter import convert_to_java_type, convert_to_python_type

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class PgxMap(PgxContextManager):
    """A map is a collection of key-value pairs."""

    _java_class = 'oracle.pgx.api.PgxMap'

    def __init__(self, graph: Optional["PgxGraph"], java_map) -> None:
        self._map = java_map
        self.name = java_map.getName()
        self.key_type = java_map.getKeyType().toString()
        self.value_type = java_map.getValueType().toString()
        self.graph = graph
        self.session_id = java_handler(self._map.getSessionId, [])

    @property
    def size(self) -> int:
        """Map size."""
        return self._map.size()

    def put(self, key, value) -> None:
        """Set the value for a key in the map specified by the given name.

        :param key: Key of the entry
        :param value: New value
        """
        key = convert_to_java_type(key)
        key = self._wrap(key, self.key_type)
        value = convert_to_python_type(value, self.graph)
        value = self._wrap(value, self.value_type)
        java_caster(self._map.put, (key, self.key_type), (value, self.value_type))

    def remove(self, key) -> bool:
        """Remove the entry specified by the given key from the map with the given name.

        Returns true if the map did contain an entry with the given key, false otherwise.

        :param key: Key of the entry
        :returns: True if the map contained the key
        """
        key = convert_to_java_type(key)
        key = self._wrap(key, self.key_type)
        return java_caster(self._map.remove, (key, self.key_type))

    def get(self, key) -> Any:
        """Get the entry with the specified key.

        :param key: Key of the entry
        :returns: Value
        """
        key = convert_to_java_type(key)
        key = self._wrap(key, self.key_type)
        value = java_caster(self._map.get, (key, self.key_type))

        return convert_to_python_type(value, self.graph)

    def contains_key(self, key) -> bool:
        """
        :param key: Key of the entry
        """
        key = convert_to_java_type(key)
        key = self._wrap(key, self.key_type)
        return java_handler(self._map.containsKey, [key])

    def keys(self) -> list:
        """Return a key set."""
        return list(self)

    def entries(self) -> dict:
        """Return an entry set."""
        map_dict = {}
        for key in self:
            map_dict[key] = self.get(key)
        return map_dict

    def _wrap(self, item, item_type: str):
        """
        :param item:
        :param item_type:
        """
        if isinstance(item, (int, str)) and item_type == 'vertex':
            return java_handler(self.graph._graph.getVertex, [item])
        elif isinstance(item, (int, str)) and item_type == 'edge':
            return java_handler(self.graph._graph.getEdge, [item])
        else:
            return item

    def destroy(self) -> None:
        """Destroy this map."""
        java_handler(self._map.destroy, [])

    def __iter__(self) -> Iterator:
        it = self._map.keys().iterator()
        return (convert_to_python_type(item, self.graph) for item in islice(it, 0, self.size))

    def __getitem__(self, key) -> Any:
        return self.get(key)

    def __setitem__(self, key, value) -> None:
        self.put(key, value)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "{}(name: {}, {}: {}, key_type: {}, value_type: {}, size: {})".format(
            self.__class__.__name__,
            self.name,
            'session' if self.graph is None else 'graph',
            self.session_id if self.graph is None else self.graph.name,
            self.key_type,
            self.value_type,
            self.size,
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        if self.graph is None:
            return hash((str(self), str(self.session_id)))
        else:
            return hash((str(self), str(self.graph.name)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._map.equals(other._map)
