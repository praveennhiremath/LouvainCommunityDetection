#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import Any, TYPE_CHECKING

from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler, java_caster
from pypgx._utils.item_converter import convert_to_java_type, convert_to_python_type

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph


class Scalar(PgxContextManager):
    """A scalar value."""

    _java_class = 'oracle.pgx.api.Scalar'

    def __init__(self, graph: "PgxGraph", java_scalar) -> None:
        self._scalar = java_scalar
        self.name = java_scalar.getName()
        self.type = java_scalar.getType().toString()
        self.graph = graph

    def set(self, value) -> None:
        """Set the scalar value.

        :param value: Value to be assigned
        """
        value = convert_to_java_type(value)
        value = self._wrap(value, self.type)
        java_caster(self._scalar.set, (value, self.type))

    def get(self) -> Any:
        """Get scalar value."""
        value = java_handler(self._scalar.get, [])
        return convert_to_python_type(value, self.graph)

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
        """Free resources on the server taken up by this Scalar."""
        java_handler(self._scalar.destroy, [])

    def __repr__(self) -> str:
        return "{}(value: {}, name: {}, type: {}, graph: {})".format(
            self.__class__.__name__, self.get(), self.name, self.type, self.graph.name
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((str(self), str(self.graph.name)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._scalar.equals(other._scalar)
