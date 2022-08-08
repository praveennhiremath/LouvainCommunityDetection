#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from jnius import autoclass

from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler


class PgxId(PgxContextManager):
    """Internal identifier class to uniquely identify PGX objects.

    Do not create PgxId objects yourself but use appropriate generator methods.
    """

    _java_class = 'oracle.pgx.common.PgxId'

    def __init__(self, java_pgx_id) -> None:
        self._id = java_pgx_id

    def __str__(self) -> str:
        return java_handler(self._id.toString, [])

    def __eq__(self, other: object) -> bool:
        if other is None or not isinstance(other, PgxId):
            return False
        return java_handler(self._id.equals, [other._id])

    def __hash__(self) -> int:
        return java_handler(self._id.hashCode, [])

    @staticmethod
    def from_string(value: str) -> "PgxId":
        """Parse `value` as a UUID and generate a PgxId object from it.

        If `value` does not represent a valid UUID, `IllegalArgumentException`
        is thrown by the JVM.

        :param value: input UUID
        :type value: str

        :rtype: PgxId
        """
        java_pgx_id = autoclass('oracle.pgx.common.PgxId').fromString(value)
        return PgxId(java_pgx_id)

    @staticmethod
    def generate_from_string(value: str) -> "PgxId":
        """Generate a pseudo-random PgxId starting from value.

        :param value: the starting value
        :type value: str

        :rtype: PgxId
        """
        java_pgx_id = autoclass('oracle.pgx.common.PgxId').generateFromString(value)
        return PgxId(java_pgx_id)
