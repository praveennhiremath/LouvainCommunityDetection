#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass
from pypgx._utils.pgx_types import authorization_types

_PgxRoleClass = autoclass('oracle.pgx.common.auth.PgxRole')
_PgxUserClass = autoclass('oracle.pgx.common.auth.PgxUser')


class PermissionEntity:
    """Class representing an entity which can receive permissions."""

    _java_class = 'oracle.pgx.common.auth.PermissionEntity'

    def __init__(self, name: str, authorization_type, java_permission_entity_class) -> None:
        self._permission_entity = java_permission_entity_class(name)
        self.name = name
        self.type = authorization_type

    def get_name(self) -> str:
        """Get the entity name.

        :return: the entity name
        :rtype: str
        """
        return self.name

    def get_type(self):
        """Get the authorization type.

        This indicates if the entity is a user or a role.

        :return: the authorization type
        """
        # TODO: This returns a java object, which is probably not what we want.
        return self.type


class PgxUser(PermissionEntity):
    """Class representing a user for the purposes of permission management."""

    _java_class = 'oracle.pgx.common.auth.PgxUser'

    def __init__(self, name: str) -> None:
        super().__init__(name, authorization_types["user"], _PgxUserClass)


class PgxRole(PermissionEntity):
    """Class representing a role for the purposes of permission management."""

    _java_class = 'oracle.pgx.common.auth.PgxRole'

    def __init__(self, name: str) -> None:
        super().__init__(name, authorization_types["role"], _PgxRoleClass)
