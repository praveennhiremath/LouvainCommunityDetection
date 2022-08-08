#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import Optional

from pypgx._utils.error_handling import java_handler


class SessionContext:
    """A session context.

    Contains information describing a :class:`PgxSession` instance
    """

    _java_class = 'oracle.pgx.api.SessionContext'

    def __init__(self, java_session_context) -> None:
        self._session_context = java_session_context

    def get_session_id(self) -> str:
        """Get the session ID.

        :return: the session ID
        :rtype: str
        """
        return java_handler(self._session_context.getSessionId, [])

    def get_sticky_cookie_value(self) -> Optional[str]:
        """Get the value of the sticky session cookie.

        :return: the sticky cookie value
        :rtype: str
        """
        return java_handler(self._session_context.getStickyCookieValue, [])
