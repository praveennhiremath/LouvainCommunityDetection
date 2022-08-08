#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass

from pypgx.api._pgx_session import PgxSession
from pypgx.api._server_instance import ServerInstance
from pypgx._utils.error_handling import java_handler, PgxError
from pypgx._utils.error_messages import ARG_MUST_BE
from typing import Optional


class Pgx:
    """Main entry point for PGX applications."""

    _java_class = 'oracle.pgx.api.Pgx'

    def __init__(self, java_pgx_class) -> None:
        self._pgx = java_pgx_class
        self.default_url = java_pgx_class.getDefaultUrl()

    def set_default_url(self, url: str) -> None:
        """Set the default base URL used by invocations of get_instance().

        The new default URL affects sub-sequent calls of getInstance().

        :param url: New URL
        """
        if not isinstance(url, str):
            raise TypeError(ARG_MUST_BE.format(arg='url', type=str.__name__))

        java_handler(self._pgx.setDefaultUrl, [url])
        self.default_url = url

    def get_instance(
        self, base_url: Optional[str] = None, token: Optional[str] = None
    ) -> ServerInstance:
        """Get a handle to a PGX instance.

        :param base_url: The base URL in the format host [ : port][ /path]
            of the PGX server REST end-point.
            If `base_url` is None, the default will
            be used which points to embedded PGX instance.
        :param token: The access token
        """
        from pypgx import PACKAGED_EMBEDDED_DIST

        if token is not None:
            args = [base_url, token]
        elif base_url is not None:
            args = [base_url]
        else:
            args = []

        try:
            java_server_instance = java_handler(self._pgx.getInstance, args)
        except RuntimeError as exc:
            # Normally, users should always provide a base URL if they are not using an embedded
            # distribution. We still try the call to getInstance because in some extraordinary
            # circumstances, a call with base_url set to None might succeed in a non-embedded
            # distribution (if the default URL has been changed, or the user has manually added the
            # missing JAR files).
            if not PACKAGED_EMBEDDED_DIST and base_url is None:
                message = (
                    "\n"
                    "This distribution of PyPGX does not support PGX embedded mode.\n"
                    "Provide a base URL to connect to a running PGX server."
                )
                raise PgxError(message) from exc

            # Unknown cause of exception, re-raise.
            raise

        return ServerInstance(java_server_instance)

    def create_session(self, source: Optional[str] = None, base_url: str = None) -> PgxSession:
        """Create and return a session.

        :param source: The session source string. Default value is "pgx_python".
        :param base_url: The base URL in the format host [ : port][ /path]
            of the PGX server REST end-point.
            If `base_url` is None, the default will
            be used which points to embedded PGX instance.
        """
        if source is None:
            source = "pgx_python"

        if base_url is None:
            java_session = java_handler(self._pgx.createSession, [source])
        else:
            java_session = java_handler(self._pgx.createSession, [base_url, source])

        return PgxSession(java_session)

    def __repr__(self) -> str:
        return "Pgx"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._pgx.equals(other._pgx)


def get_session(
    base_url: Optional[str] = None,
    session_name: Optional[str] = None,
    token: Optional[str] = None,
    idle_timeout: Optional[int] = None,
    task_timeout: Optional[int] = None,
    pgx_conf: Optional[str] = None,
    redirect_stdout: None = None,
    redirect_stderr: None = None,
) -> PgxSession:
    """Connect to a PGX server and return a PgxSession instance.

    :param base_url:  The URL of the running session. If None a session is created.
    :param session_name:  Session name. If None "python_pgx_client" is the default value.
    :param token:  The access token
    :param idle_timeout:  Number of seconds after which the shell session will time out.
    :param task_timeout:  Number of seconds after which tasks submitted by the shell session will
        time out.
    :param pgx_conf:  Path to PGX config file. If not set, PGX looks at default locations. See the
        'Engine and Runtime Configuration Guide' in the PGX documentation.
    :param redirect_stdout:  Deprecated.
    :param redirect_stderr:  Deprecated.
    :return: PgxSession
    """
    if session_name is None:
        session_name = "pypgx"

    java_pgx_class = autoclass('oracle.pgx.api.Pgx')
    pgx = Pgx(java_pgx_class)

    # create instance, start engine on pgx_conf, and return session
    server_instance = pgx.get_instance(base_url, token=token)
    if server_instance.is_embedded_instance:
        if pgx_conf is not None:
            try:
                if server_instance.is_engine_running():
                    server_instance.update_pgx_config(pgx_conf)
                else:
                    server_instance.start_engine(pgx_conf)
            except Exception:
                raise RuntimeError("Server instance failed to load pgx config")
    session = server_instance.create_session(session_name, idle_timeout, task_timeout, 'seconds')

    return session


def get_instance(base_url: Optional[str] = None, token: Optional[str] = None) -> ServerInstance:
    """Get a handle to a PGX instance.

    :param base_url: The base URL in the format host [ : port][ /path]
        of the PGX server REST end-point.
        If `base_url` is None, the default will
        be used which points to embedded PGX instance.
    :param token: The access token
    """
    java_pgx_class = autoclass('oracle.pgx.api.Pgx')
    pgx = Pgx(java_pgx_class)
    return pgx.get_instance(base_url, token=token)
