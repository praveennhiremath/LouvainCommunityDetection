#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

import json
from jnius import autoclass

from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import INVALID_OPTION
from pypgx._utils.pgx_types import time_units
from pypgx._utils.item_converter import convert_to_java_map
from typing import Dict, Optional, Any, Union, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_session import PgxSession
    from pypgx.api._version_info import VersionInfo


class ServerInstance:
    """A PGX server instance."""

    _java_class = 'oracle.pgx.api.ServerInstance'

    def __init__(self, java_server_instance) -> None:
        self._server_instance = java_server_instance
        self.is_embedded_instance = java_server_instance.isEmbeddedInstance()
        self.username = java_server_instance.getUsername()
        self.base_url = java_server_instance.getBaseUrl()
        self.prefetch_size = java_server_instance.getPrefetchSize()
        self.upload_batch_size = java_server_instance.getUploadBatchSize()
        self.remote_future_timeout = java_server_instance.getRemoteFutureTimeout()
        self.client_server_interaction_mode = (
            java_server_instance.getClientServerInteractionMode().toString()
        )
        self.remote_future_pending_retry_interval = (
            java_server_instance.getRemoteFuturePendingRetryInterval()
        )
        self.version = java_server_instance.getVersion().toString()
        self.pgx_version = java_server_instance.getVersion()

    def create_session(
        self,
        source: str,
        idle_timeout: Optional[int] = None,
        task_timeout: Optional[int] = None,
        time_unit: str = 'milliseconds',
    ) -> "PgxSession":
        """
        :param source: A descriptive string identifying the client
        :param idle_timeout: If not null, tries to overwrite server default idle timeout
        :param task_timeout: If not null, tries to overwrite server default task timeout
        :param time_unit: Time unit of idleTimeout and taskTimeout
             ('days', 'hours', 'microseconds', 'milliseconds', 'minutes', 'nanoseconds', 'seconds')
        :returns: PgxSession
        """
        from pypgx.api._pgx_session import PgxSession

        if time_unit not in time_units:
            raise ValueError(INVALID_OPTION.format(var='time_unit', opts=list(time_units.keys())))

        # Convert timeouts to Long, as Pyjnius only converts to int, and createSession() doesn't
        # accept it
        long = autoclass('java.lang.Long')
        idle_timeout = long(idle_timeout) if idle_timeout is not None else None
        task_timeout = long(task_timeout) if task_timeout is not None else None
        time_unit = time_units[time_unit]
        session = java_handler(
            self._server_instance.createSession, [source, idle_timeout, task_timeout, time_unit]
        )
        return PgxSession(session)

    def get_session(self, session_id: str) -> "PgxSession":
        """Get a session by ID.

        :param session_id: Id of the session
        :returns: PgxSession
        """
        from pypgx.api._pgx_session import PgxSession

        session = java_handler(self._server_instance.getSession, [session_id])
        return PgxSession(session)

    def get_pgx_config(self) -> Dict[str, Any]:
        """Get the PGX config.

        :returns: Dict containing current config
        """
        config = self._server_instance.getPgxConfig()
        pgx_config = {}
        for k in config.keySet():
            key = k
            value = config.get(k)
            if not isinstance(key, str):
                tmp = getattr(key, "toString", None)
                key = tmp() if tmp is not None else str(key)
            if not isinstance(value, str):
                tmp = getattr(value, "toString", None)
                value = tmp() if tmp is not None else str(value)
            pgx_config[key] = value
        return pgx_config

    def get_server_state(self) -> Dict[str, Any]:
        """
        :return: Server state as a dict
        """
        server_state = self._server_instance.getServerState()
        return json.loads(server_state.toString())

    def get_version(self) -> "VersionInfo":
        """Get the PGX extended version of this instance.

        :returns: VersionInfo object
        """
        from pypgx.api._version_info import VersionInfo

        version = self._server_instance.getVersion()
        return VersionInfo(version)

    def kill_session(self, session_id: str) -> None:
        """Kill a session.

        :param session_id: Session id
        """
        java_handler(self._server_instance.killSession, [session_id])

    def is_engine_running(self) -> bool:
        """Boolean of whether or not the engine is running"""
        return java_handler(self._server_instance.isEngineRunning, [])

    def start_engine(self, config: Optional[Union[str, Mapping[str, Any]]] = None) -> None:
        """Start the PGX engine.

        :param config: path to json file or dict-like containing the PGX config"""

        if config is None:
            java_handler(self._server_instance.startEngine, [])
        elif isinstance(config, str):
            java_handler(self._server_instance.startEngine, [config])
        else:
            java_handler(self._server_instance.startEngine, [convert_to_java_map(config)])

    def update_pgx_config(self, config: Union[str, Mapping[str, Any]]) -> None:
        """Replace the current PGX config with the given configuration.

        This only affects static permissions (i.e. non-graph) and redaction rules for pre-loaded
        graphs. Existing permissions on graphs and frames will not be changed.

        :param config: path to json file or dict-like PGX config containing the new authorization
            config
        """

        if isinstance(config, str):
            java_handler(self._server_instance.updatePgxConfig, [config])
        else:
            java_handler(self._server_instance.updatePgxConfig, [convert_to_java_map(config)])

    def shutdown_engine(self) -> None:
        """Force the engine to stop and clean up resources"""
        java_handler(self._server_instance.shutdownEngineNowIfRunning, [])

    def __repr__(self) -> str:
        if self.is_embedded_instance:
            return "{}(embedded: {}, version: {})".format(
                self.__class__.__name__, self.is_embedded_instance, self.pgx_version
            )
        else:
            return "{}(embedded: {}, base_url: {}, version: {})".format(
                self.__class__.__name__, self.is_embedded_instance, self.base_url, self.pgx_version
            )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._server_instance.equals(other._server_instance)
