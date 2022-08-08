#
# Copyright (C) 2013 - 2022, Oracle and/or its affiliates. All rights reserved.
# ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
#

"""Python PGX client.

Some core functions are provided directly at the top level of the package.

The documented behaviour of the PyPGX API is stable between versions. Other parts of
the API, in particular attributes whose name starts with an underscore, are considered
internal implementation details and may change between versions.
"""

import os as _os
import sys as _sys

import jnius_config

# Jnius config should be done before importing jnius. So we do this now, before other imports.
if not jnius_config.vm_running:
    from ._utils import env_vars as _env_vars
    jnius_config.set_classpath(_env_vars.OPG_CLASSPATH)

from pypgx._utils.error_handling import PgxError
from pypgx._utils.loglevel import setloglevel
from pypgx.api._pgx import get_instance, get_session
from pypgx.api.filters import EdgeFilter, VertexFilter, GraphFilter

from pypgx._utils.deprecation import (
    DeprecatedAttribute as _DeprecatedAttribute,
    Module as _Module,
    RemovedAttribute as _RemovedAttribute,
)

_deprecations = {
    "EdgeFilter": _DeprecatedAttribute("pypgx.api.filters.EdgeFilter", since_version="21.4"),
    "VertexFilter": _DeprecatedAttribute("pypgx.api.filters.VertexFilter", since_version="21.4"),
    "GraphFilter": _DeprecatedAttribute("pypgx.api.filters.GraphFilter", since_version="21.4"),
}
_removals = {
    "utils": _RemovedAttribute(reason="non-public API", since_version="21.4"),
    "common": _RemovedAttribute(
        reason="public classes have been moved to the pypgx.api package",
        since_version="21.4"
    ),
}
_sys.modules[__name__] = _Module(_sys.modules[__name__], _deprecations, _removals)

__all__ = ["get_instance", "get_session", "setloglevel", "PgxError"]