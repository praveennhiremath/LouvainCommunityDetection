#
# Copyright (C) 2013 - 2022, Oracle and/or its affiliates. All rights reserved.
# ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
#

"""Allows to connect to a graph server"""

from .graph_server import get_embedded_instance
from .graph_server import get_instance
from .graph_server import reauthenticate
from .graph_server import generate_token

__all__ = ["get_embedded_instance", "get_instance", "reauthenticate", "generate_token"]