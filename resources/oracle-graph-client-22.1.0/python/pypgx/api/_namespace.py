#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass

_JavaNamespace = autoclass('oracle.pgx.api.Namespace')


class Namespace:
    """Represents a namespace for objects (e.g. graphs, properties) in PGX.

    .. note:: This class is just a thin wrapper and does not check if the input
       is actually a java namespace.
    """

    _java_class = 'oracle.pgx.api.Namespace'

    def __init__(self, java_namespace) -> None:
        self._namespace = java_namespace

    def get_java_namespace(self):
        """Get the java namespace object.

        :rtype: oracle.pgx.api.Namespace
        """
        return self._namespace


NAMESPACE_PRIVATE = Namespace(_JavaNamespace.PRIVATE)
NAMESPACE_PUBLIC = Namespace(_JavaNamespace.PUBLIC)
