#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass


class MutationStrategy:
    """Represents a strategy for mutating a `PgxGraph`."""

    _java_class = autoclass('oracle.pgx.common.mutations.MutationStrategy')

    def __init__(self, java_mutation_strategy):
        self._mutation_strategy = java_mutation_strategy
