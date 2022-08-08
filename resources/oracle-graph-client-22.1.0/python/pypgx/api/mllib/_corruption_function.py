#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#


class CorruptionFunction:
    """Abstract Corruption Function which generate the corrupted subgraph for DGI"""

    _java_class = "oracle.pgx.config.mllib.corruption.CorruptionFunction"

    def __init__(self, java_corruption_function) -> None:
        self._corruption_function = java_corruption_function


class PermutationCorruption(CorruptionFunction):
    """Permutation Function which shuffle the nodes to generate the corrupted subgraph for DGI"""

    _java_class = "oracle.pgx.config.mllib.corruption.PermutationCorruption"

    def __init__(self, java_permutation_corruption) -> None:
        super().__init__(java_permutation_corruption)

    def __repr__(self) -> str:
        return "%s" % (self.__class__.__name__)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._corruption_function.equals(other._corruption_function)
