#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

"""Tools for handling deprecations."""

import types
import warnings
from typing import Any, List, Mapping, NamedTuple


class DeprecatedAttribute(NamedTuple):
    """A module attribute that has been deprecated."""

    new_name: str
    since_version: str


class RemovedAttribute(NamedTuple):
    """A module attribute that has been removed."""

    reason: str
    since_version: str


class Module(types.ModuleType):
    """A module that emits warnings for access on deprecated attributes."""

    # An instance of this class should be added to sys.modules to enable deprecation warnings on
    # deprecated attributes.
    #
    # In Python 3.7+, a module-level __getattr__() is a better approach than using this class:
    # https://www.python.org/dev/peps/pep-0562/

    def __init__(
        self,
        wrapped_module: types.ModuleType,
        deprecated_attributes: Mapping[str, DeprecatedAttribute],
        removed_attributes: Mapping[str, RemovedAttribute] = None,
    ):
        if removed_attributes is None:
            removed_attributes = {}

        # Initialize __dict__.
        super().__init__(wrapped_module.__name__, wrapped_module.__doc__)

        # Circumvent __setattr__.
        self.__dict__['_wrapped_module'] = wrapped_module
        self.__dict__['_deprecated_attributes'] = deprecated_attributes
        self.__dict__['_removed_attributes'] = removed_attributes

    def __getattr__(self, attr: str) -> Any:
        if attr in self._removed_attributes and not hasattr(self._wrapped_module, attr):
            removal = self._removed_attributes[attr]
            msg = "{}.{} was removed in PyPGX {} (reason: {})".format(
                self.__name__,
                attr,
                removal.since_version,
                removal.reason,
            )
            raise AttributeError(msg)

        if attr in self._deprecated_attributes:
            deprecation = self._deprecated_attributes[attr]
            msg = "accessing {}.{} is deprecated since PyPGX {}, use {} instead".format(
                self.__name__,
                attr,
                deprecation.since_version,
                deprecation.new_name,
            )
            warnings.warn(msg, DeprecationWarning)
            # For future reference: the default visibility of DeprecationWarning changed in
            # Python 3.7.
            # https://www.python.org/dev/peps/pep-0565/
        return getattr(self._wrapped_module, attr)

    def __setattr__(self, attr: str, value: object) -> None:
        setattr(self._wrapped_module, attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._wrapped_module, attr)

    def __dir__(self) -> List[str]:
        return dir(self._wrapped_module)
