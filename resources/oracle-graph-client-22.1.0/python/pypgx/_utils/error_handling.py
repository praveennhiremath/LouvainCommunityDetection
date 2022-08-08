#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

"""Tools for handling exceptions that happen in Java."""

from jnius import cast, JavaClass
from jnius.jnius import JavaException
from typing import Any, Callable, Optional, Tuple, Iterable


def java_caster(method: Callable, *arg_type_pairs: Tuple[Any, Optional[str]]) -> Any:
    """Cast arguments and then call a Java method.

    Each element of `arg_type_pairs` is a pair (arg, type).  `type` should be a
    string or None.  If `type` is a string, it is used as a key into java_types.
    If `type` is None, no cast is done."""

    # Import here to prevent a circular import.
    from pypgx._utils.pgx_types import java_types

    casted_args = []
    for arg, arg_type in arg_type_pairs:
        if arg_type is None or arg_type == 'vertex' or arg_type == 'edge':
            casted_arg = arg
        else:
            java_type = java_types[arg_type]
            casted_arg = _cast_to_java(arg, java_type)
        casted_args.append(casted_arg)
    return java_handler(method, casted_args)


def _cast_to_java(value, java_type):
    """Cast `value` to Java type.

    Raise a RuntimeError if something goes wrong on the Java side."""

    if isinstance(value, JavaClass):
        return java_handler(cast, [java_type, value])
    return java_handler(java_type, [value])


def java_handler(
    callable: Callable,
    arguments: Iterable[Any],
    expected_pgx_exception: Optional[str] = None,
) -> Any:
    """Call `callable` with the given arguments.

    Raise a RuntimeError if something goes wrong on the Java side.
    If java exception matches expected_pgx_exception, raise PgxError instead.

    :param callable: Java callable
    :param arguments: list of arguments for callable
    :param expected_pgx_exception: string representing java exception class.
        E.g. "java.lang.UnsupportedOperationException"
    """

    try:
        return callable(*arguments)
    except JavaException as e:
        message = str(e)
        stacktrace = getattr(e, "stacktrace", None)
        if stacktrace is not None:
            message += "\nJava stack trace:\n"
            stacktrace_java = [
                ("    " + line if "Caused by" not in line else line) for line in stacktrace
            ]
            message += "\n".join(stacktrace_java)
        if expected_pgx_exception is not None and expected_pgx_exception == e.classname:
            raise PgxError(message) from None
        raise RuntimeError(message) from None


class PgxError(Exception):
    """An error representing exceptions from PGX."""

    pass
