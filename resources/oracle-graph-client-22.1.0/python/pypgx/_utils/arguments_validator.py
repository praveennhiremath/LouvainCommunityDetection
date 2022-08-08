#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

# Import directly from private modules to avoid circular imports which are a problem in Python 3.6.
from pypgx.api._pgx_map import PgxMap
from pypgx.api._property import VertexProperty, EdgeProperty
from typing import Any, Dict


def validate_arguments(arguments: Dict[str, Any], algorithm_metadata: Dict[str, Any]) -> None:

    all_args = {**algorithm_metadata['in_arguments'], **algorithm_metadata['out_arguments']}
    arg_mismatch = False

    for arg in all_args:
        expected_type = all_args[arg]["type"]
        curr_arg = arguments[arg]

        if not isinstance(curr_arg, tuple(expected_type)):
            arg_mismatch = True

        if "subtype" in all_args[arg] and not arg_mismatch:

            if isinstance(curr_arg, (VertexProperty, EdgeProperty)):
                prop_dim = all_args[arg]["subtype"][curr_arg.__class__]["dimension"]
                prop_type = all_args[arg]['subtype'][curr_arg.__class__]["type"]
                if isinstance(prop_dim, str):
                    all_args[arg]["subtype"][curr_arg.__class__]["dimension"] = arguments[prop_dim]
                arg_mismatch = prop_dim != curr_arg.dimension or prop_type != curr_arg.type

            if isinstance(curr_arg, PgxMap):
                key_type = all_args[arg]["subtype"][PgxMap]["key_type"]
                val_type = all_args[arg]['subtype'][PgxMap]["value_type"]
                arg_mismatch = key_type != curr_arg.key_type or val_type != curr_arg.value_type

        if arg_mismatch:
            msg = "Argument '" + arg + "' must be any of the following types:\n"
            for etype in expected_type:
                msg += "\t\t* '" + etype.__name__ + "'"
                if "subtype" in all_args[arg] and etype in all_args[arg]["subtype"]:
                    msg += " with "
                    sub_specs = all_args[arg]["subtype"][etype]
                    for spec in sub_specs:
                        msg += spec + ":'" + str(sub_specs[spec]) + "' and "
                    msg = msg[:-4]
                msg += "\n"

            raise TypeError(msg)
