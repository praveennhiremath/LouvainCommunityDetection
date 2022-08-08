#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

ARG_MUST_BE = "'{arg}' must be: {type}."
ARG_MUST_BE_REASON = "'{arg}' must be: {value}. {cause}"
INVALID_OPTION = "Invalid '{var}'. Valid options are: {opts}"
NO_SUCH_FILE = "No such file: '{file}'"
VALID_PATH_OR_LIST_OF_PATHS = "'{path}' must be a valid path, or a list of valid paths."
VALID_PATH_LISTS = "'{path1}' and '{path2}' must be lists of valid paths."
VALID_CONFIG_ARG = (
    "'{config}' must be a valid file path, dict, config string or '{config_type}' object."
)
MODEL_NOT_FITTED = "The model has not been fitted."
VERTEX_ID_OR_COLLECTION_OF_IDS = "'{var}' must be a vertex ID or a collection of vertex IDs."
VERTEX_ID_OR_PGXVERTEX = "'{var}' must be a vertex ID or a PgxVertex."
PROPERTY_NOT_FOUND = "Property '{prop}' not found."
INDEX_OUT_OF_BOUNDS = "'{idx}' must be an integer: 0 <= '{idx}' <= {max_idx}"
VALID_INTERVAL = (
    "'start': {start} and 'stop': {stop} must define a valid interval within the range: "
    "[0, {max_idx}]"
)
COMPARE_VECTOR = "vector comparison is not supported."
INVALID_TYPE_OR_ITERABLE_TYPE = (
    "'{var}' must be of type: '{type}', or an iterable with len: {size} and type: '{type}'"
)
WRONG_NUMBER_OF_ARGS = "expected {expected} arguments but received {received}"
UNHASHABLE_TYPE = "Unhashable type: '{type_name}'"
