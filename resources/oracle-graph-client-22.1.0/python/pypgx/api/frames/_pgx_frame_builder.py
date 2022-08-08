#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from jnius import autoclass

from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import ARG_MUST_BE
from pypgx._utils.item_converter import convert_to_java_type
from pypgx.api.frames._pgx_frame import PgxFrame
from typing import Dict, Any


class PgxFrameBuilder:
    """A frame builder for constructing a :class:`PgxFrame`."""

    _java_class = 'oracle.pgx.api.frames.PgxFrameBuilder'

    def __init__(self, java_pgx_frame_builder) -> None:
        self._frame_builder = java_pgx_frame_builder

    def add_rows(self, column_data: Dict[str, Any]) -> "PgxFrameBuilder":
        """Add the data to the frame builder.

        :param column_data: the column data in a dictionary

        :return: self"""
        if not isinstance(column_data, dict):
            raise TypeError(ARG_MUST_BE.format(arg='column_data', type=dict))
        java_data = autoclass('java.util.HashMap')()
        for column_name in column_data:
            python_list = column_data[column_name]
            java_list = autoclass('java.util.ArrayList')()
            for x in python_list:
                java_list.add(convert_to_java_type(x))
            java_data.put(column_name, java_list)
        java_handler(self._frame_builder.addRows, [java_data])
        return self

    def build(self, frame_name: str) -> PgxFrame:
        """Build the frame with the given frame name.

        :param frame_name: the name of the frame to create

        :return: the newly frame created"""
        if not isinstance(frame_name, str):
            raise TypeError(ARG_MUST_BE.format(arg='frame_name', type=str))
        java_pgx_frame = java_handler(self._frame_builder.build, [frame_name])
        return PgxFrame(java_pgx_frame)
