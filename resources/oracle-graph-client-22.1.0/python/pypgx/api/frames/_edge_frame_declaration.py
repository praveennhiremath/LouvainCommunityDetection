#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from pypgx.api.frames._pgx_frame import PgxFrame
from pypgx._utils.error_messages import ARG_MUST_BE


class EdgeFrameDeclaration:
    """A class containing the necessary information to create an edge provider"""

    _java_class = 'oracle.pgx.api.frames.internal.EdgeFrameDeclaration'

    def __init__(
        self,
        provider_name: str,
        source_provider: str,
        destination_provider: str,
        frame: PgxFrame,
        source_vertex_column: str = "src",
        destination_vertex_column: str = "dst",
    ) -> None:
        if not isinstance(provider_name, str):
            raise TypeError(ARG_MUST_BE.format(arg='provider_name', type=str))
        if not isinstance(source_provider, str):
            raise TypeError(ARG_MUST_BE.format(arg='source_provider', type=str))
        if not isinstance(destination_provider, str):
            raise TypeError(ARG_MUST_BE.format(arg='destination_provider', type=str))
        if not isinstance(frame, PgxFrame):
            raise TypeError(ARG_MUST_BE.format(arg='frame', type=PgxFrame))
        if not isinstance(source_vertex_column, str):
            raise TypeError(ARG_MUST_BE.format(arg='source_vertex_column', type=str))
        if not isinstance(destination_vertex_column, str):
            raise TypeError(ARG_MUST_BE.format(arg='destination_vertex_column', type=str))
        self.provider_name = provider_name
        self.source_provider = source_provider
        self.destination_provider = destination_provider
        self.frame = frame
        self.source_vertex_column = source_vertex_column
        self.destination_vertex_column = destination_vertex_column
