#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from jnius import autoclass
from pypgx._utils.error_messages import INVALID_OPTION

JavaDataTypes = autoclass("oracle.pgx.api.frames.schema.datatypes.DataTypes")
vector_types = ('INTEGER_TYPE', 'LONG_TYPE', 'DOUBLE_TYPE', 'FLOAT_TYPE')


class VectorType:
    _java_class = 'oracle.pgx.api.frames.schema.datatypes.collection.VectorType'

    def __init__(self, componenttype: str, dimension: int) -> None:
        self.componenttype = componenttype
        self.dimension = dimension
        self._vector = self.create_vector(componenttype, dimension)

    def create_vector(self, componenttype, dimension):
        if componenttype == "INTEGER_TYPE":
            c = JavaDataTypes.INTEGER_TYPE
            return JavaDataTypes.vector(c, dimension)
        elif componenttype == "LONG_TYPE":
            c = JavaDataTypes.LONG_TYPE
            return JavaDataTypes.vector(c, dimension)
        elif componenttype == "DOUBLE_TYPE":
            c = JavaDataTypes.DOUBLE_TYPE
            return JavaDataTypes.vector(c, dimension)
        elif componenttype == "FLOAT_TYPE":
            c = JavaDataTypes.FLOAT_TYPE
            return JavaDataTypes.vector(c, dimension)
        else:
            raise ValueError(INVALID_OPTION.format(var='data_type', opts=list(vector_types)))


class DataTypes:
    _java_class = 'oracle.pgx.api.frames.schema.datatypes.DataTypes'

    @staticmethod
    def vector(componenttype: str, dimension: int) -> VectorType:
        return VectorType(componenttype, dimension)

    _data_type_instances = {
        'BOOLEAN_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.BooleanType")(),
        'STRING_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.StringType")(),
        'EDGE_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.graph.EdgeType")(),
        'VERTEX_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.graph.VertexType")(),
        'FLOAT_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.numeric.FloatType")(),
        'DOUBLE_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.numeric.DoubleType")(),
        'INTEGER_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.numeric.IntegerType")(),
        'LONG_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.numeric.LongType")(),
        'POINT2D_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.spatial.Point2dType")(),
        'LOCAL_DATE_TYPE': autoclass(
            "oracle.pgx.api.frames.schema.datatypes.temporal.LocalDateType"
        )(),
        'TIMESTAMP_TYPE': autoclass(
            "oracle.pgx.api.frames.schema.datatypes.temporal.TimestampType"
        )(),
        'TIMESTAMP_WITH_TIMEZONE_TYPE': autoclass(
            "oracle.pgx.api.frames.schema.datatypes.temporal.TimestampWithTimezoneType"
        )(),
        'TIME_TYPE': autoclass("oracle.pgx.api.frames.schema.datatypes.temporal.TimeType")(),
        'TIME_WITH_TIMEZONE_TYPE': autoclass(
            "oracle.pgx.api.frames.schema.datatypes.temporal.TimeWithTimezoneType"
        )(),
    }
