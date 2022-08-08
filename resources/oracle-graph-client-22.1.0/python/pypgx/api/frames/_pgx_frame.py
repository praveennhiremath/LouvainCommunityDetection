#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
import sys

from jnius import autoclass

from pypgx.api._pgx_context_manager import PgxContextManager
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import UNHASHABLE_TYPE, ARG_MUST_BE
from pypgx.api._pgql_result_set import PgqlResultSet
from pypgx.api.frames._pgx_frame_storer import PgxGenericFrameStorer
from typing import List, Optional, Tuple, Mapping, TextIO, NoReturn

ByteArrayOutputStream = autoclass('java.io.ByteArrayOutputStream')
PrintStream = autoclass('java.io.PrintStream')
JavaPgxFrame = autoclass('oracle.pgx.api.frames.PgxFrame')
ColumnRenaming = autoclass('oracle.pgx.api.frames.functions.ColumnRenaming')

DEFAULT_PRINT_LIMIT = JavaPgxFrame.PRINT_LIMIT


class PgxFrame(PgxContextManager):
    """Data-structure to load/store and manipulate tabular data.

    It contains rows and columns. A PgxFrame can contain multiple columns where each column
    consist of elements of the same data type, and has a name. The list of the columns
    with their names and data types defines the schema of the frame.
    (The number of rows in the PgxFrame is not part of the schema of the frame.)
    """

    _java_class = 'oracle.pgx.api.frames.PgxFrame'

    def __init__(self, java_pgx_frame) -> None:
        self._frame = java_pgx_frame

    def __repr__(self) -> str:
        return '{}(columns: {})'.format(self.__class__.__name__, self.get_column_descriptors())

    def __update_frame(self, java_pgx_frame) -> None:
        java_handler(self._frame.destroy, [])
        self._frame = java_pgx_frame

    @property
    def columns(self) -> List[str]:
        """Get the names of the columns contained in the PgxFrame.

        :rtype: list
        """
        return [
            java_handler(col_desc.getColumnName, [])
            for col_desc in java_handler(self._frame.getColumnDescriptors, [])
        ]

    @property
    def length(self) -> int:
        """Return the number of rows in the frame.

        :returns: number of rows"""
        return java_handler(self._frame.length, [])

    def count(self) -> int:
        """Count number of elements in the frame."""
        return self._frame.count()

    def get_column_descriptors(self) -> List[Tuple[str, str]]:
        """Return a list containing the description of the different columns of the frames."""

        return [
            (
                java_handler(col_desc.getColumnName, []),
                java_handler(col_desc.getColumnType().simpleString, []),
            )
            for col_desc in java_handler(self._frame.getColumnDescriptors, [])
        ]

    def get_column(self, name: str) -> "PgxFrameColumn":
        """Return a PgxFrameColumn.

        :param name: Column name
        :returns: PgxFrameColumn
        """

        java_pgx_frame_column = java_handler(self._frame.getColumn, [name])
        return PgxFrameColumn(java_pgx_frame_column)

    def rename_column(
        self, old_column_name: str, new_column_name: str, inplace: bool = False
    ) -> "PgxFrame":
        """Return a PgxFrame with the column name modified.

        :param old_column_name: name of the column to rename
        :param new_column_name: name of the column after the operation
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        if not inplace:
            java_pgx_frame = java_handler(
                self._frame.renameColumn, [old_column_name, new_column_name]
            )
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.renameColumnInPlace, [old_column_name, new_column_name])
        return self

    def rename_columns(
        self,
        column_renaming: Mapping[str, str],
        inplace: bool = False,
    ) -> "PgxFrame":
        """Return a PgxFrame with the column name modified.

        :param column_renaming: dict-like holding old_column names as keys and new column names as
            values
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        names = []
        for key in column_renaming:
            names.append(java_handler(ColumnRenaming.renaming, [key, column_renaming[key]]))
        if not inplace:
            java_pgx_frame = java_handler(self._frame.renameColumns, names)
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.renameColumnsInPlace, names)
        return self

    def select(self, *columns: str, inplace: bool = False) -> "PgxFrame":
        """Select multiple columns by column name.

        :param columns: Column names
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        if not inplace:
            java_pgx_frame = java_handler(self._frame.select, columns)
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.selectInPlace, columns)
        return self

    def flatten(self, *columns: str, inplace: bool = False):
        """Create a new PgxFrame with all the specified columns and vector columns flattened into
        multiple columns.

        :param columns: Column names
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        if not inplace:
            java_pgx_frame = java_handler(self._frame.flatten, columns)
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.flattenInPlace, columns)
        return self

    def flatten_all(self, inplace: bool = False) -> "PgxFrame":
        """Create a new PgxFrame with all nested columns and vector columns flattened into multiple
        columns.

        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        if not inplace:
            java_pgx_frame = java_handler(self._frame.flattenAll, [])
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.flattenAllInPlace, [])
        return self

    def head(self, num_rows: int = 10, inplace: bool = False) -> "PgxFrame":
        """Return the first num_rows elements of the frame

        :param num_rows: Number of rows to take
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame"""
        if not inplace:
            java_pgx_frame = java_handler(self._frame.head, [num_rows])
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.headInPlace, [num_rows])
        return self

    def tail(self, num_rows: int = 10, inplace: bool = False) -> "PgxFrame":
        """Return the last num_rows elements of the frame

        :param num_rows: Number of rows to take
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame"""
        if not inplace:
            java_pgx_frame = java_handler(self._frame.tail, [num_rows])
            return PgxFrame(java_pgx_frame)
        java_handler(self._frame.tailInPlace, [num_rows])
        return self

    def join(
        self,
        right: "PgxFrame",
        join_key_column: Optional[str] = None,
        left_join_key_column: Optional[str] = None,
        right_join_key_column: Optional[str] = None,
        left_prefix: Optional[str] = None,
        right_prefix: Optional[str] = None,
        inplace: bool = False,
    ) -> "PgxFrame":
        """Create a new PgxFrame by performing a join operation.

         Create a new PgxFrame by adding the columns of the right frame to this frame, aligned on
         equality of entries in column left_join_key_column for this frame and column
         right_join_key_column for the right frame, or join_key_columns on both frames. The
         resulting frame will contain the columns of this frame prefixed by left_prefix and the
         columns of right frame prefixed by right_prefix (if the prefixes are not null). Prefixes
         must ether not be set or both be set.

        :param right: PgxFrame whose columns will be added to the columns of this PgxFrame
        :param join_key_column: Column of both frames on which the equality test will be performed
        :param left_join_key_column: Column of this frame on which the equality test will be
            performed with right_join_key_column
        :param right_join_key_column: Column of right frame on which the equality test will be
            performed with leftJoinKeyColumn
        :param left_prefix: Prefix of the columns name of this frame in the resulting frame
        :param right_prefix: Prefix of the columns name of right frame in the resulting frame
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        if not isinstance(right, PgxFrame):
            raise TypeError(ARG_MUST_BE.format(arg='frames', type=PgxFrame.__name__))

        xor_left_right_key = (left_join_key_column is None) != (right_join_key_column is None)
        both_left_right_key = (left_join_key_column is not None) and (
            right_join_key_column is not None
        )

        if (
            (join_key_column is None and not both_left_right_key)
            or (join_key_column is not None and both_left_right_key)
            or xor_left_right_key
        ):
            raise ValueError(
                "Either join_key_column or both left_* and right_join_key_column must " "be set"
            )
        if (join_key_column is not None) and ((left_prefix is None) or (right_prefix is None)):
            raise ValueError("join_key_column must be called alongside left_* and right_prefix")
        if (left_prefix is None) != (right_prefix is None):
            raise ValueError("Both prefixes must be set or not set")

        # Getting all non-None values into a list
        inputs = [
            i
            for i in [
                right._frame,
                join_key_column,
                left_join_key_column,
                right_join_key_column,
                left_prefix,
                right_prefix,
            ]
            if i is not None
        ]
        java_pgx_frame = java_handler(self._frame.join, inputs)
        if not inplace:
            return PgxFrame(java_pgx_frame)
        self.__update_frame(java_pgx_frame)
        return self

    def union(self, *frames: "PgxFrame", inplace: bool = False) -> "PgxFrame":
        """Create a PgxFrame by concatenating the rows of this frame with the rows of the
        frames in frames. The different frames should have the same columns (same names, types and
        dimensions), in the same order. The resulting frame is not guaranteed to have any specific
        ordering of its rows.

        :param frames: Frames tu add through union
        :param inplace: Apply the changes inplace and return self
        :returns: PgxFrame
        """
        for frame in frames:
            if not isinstance(frame, PgxFrame):
                raise TypeError(ARG_MUST_BE.format(arg='frames', type=PgxFrame.__name__))
        java_frames = map((lambda f: f._frame), frames)
        java_pgx_frame = java_handler(self._frame.union, java_frames)
        if not inplace:
            return PgxFrame(java_pgx_frame)
        self.__update_frame(java_pgx_frame)
        return self

    def clone(self) -> "PgxFrame":
        """Create a new PgxFrame with the same content as the current frame

        :returns: PgxFrame
        """
        return self.head(self.count())

    def to_pgql_result_set(self) -> PgqlResultSet:
        """Create a new PgqlResultSet having the same content as this frame.

        :return: PgqlResultSet
        """
        from pypgx.api import PgqlResultSet

        java_pgql_result_set = java_handler(self._frame.toPgqlResultSet, [])
        return PgqlResultSet(None, java_pgql_result_set)

    def to_pandas(self):
        """
        Convert to pandas DataFrame.

        This method may change result_set cursor.

        This method requires pandas.

        :return: PgxFrame as a Pandas Dataframe
        """
        result_set = self.to_pgql_result_set()
        df = result_set.to_pandas()
        result_set.close()

        return df

    def print(
        self,
        file: Optional[TextIO] = None,
        num_results: int = DEFAULT_PRINT_LIMIT,
        start: int = 0,
    ) -> None:
        """Print the frame.

        :param file: File to which results are printed (default is ``sys.stdout``)
        :param num_results: Number of results to be printed
        :param start: Index of the first result to be printed
        """
        if file is None:
            # We don't have sys.stdout as a default parameter so that any changes
            # to sys.stdout are taken into account by this function
            file = sys.stdout

        # GM-21982: redirect output to the right file
        output_stream = ByteArrayOutputStream()
        print_stream = PrintStream(output_stream, True)
        self._frame.print(print_stream, num_results, start)
        print(output_stream.toString(), file=file)
        print_stream.close()
        output_stream.close()

    def store(self, path: str, file_format: str = 'csv', overwrite: bool = True) -> None:
        """Store the frame in a file.

        :param path: Path where to store the frame
        :param file_format: Storage format
        :param overwrite: Overwrite current file
        """

        self._frame.write().format(file_format).overwrite(overwrite).store(path)

    def write(self) -> PgxGenericFrameStorer:
        """Get Pgx Frame storer

        :return: PgxGenericFrameStorer
        """
        return PgxGenericFrameStorer(java_handler(self._frame.write, []))

    def close(self) -> None:
        """Free resources on the server taken up by this frame."""
        java_handler(self._frame.close, [])

    def destroy(self) -> None:
        """Free resources on the server taken up by this frame."""
        java_handler(self._frame.destroy, [])

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxFrameColumn(PgxContextManager):
    """Class representing one column of a :class:`PgxFrame`."""

    _java_class = 'oracle.pgx.api.frames.PgxFrameColumn'

    def __init__(self, java_pgx_frame_column) -> None:
        self._column = java_pgx_frame_column

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.get_descriptor())

    def get_descriptor(self) -> Tuple[str, str]:
        """Return a description of the column."""
        col_desc = self._column.getDescriptor()
        return col_desc.getColumnName(), col_desc.getColumnType().simpleString()

    def destroy(self) -> None:
        """Free resources on the server taken up by this column."""
        java_handler(self._column.destroy, [])

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
