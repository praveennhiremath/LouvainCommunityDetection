#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
import sys
from jnius import autoclass
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import INDEX_OUT_OF_BOUNDS, VALID_INTERVAL
from pypgx._utils.item_converter import convert_to_python_type
from pypgx._utils.pgx_types import col_types
from pypgx.api._pgx_context_manager import PgxContextManager
from datetime import date, datetime, time
from typing import Any, Iterator, List, Optional, Tuple, Union, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._pgx_graph import PgxGraph
    from pypgx.api.frames import PgxFrame
    from pypgx.api._pgx_entity import PgxEdge, PgxVertex

ByteArrayOutputStream = autoclass('java.io.ByteArrayOutputStream')
PrintStream = autoclass('java.io.PrintStream')
ResultSetFormatter = autoclass('oracle.pgql.lang.ResultSetFormatter')
PythonClientResultSetUtil = autoclass('oracle.pgx.pypgx.internal.PythonClientResultSetUtil')

DEFAULT_PRINT_LIMIT = ResultSetFormatter.DEFAULT_PRINT_LIMIT


class PgqlResultSet(PgxContextManager):
    """Result set of a pattern matching query.

    Note: retrieving results from the server is not thread-safe.
    """

    _java_class = 'oracle.pgx.api.PgqlResultSet'

    def __init__(self, graph: Optional["PgxGraph"], java_pgql_result_set) -> None:
        self._pgql_result_set = java_pgql_result_set
        self._result_set_util = PythonClientResultSetUtil(java_pgql_result_set)
        self.graph = graph
        self.id = java_pgql_result_set.getId()
        self.num_results = java_pgql_result_set.getNumResults()
        self.pgql_result_elements = {}
        self._cached_data = {}
        self._cache_ceil = -1
        self._id_cols = {}
        self.is_closed = False
        metadata = java_handler(self._pgql_result_set.getMetaData, [])
        self.col_count = java_handler(metadata.getColumnCount, [])
        self.columns = [
            java_handler(metadata.getColumnName, [i + 1]) for i in range(self.col_count)
        ]

        result_elements = java_pgql_result_set.getPgqlResultElements()
        for idx in range(result_elements.size()):
            col_name = result_elements.get(idx).getVarName()
            col_type = str(result_elements.get(idx).getElementType().toString())
            self.pgql_result_elements[idx] = col_name
            if col_type in col_types:
                self._id_cols[idx] = col_type

    def _assert_not_closed(self) -> None:
        if self.is_closed:
            raise RuntimeError("result set closed")

    def get_row(self, row: int) -> Any:
        """Get row from result_set.
        This method may change result_set cursor.

        :param row: Row index
        """
        self._assert_not_closed()
        if row < 0 or row > self.num_results - 1:
            raise RuntimeError(INDEX_OUT_OF_BOUNDS.format(idx='row', max_idx=self.num_results - 1))

        if row in self._cached_data:
            return self._cached_data[row]
        else:
            tmp_row = self._result_set_util.toList(row, row + 1)[0]
            cached_row = list(tmp_row)

            for idx in self._id_cols.keys():
                cached_row[idx] = convert_to_python_type(tmp_row[idx], self.graph)

            if len(self.pgql_result_elements) == 1:
                cached_row = cached_row[0]
            self._cached_data[row] = cached_row

        return cached_row

    def _insert_slice_to_cache(self, typed_query_list, start, stop):
        """Insert whole slice of rows from result_list into cache."""
        for i in range(0, stop - start + 1):
            self._cached_data[i + start] = typed_query_list[i]

    def _convert_row_to_python(self, item):
        """Wrap convert_to_python_type to convert whole row,
        since lambdas are not allowed.

        :param item: row to convert
        """
        row = item
        for i in self._id_cols.keys():
            row[i] = convert_to_python_type(item[i], self.graph)
        return row

    def _convert_item_to_python(self, item):
        """Wrap convert_to_python_type to add argument,
        since lambdas are not allowed.

        :param item: Item to convert
        """
        return convert_to_python_type(item, self.graph)

    def get_slice(self, start: int, stop: int, step: int = 1) -> List[list]:
        """Get slice from result_set.
        This method may change result_set cursor.

        :param start: Start index
        :param stop: Stop index
        :param step: Step size
        """
        self._assert_not_closed()

        if start < 0 or stop > self.num_results - 1 or start > stop:
            raise RuntimeError(
                VALID_INTERVAL.format(start=start, stop=stop, max_idx=self.num_results)
            )

        # fill cache first if data is not available
        if stop >= self._cache_ceil:
            query_list = self._result_set_util.toList(self._cache_ceil+1, stop+1)
            typed_query_list = list(map(self._convert_row_to_python, query_list))
            self._insert_slice_to_cache(typed_query_list, self._cache_ceil+1, stop)
            self._cache_ceil = stop

        # fill the slice
        typed_query_list = []
        for row in range(start, stop+1, step):
            typed_query_list.append(self._cached_data[row])

        return typed_query_list

    def to_frame(self) -> "PgxFrame":
        """Copy the content of this result set into a new PgxFrames

        :return: a new PgxFrame containing the content of the result set
        """
        self._assert_not_closed()
        from pypgx.api.frames._pgx_frame import PgxFrame

        java_frame = java_handler(self._pgql_result_set.toFrame, [])
        return PgxFrame(java_frame)

    def to_pandas(self):
        """
        Convert to pandas DataFrame.

        This method may change result_set cursor.

        This method requires pandas.

        :return: PgqlResultSet as a Pandas Dataframe
        """
        self._assert_not_closed()
        try:
            import pandas as pd
        except Exception:
            raise ImportError("Could not find pandas package")

        untyped_results = self._result_set_util.toList(0, self.num_results)

        df = pd.DataFrame(untyped_results, columns=self.columns)
        for idx in self._id_cols.keys():
            if self._id_cols[idx] == 'boolean':
                df[self.pgql_result_elements[idx]] = df[self.pgql_result_elements[idx]].astype(
                    'bool'
                )
                continue
            df[self.pgql_result_elements[idx]] = df[self.pgql_result_elements[idx]].apply(
                self._convert_item_to_python
            )

        created_df = df.sort_index().infer_objects()

        return created_df

    def absolute(self, row: int) -> bool:
        """Move the cursor to the given row number in this ResultSet object.

        If the row number is positive, the cursor moves to the given row number with respect to the
        beginning of the result set. The first row is 1, so absolute(1) moves the cursor to the
        first row.

        If the row number is negative, the cursor moves to the given row number with respect to the
        end of the result set. So absolute(-1) moves the cursor to the last row.

        :param row: Row to move to

        :return: True if the cursor is moved to a position in the ResultSet object;
            False if the cursor is moved before the first or after the last row
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.absolute, [row])

    def relative(self, rows: int) -> bool:
        """Move the cursor a relative number of row with respect to the current position. A
        negative number will move the cursor backwards.

        Note: Calling relative(1) is equal to next() and relative(-1) is equal to previous. Calling
        relative(0) is possible when the cursor is positioned at a row, not when it is positioned
        before the first or after the last row. However, relative(0) will not update the position of
        the cursor.

        :param rows: Relative number of rows to move from current position

        :return: True if the cursor is moved to a position in the ResultSet object; False if
            the cursor is moved before the first or after the last row
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.relative, [rows])

    def before_first(self) -> None:
        """Set the cursor before the first row"""
        self._assert_not_closed()
        java_handler(self._pgql_result_set.beforeFirst, [])

    def after_last(self) -> None:
        """Place the cursor after the last row"""
        self._assert_not_closed()
        java_handler(self._pgql_result_set.afterLast, [])

    def first(self) -> bool:
        """Move the cursor to the first row in the result set

        :return: True if the cursor points to a valid row; False if the result set does not
            have any results
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.first, [])

    def last(self) -> bool:
        """Move the cursor to the first row in the result set

        :return: True if the cursor points to a valid row; False if the result set does not
            have any results
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.last, [])

    def next(self) -> bool:
        """Move the cursor forward one row from its current position

        :return: True if the cursor points to a valid row; False if the new cursor is positioned
            after the last row
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.next, [])

    def previous(self) -> bool:
        """Move the cursor to the previous row from its current position

        :return: True if the cursor points to a valid row; False if the new cursor is positioned
            before the first row
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.previous, [])

    def get(self, element: Union[str, int]) -> Any:
        """Get the value of the designated element by element index or name

        :param element: Integer or string representing index or name
        :return: Content of cell
        """
        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getObject, [element]))

    def get_boolean(self, element: Union[str, int]) -> bool:
        """Get the value of the designated element by element index or name as a Boolean

        :param element: Integer or String representing index or name
        :return: Boolean
        """
        self._assert_not_closed()
        return bool(java_handler(self._pgql_result_set.getBoolean, [element]))

    def get_date(self, element: Union[str, int]) -> date:
        """Get the value of the designated element by element index or name as a datetime Date

        :param element: Integer or String representing index or name
        :return: datetime.date
        """
        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getDate, [element]))

    def get_double(self, element: Union[str, int]) -> float:
        """Get the value of the designated element by element index or name as a Float

        This method is for precision, as a Java floats and doubles have different precisions

        :param element: Integer or String representing index or name
        :return: Float
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.getDouble, [element])

    def get_edge(self, element: Union[str, int]) -> "PgxEdge":
        """Get the value of the designated element by element index or name as a PgxEdge

        :param element: Integer or String representing index or name
        :return: PgxEdge
        """
        self._assert_not_closed()
        java_edge = java_handler(self._pgql_result_set.getEdge, [element])
        return convert_to_python_type(java_edge, self.graph)

    def get_float(self, element: Union[str, int]) -> float:
        """Get the value of the designated element by element index or name as a Float

        :param element: Integer or String representing index or name
        :return: Float
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.getFloat, [element])

    def get_integer(self, element: Union[str, int]) -> int:
        """Get the value of the designated element by element index or name as an Integer

        :param element: Integer or String representing index or name
        :return: Integer
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.getInteger, [element])

    def get_legacy_datetime(self, element: Union[str, int]) -> datetime:
        """Get the value of the designated element by element index or name as a Datetime.
        Works with most time and date type cells. If the date is not specified, default is set to
        to Jan 1 1970.

        :param element: Integer or String representing index or name
        :return: datetime.datetime
        """
        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getLegacyDate, [element]))

    def get_list(self, element: Union[str, int]) -> Optional[List[str]]:
        """Get the value of the designated element by element index or name as a List

        :param element: Integer or String representing index or name
        :return: List
        """
        # The return type is Optional[...] because this method returns None if the PQGL query gives
        # a null value, which can happen with ARRAY_AGG.

        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getList, [element]))

    def get_long(self, element: Union[str, int]) -> int:
        """Get the value of the designated element by element index or name as a Long

        :param element: Integer or String representing index or name
        :return: Long
        """
        return java_handler(self._pgql_result_set.getLong, [element])

    def get_point2d(self, element: Union[str, int]) -> Tuple[float, float]:
        """Get the value of the designated element by element index or name as a 2D tuple

        :param element: Integer or String representing index or name
        :return: (X, Y)
        """
        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getPoint2D, [element]))

    def get_string(self, element: Union[str, int]) -> str:
        """Get the value of the designated element by element index or name as a String

        :param element: Integer or String representing index or name
        :return: String
        """
        self._assert_not_closed()
        return java_handler(self._pgql_result_set.getString, [element])

    def get_time(self, element: Union[str, int]) -> time:
        """Get the value of the designated element by element index or name as a datetime Time

        :param element: Integer or String representing index or name
        :return: datetime.time
        """
        self._assert_not_closed()
        return convert_to_python_type(java_handler(self._pgql_result_set.getTime, [element]))

    def get_time_with_timezone(self, element: Union[str, int]) -> time:
        """Get the value of the designated element by element index or name as a datetime Time that
        includes timezone

        :param element: Integer or String representing index or name
        :return: datetime.time
        """
        self._assert_not_closed()
        time = java_handler(self._pgql_result_set.getTimeWithTimezone, [element])
        return convert_to_python_type(time)

    def get_timestamp(self, element: Union[str, int]) -> datetime:
        """Get the value of the designated element by element index or name as a Datetime

        :param element: Integer or String representing index or name
        :return: datetime.datetime
        """
        self._assert_not_closed()
        java_timestamp = java_handler(self._pgql_result_set.getTimestamp, [element])
        return convert_to_python_type(java_timestamp)

    def get_timestamp_with_timezone(self, element: Union[str, int]) -> datetime:
        """Get the value of the designated element by element index or name as a Datetime

        :param element: Integer or String representing index or name
        :return: datetime.datetime
        """
        self._assert_not_closed()
        java_timestamp = java_handler(self._pgql_result_set.getTimestampWithTimezone, [element])
        return convert_to_python_type(java_timestamp)

    def get_vertex(self, element: Union[str, int]) -> "PgxVertex":
        """Get the value of the designated element by element index or name as a PgxVertex

        :param element: Integer or String representing index or name
        :return: PgxVertex
        """
        self._assert_not_closed()
        java_vertex = java_handler(self._pgql_result_set.getVertex, [element])
        return convert_to_python_type(java_vertex, self.graph)

    def get_vertex_labels(self, element: Union[str, int]) -> List[str]:
        """Get the value of the designated element by element index or name a list of labels

        :param element: Integer or String representing index or name
        :return: list
        """
        self._assert_not_closed()
        return list(java_handler(self._pgql_result_set.getVertexLabels, [element]))

    def __len__(self) -> int:
        self._assert_not_closed()
        return self.num_results

    def __getitem__(self, idx: Union[slice, int]) -> Union[List[list], List[Any]]:
        self._assert_not_closed()
        if isinstance(idx, slice):
            istart = 0 if idx.start is None else idx.start
            istop = self.num_results if idx.stop is None else idx.stop
            istep = 1 if idx.step is None else idx.step
            return self.get_slice(start=istart, stop=istop - 1, step=istep)
        else:
            return self.get_row(idx)

    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over result_set object
        This method may change result_set cursor.
        """
        self._assert_not_closed()
        return (self.get_row(row) for row in range(self.num_results))

    def __repr__(self) -> str:
        self._assert_not_closed()
        return '{}(id: {}, num. results: {}, graph: {})'.format(
            self.__class__.__name__,
            self.id,
            self.num_results,
            (self.graph.name if self.graph is not None else 'None'),
        )

    def __str__(self) -> str:
        self._assert_not_closed()
        return repr(self)

    def print(
        self,
        file: Optional[TextIO] = None,
        num_results: int = DEFAULT_PRINT_LIMIT,
        start: int = 0,
    ) -> None:
        """Print the result set.

        :param file: File to which results are printed (default is ``sys.stdout``)
        :param num_results: Number of results to be printed
        :param start: Index of the first result to be printed
        """
        self._assert_not_closed()
        if file is None:
            # We don't have sys.stdout as a default parameter so that any changes
            # to sys.stdout are taken into account by this function
            file = sys.stdout

        # GM-21982: redirect output to the right file
        output_stream = ByteArrayOutputStream()
        print_stream = PrintStream(output_stream, True)
        self._pgql_result_set.print(print_stream, num_results, start)
        print(output_stream.toString(), file=file)
        print_stream.close()
        output_stream.close()

    def __hash__(self) -> int:
        self._assert_not_closed()
        return hash((str(self), str(self.graph.name), str()))

    def __eq__(self, other: object) -> bool:
        self._assert_not_closed()
        if not isinstance(other, self.__class__):
            return False
        return self._pgql_result_set.equals(other._pgql_result_set)

    def close(self) -> None:
        """Free resources on the server taken up by this frame."""
        java_handler(self._pgql_result_set.close, [])
        self.is_closed = True
