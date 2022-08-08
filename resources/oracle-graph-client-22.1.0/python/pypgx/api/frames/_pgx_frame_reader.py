#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import Tuple, List, Union, NoReturn, Optional

from pypgx._utils.error_handling import java_handler
from pypgx._utils.pgx_types import get_data_type
from ._pgx_frame import PgxFrame
from jnius import autoclass
from pypgx._utils.error_messages import UNHASHABLE_TYPE

_String = autoclass("java.lang.String")
_ColumnDescriptor = autoclass('oracle.pgx.api.frames.schema.ColumnDescriptor')


class PgxGenericFrameReader:
    """A generic class for reading :class:`PgxFrame` objects from various sources.

    The class allows configuration of how the data should be read and
    facilitates the creation of specialized frame readers
    (:class:`PgxCsvFrameReader` and :class:`PgxPgbFrameReader`).
    """

    _java_class = 'oracle.pgx.api.frames.PgxGenericFrameReader'

    def __init__(self, java_pgx_generic_frame_reader) -> None:
        self.java_pgx_generic_frame_reader = java_pgx_generic_frame_reader

    def clear_columns(self) -> None:
        """Clear the current configuration of which columns should be loaded and how.

        :return: None
        """
        java_handler(self.java_pgx_generic_frame_reader.clearColumns, [])

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxGenericFrameReader":
        """Set the columns to be loaded from their columnDescriptors.

        Executing this function disables autodetection of columns.

        :param column_descriptors: List of tuples (columnName, columnType)
        :return: self
        """
        java_column_descriptors = []
        for col_desc in column_descriptors:
            column_name = col_desc[0]
            column_type = get_data_type(col_desc[1])

            java_column_descriptors.append(
                _ColumnDescriptor.columnDescriptor(column_name, column_type)
            )

        java_handler(self.java_pgx_generic_frame_reader.columns, java_column_descriptors)

        return self

    def auto_detect_columns(self, auto_detect: bool) -> "PgxGenericFrameReader":
        """Enable or disable the autodetection of columns from the table.

        Not all formats support autodetection of the columns (only DB in fact).

        Executing this function clears the currently loaded column descriptors.

        :param auto_detect: True if the columns should be autodetected, False otherwise
        :type auto_detect: bool

        :return: self
        """
        java_handler(self.java_pgx_generic_frame_reader.autodetectColumns, [auto_detect])
        return self

    def name(self, frame_name: str) -> "PgxGenericFrameReader":
        """Set the frame name.

        :param frame_name: New name for the :class:`PgxFrame`
        :type frame_name: str

        :return: self
        """
        java_handler(self.java_pgx_generic_frame_reader.name, [frame_name])
        return self

    def format(self, format: str) -> Union["PgxCsvFrameReader", "PgxPgbFrameReader"]:
        """Return a frame reader for the type format specified.

        :param format: format to be loaded
        :type format: str
        :return: a loader for the format
        :rtype: PgxCsvFrameReader or PgxPgbFrameReader
        """

        java_pgx_frame_reader = java_handler(self.java_pgx_generic_frame_reader.format, [format])
        if format.lower() == 'csv':
            return PgxCsvFrameReader(java_pgx_frame_reader)
        elif format.lower() == 'pgb':
            return PgxPgbFrameReader(java_pgx_frame_reader)

    def csv(self, uris: Optional[str] = None) -> "PgxCsvFrameReader":
        """Create a :class:`PgxCsvFrameReader` object for loading CSV files from URIs.

        :param uris: List of paths to the csv files
        :rtype: PgxCsvFrameReader
        """
        if uris is None:
            java_pgx_csv_frame_reader = java_handler(self.java_pgx_generic_frame_reader.csv, [])
        else:
            java_pgx_csv_frame_reader = java_handler(self.java_pgx_generic_frame_reader.csv, [uris])
        return PgxCsvFrameReader(java_pgx_csv_frame_reader)

    def csv_async(self, uris: str):
        """Read a :class:`PgxFrame` from a list of URIs to CSV files.

        :param uris: list denoting the URIs
        :return: the read frame

        .. note: The asynchronous behavior is not implemented yet. This is blocking.
        """
        java_pgx_frame_future = java_handler(self.java_pgx_generic_frame_reader.csvAsync, [uris])
        java_pgx_frame = java_pgx_frame_future.get()
        return PgxFrame(java_pgx_frame)

    def pgb(self, uris: Optional[str] = None) -> "PgxPgbFrameReader":
        """Create a :class:`PgxPgbFrameReader` object for loading PGB files.

        :param uris: List of paths to the PGB files
        :rtype: PgxPgbFrameReader
        """
        if uris is None:
            java_pgx_frame_reader = java_handler(self.java_pgx_generic_frame_reader.pgb, [])
        else:
            java_pgx_frame_reader = java_handler(self.java_pgx_generic_frame_reader.pgb, [uris])
        return PgxPgbFrameReader(java_pgx_frame_reader)

    def pgb_async(self, uris: str):
        """Read a :class:`PgxFrame` from a list of URIs to PGB files.

        :param uris: list denoting the URIs
        :return: the read frame

        .. note: The asynchronous behavior is not implemented yet. This is blocking.
        """
        java_pgx_frame_future = java_handler(self.java_pgx_generic_frame_reader.pgbAsync, [uris])
        java_pgx_frame = java_pgx_frame_future.get()
        return PgxFrame(java_pgx_frame)

    def db(self) -> "PgxDbFrameReader":
        """Create a :class:`PgxDbFrameReader` object to load :class:`PgxFrame` from a database.

        :rtype: PgxDbFrameReader
        """
        return PgxDbFrameReader(java_handler(self.java_pgx_generic_frame_reader.db, []))

    # TODO : java implementation of dbAsync ?

    def load_async(self):
        # noqa: D102
        # NOTE: This function throws a not implemented exception on the java side
        java_pgx_frame = java_handler(self.java_pgx_generic_frame_reader.loadAsync, [])
        return PgxFrame(java_pgx_frame)

    def load(self):
        # noqa: D102
        # NOTE: This function throws a not implemented exception on the java side
        java_pgx_frame = java_handler(self.java_pgx_generic_frame_reader.load, [])
        return PgxFrame(java_pgx_frame)

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxCsvFrameReader:
    """Class for reading :class:`PgxFrame` objects from CSV files."""

    _java_class = 'oracle.pgx.api.frames.PgxCsvFrameReader'

    def __init__(self, java_pgx_csv_frame_reader) -> None:
        self.java_pgx_csv_frame_reader = java_pgx_csv_frame_reader

    def name(self, frame_name: str) -> "PgxCsvFrameReader":
        """Set the frame name.

        :param frame_name: New name for the :class:`PgxFrame`
        :type frame_name: str

        :return: self
        """
        java_handler(self.java_pgx_csv_frame_reader.name, [frame_name])
        return self

    def clear_columns(self) -> None:
        """Clear the current configuration of which columns should be loaded and how.

        :return: None
        """

        java_handler(self.java_pgx_csv_frame_reader.clearColumns, [])

    def separator(self, sep: str) -> "PgxCsvFrameReader":
        """Set the separator for CSV parsing to `sep`.

        :param sep: char denoting the separator
        :return: self
        """
        c = _String(sep).charAt(0)
        java_handler(self.java_pgx_csv_frame_reader.separator, [c])
        return self

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxCsvFrameReader":
        """Set the columns to be loaded from their columnDescriptors.

        :param column_descriptors: List of tuples (columnName, columnType)
        :return: self
        """
        java_column_descriptors = []
        for col_desc in column_descriptors:
            column_name = col_desc[0]
            column_type = get_data_type(col_desc[1])
            java_column_descriptor = _ColumnDescriptor.columnDescriptor(column_name, column_type)

            java_column_descriptors.append(java_column_descriptor)

        java_handler(self.java_pgx_csv_frame_reader.columns, java_column_descriptors)

        return self

    def auto_detect_columns(self, auto_detect: bool) -> "PgxCsvFrameReader":
        """Enable or disable the autodetection of columns from the table.

        Executing this function clears the currently loaded column descriptors.

        :param auto_detect: True if the columns should be autodetected, False otherwise
        :type auto_detect: bool

        :return: self
        """
        java_handler(self.java_pgx_csv_frame_reader.autodetectColumns, [auto_detect])
        return self

    def load_async(self, uris: str):
        """Load a :class:`PgxFrame` from the provided URIs.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance

        .. note: The asynchronous behavior is not implemented yet. This is blocking.
           Same as :method:`load`
        """
        java_pgx_frame_future = java_handler(self.java_pgx_csv_frame_reader.loadAsync, [uris])
        java_pgx_frame = java_pgx_frame_future.get()
        return PgxFrame(java_pgx_frame)

    def load(self, uris: str) -> PgxFrame:
        """Load a :class:`PgxFrame` from the provided URIs.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance
        """
        java_pgx_frame = java_handler(self.java_pgx_csv_frame_reader.load, [uris])
        return PgxFrame(java_pgx_frame)

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxPgbFrameReader:
    """Class for reading `PgxFrame` objects from PGB files."""

    _java_class = 'oracle.pgx.api.frames.PgxPgbFrameReader'

    def __init__(self, java_pgx_pgb_frame_reader) -> None:
        self.java_pgx_pgb_frame_reader = java_pgx_pgb_frame_reader

    def name(self, frame_name: str) -> "PgxPgbFrameReader":
        """Set the frame name.

        :param frame_name: New name for the :class:`PgxFrame`
        :type frame_name: str

        :return: self
        """
        java_handler(self.java_pgx_pgb_frame_reader.name, [frame_name])
        return self

    def clear_columns(self) -> None:
        """Clear the current configuration of which columns should be loaded and how.

        :return: None
        """
        java_handler(self.java_pgx_pgb_frame_reader.clearColumns, [])

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxPgbFrameReader":
        """Set the columns to be loaded from their columnDescriptors.

        Executing this function disables autodetection of columns.

        :param column_descriptors: List of tuples (columnName, columnType)
        :return: self
        """
        java_column_descriptors = []
        for col_desc in column_descriptors:
            column_name = col_desc[0]
            column_type = get_data_type(col_desc[1])

            java_column_descriptors.append(
                _ColumnDescriptor.columnDescriptor(column_name, column_type)
            )

        java_handler(self.java_pgx_pgb_frame_reader.columns, java_column_descriptors)

        return self

    def auto_detect_columns(self, auto_detect: bool) -> "PgxPgbFrameReader":
        """Enable or disable the autodetection of columns from the table.

        Executing this function clears the currently loaded column descriptors.

        :param auto_detect: True if the columns should be autodetected, False otherwise
        :type auto_detect: bool

        :return: self
        """
        java_handler(self.java_pgx_pgb_frame_reader.autodetectColumns, [auto_detect])
        return self

    def load_async(self, uris: str):
        """Load a :class:`PgxFrame` from the provided URIs.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance

        .. note: The asynchronous behavior is not implemented yet. This is blocking.
           Same as :method:`load`
        """
        java_pgx_frame_future = java_handler(self.java_pgx_pgb_frame_reader.pgbAsync, [uris])
        java_pgx_frame = java_pgx_frame_future.get()
        return PgxFrame(java_pgx_frame)

    def load(self, uris: str) -> PgxFrame:
        """Load a :class:`PgxFrame` from the provided URIs.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance
        """
        java_pgx_frame = java_handler(self.java_pgx_pgb_frame_reader.pgbAsync, [uris])
        return PgxFrame(java_pgx_frame)

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxDbFrameReader:
    """Class for reading :class:`PgxFrame` objects from a database."""

    _java_class = 'oracle.pgx.api.frames.PgxDbFrameReader'

    def __init__(self, java_pgx_db_frame_reader) -> None:
        self.java_pgx_db_frame_reader = java_pgx_db_frame_reader

    def name(self, frame_name: str) -> "PgxDbFrameReader":
        """Set the frame name.

        :param frame_name: New name for the :class:`PgxFrame`
        :type frame_name: str

        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.name, [frame_name])
        return self

    def clear_columns(self) -> None:
        """Clear the current configuration of which columns should be loaded and how.

        :return: None
        """
        java_handler(self.java_pgx_db_frame_reader.clearColumns, [])

    def jdbc_url(self, jdbc_url):
        """Set the jdbc URL to use for connecting to the DB.

        :param jdbc_url: the jdbc URL
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.jdbcUrl, [jdbc_url])
        return self

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxDbFrameReader":
        """Set the columns to be loaded from their columnDescriptors.

        Executing this function disables autodetection of columns.

        :param column_descriptors: List of tuples (columnName, columnType)
        :return: self
        """
        java_column_descriptors = []
        for col_desc in column_descriptors:
            column_name = col_desc[0]
            column_type = get_data_type(col_desc[1])

            java_column_descriptors.append(
                _ColumnDescriptor.columnDescriptor(column_name, column_type)
            )

        java_handler(self.java_pgx_db_frame_reader.columns, java_column_descriptors)

        return self

    def auto_detect_columns(self, auto_detect: bool) -> "PgxDbFrameReader":
        """Enable or disable the autodetection of columns from the table.

        Executing this function clears the currently loaded column descriptors.

        :param auto_detect: True if the columns should be autodetected, False otherwise
        :type auto_detect: bool

        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.autodetectColumns, [auto_detect])
        return self

    def load(self) -> PgxFrame:
        """Load a :class:`PgxFrame` from the database.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance
        """
        java_pgx_frame = java_handler(self.java_pgx_db_frame_reader.load, [])
        return PgxFrame(java_pgx_frame)

    def load_async(self):
        """Load a :class:`PgxFrame` from the database.

        :param uris: the URIs from which to load the frame
        :return: PgxFrame instance

        .. note: The asynchronous behavior is not implemented yet. This is blocking.
           Same as :method:`load`
        """
        java_pgx_frame_future = java_handler(self.java_pgx_db_frame_reader.loadAsync, [])
        java_pgx_frame = java_pgx_frame_future.get()
        return PgxFrame(java_pgx_frame)

    def table_name(self, table_name: str) -> "PgxDbFrameReader":
        """Set the table name in the database.

        :param table_name: nodes table name.
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.tablename, [table_name])
        return self

    def data_source_id(self, data_source_id: str) -> "PgxDbFrameReader":
        """Set the datasource ID.

        :param data_source_id: the datasource ID
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.dataSourceId, [data_source_id])
        return self

    def username(self, username: str) -> "PgxDbFrameReader":
        """Set the username of the database.

        :param username: username
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.username, [username])
        return self

    def keystore_alias(self, keystore_alias):
        """Set the keystore alias.

        :param keystore_alias: the keystore alias.
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.keystoreAlias, [keystore_alias])
        return self

    def password(self, password: str) -> "PgxDbFrameReader":
        """Set the password of the database.

        :param password: the password
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.password, [password])
        return self

    def schema(self, schema: str) -> "PgxDbFrameReader":
        """Set the schema of the table.

        :param schema: the schema.
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.schema, [schema])
        return self

    def owner(self, owner: str) -> "PgxDbFrameReader":
        """Set the owner of the table.

        :param owner: the owner
        :return: self
        """
        owner_frame_reader = java_handler(self.java_pgx_db_frame_reader.owner, [owner])
        return owner_frame_reader

    def connections(self, connections: int) -> "PgxDbFrameReader":
        """Set the number of connections to read/write data from/to the database provider

        :param connections: number of connections
        :return: self
        """
        java_handler(self.java_pgx_db_frame_reader.connections, [connections])
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))
