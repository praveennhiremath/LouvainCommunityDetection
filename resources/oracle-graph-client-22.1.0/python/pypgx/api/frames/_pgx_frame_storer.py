#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from typing import Union, Tuple, List, NoReturn

from pypgx._utils.error_handling import java_handler
from pypgx._utils.pgx_types import get_data_type
from jnius import autoclass
from pypgx._utils.error_messages import UNHASHABLE_TYPE

_String = autoclass("java.lang.String")
_ColumnDescriptor = autoclass('oracle.pgx.api.frames.schema.ColumnDescriptor')


class PgxGenericFrameStorer:
    """Class for configuring the storing operation of a :class:`PgxFrame` and then triggering it."""

    _java_class = 'oracle.pgx.api.frames.PgxGenericFrameStorer'

    def __init__(self, java_pgx_generic_frame_storer) -> None:
        self.java_pgx_generic_frame_storer = java_pgx_generic_frame_storer

    def format(
        self, format: str
    ) -> Union["PgxCsvFrameStorer", "PgxPgbFrameStorer", "PgxDbFrameStorer"]:
        """Create a specialized frame storer for saving the :class:`PgxFrame` in a given format.

        :param format: identifier of the wanted format.
            Can be one of 'csv', 'pgb' or 'db' (case insensitive)
        :type format: str

        :rtype: PgxCsvFrameStorer, PgxPgbFrameStorer or PgxDbFrameStorer
        """
        java_pgx_frame_storer = java_handler(self.java_pgx_generic_frame_storer.format, [format])
        if format.lower() == 'csv':
            return PgxCsvFrameStorer(java_pgx_frame_storer)
        elif format.lower() == 'pgb':
            return PgxPgbFrameStorer(java_pgx_frame_storer)
        elif format.lower() == 'db':
            return PgxDbFrameStorer(java_pgx_frame_storer)
        else:
            raise RuntimeError("format not supported")

    def name(self, frame_name: str) -> "PgxGenericFrameStorer":
        """Set the name of the stored frame.

        :param frame_name: the new frame name
        :type frame_name: str

        :return: self
        """
        self.java_pgx_generic_frame_storer = java_handler(
            self.java_pgx_generic_frame_storer.name, [frame_name]
        )
        return self

    def overwrite(self, overwrite_bool: bool) -> "PgxGenericFrameStorer":
        """Set overwrite

        :param overwrite_bool: denotes if the table should be overwritten.
        :return: self
        """
        java_handler(self.java_pgx_generic_frame_storer.overwrite, [overwrite_bool])
        return self

    def pgb(self) -> "PgxPgbFrameStorer":
        """Get a :class:`PgxPgbFrameStorer` instance for the :class:`PgxFrame`.

        :rtype: PgxPgbFrameStorer
        """
        return PgxDbFrameStorer(java_handler(self.java_pgx_generic_frame_storer.db, []))

    def db(self) -> "PgxDbFrameStorer":
        """Get a :class:`PgxDbFrameStorer` instance for the :class:`PgxFrame`

        :rtype: PgxDbFrameStorer
        """
        return PgxDbFrameStorer(java_handler(self.java_pgx_generic_frame_storer.db, []))

    def csv(self) -> "PgxCsvFrameStorer":
        """Get a :class:`PgxCsvFrameStorer` instance for the :class:`PgxFrame`

        :rtype: PgxCsvFrameStorer
        """
        return PgxCsvFrameStorer(java_handler(self.java_pgx_generic_frame_storer.csv, []))

    # NOTE: the Java function just throws NotImplementedException
    def store(self):
        # noqa: D102
        java_handler(self.java_pgx_generic_frame_storer.store, [])


class PgxDbFrameStorer:
    """Class for configuring the storing operation of a :class:`PgxFrame` to a database
    and then triggering it.
    """

    _java_class = 'oracle.pgx.api.frames.PgxDbFrameStorer'

    def __init__(self, java_pgx_db_frame_storer) -> None:
        self.java_pgx_db_frame_storer = java_pgx_db_frame_storer

    def name(self, frame_name: str) -> "PgxDbFrameStorer":
        """Set the frame name.

        :param frame_name: frame name.
        :return: self
        """
        self.java_pgx_db_frame_storer = java_handler(
            self.java_pgx_db_frame_storer.name, [frame_name]
        )
        return self

    def overwrite(self, overwrite_bool: bool) -> "PgxDbFrameStorer":
        """Set overwrite

        :param overwrite_bool:
        :return: self
        """
        java_handler(self.java_pgx_db_frame_storer.overwrite, [overwrite_bool])
        return self

    def clear_columns(self) -> None:
        """Clear columns

        :return: self
        """
        java_handler(self.java_pgx_db_frame_storer.clearColumns, [])

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxDbFrameStorer":
        """Set columns

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

        java_handler(self.java_pgx_db_frame_storer.columns, java_column_descriptors)

        return self

    def store(self) -> None:
        """Store the :class:`PgxFrame`.

        :return: None
        """
        java_handler(self.java_pgx_db_frame_storer.store, [])

    def store_async(self) -> None:
        """Store the :class:`PgxFrame`.

        :return: None
        """
        java_handler(self.java_pgx_db_frame_storer.storeAsync, []).get()

    def table_name(self, table_name: str) -> "PgxDbFrameStorer":
        """Set the table name in the database.

        :param table_name: nodes table name.
        :return: self
        """
        java_handler(self.java_pgx_db_frame_storer.tablename, [table_name])
        return self

    def data_source_id(self, data_source_id: str) -> "PgxDbFrameStorer":
        """Set the datasource ID.

        :param data_source_id: the datasource ID
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.dataSourceId, [data_source_id])
        return self

    def jdbc_url(self, jdbc_url: str) -> "PgxDbFrameStorer":
        """Set jdbc url

        :param jdbc_url:
        :return:
        """
        java_handler(self.java_pgx_db_frame_storer.jdbcUrl, [jdbc_url])
        return self

    def username(self, username: str) -> "PgxDbFrameStorer":
        """Set the username of the database.

        :param username: username
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.username, [username])
        return self

    def keystore_alias(self, keystore_alias: str) -> "PgxDbFrameStorer":
        """Set the keystore alias.

        :param keystore_alias: the keystore alias.
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.keystoreAlias, [keystore_alias])
        return self

    def password(self, password: str) -> "PgxDbFrameStorer":
        """Set the password of the database.

        :param password: the password
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.password, [password])
        return self

    def schema(self, schema: str) -> "PgxDbFrameStorer":
        """Set the schema of the table.

        :param schema: the schema.
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.schema, [schema])
        return self

    def owner(self, owner: str) -> "PgxDbFrameStorer":
        """Set the owner of the table.

        :param owner: the owner
        :return: this storer
        """
        owner_frame_reader = java_handler(self.java_pgx_db_frame_storer.owner, [owner])
        return owner_frame_reader

    def connections(self, connections: int) -> "PgxDbFrameStorer":
        """Set the number of connections to read/write data from/to the database provider

        :param connections: number of connections
        :return: this storer
        """
        java_handler(self.java_pgx_db_frame_storer.connections, [connections])
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxCsvFrameStorer:
    """Class for configuring the storing operation of a :class:`PgxFrame` to a CSV file
    and then triggering it.
    """

    _java_class = 'oracle.pgx.api.frames.PgxCsvFrameStorer'

    def __init__(self, java_pgx_csv_frame_storer) -> None:
        self.java_pgx_csv_frame_storer = java_pgx_csv_frame_storer

    def name(self, frame_name: str) -> "PgxCsvFrameStorer":
        """Set the frame name.

        :param frame_name: frame name.
        :return: this storer
        """
        self.java_pgx_csv_frame_storer = java_handler(
            self.java_pgx_csv_frame_storer.name, [frame_name]
        )
        return self

    def partitions(self, num_partitions: int) -> "PgxCsvFrameStorer":
        """Set the number of files to be created.

        :param num_partitions: number of partitions created.
        :return: this storer
        """
        self.java_pgx_csv_frame_storer = java_handler(
            self.java_pgx_csv_frame_storer.partitions, [num_partitions]
        )
        return self

    def partition_extension(self, file_extension: str) -> "PgxCsvFrameStorer":
        """Set the fileExtension of the created CSV files.

        :param file_extension: string denoting the file extension for the created files.
        :return: this storer
        """
        self.java_pgx_csv_frame_storer = java_handler(
            self.java_pgx_csv_frame_storer.partitionExtension, [file_extension]
        )
        return self

    def separator(self, sep: str) -> "PgxCsvFrameStorer":
        """Set the separator for CSV file to `sep`.

        :param sep: char denoting the separator
        :return: self
        """
        c = _String(sep).charAt(0)
        java_handler(self.java_pgx_csv_frame_storer.separator, [c])
        return self

    def overwrite(self, overwrite_bool: bool) -> "PgxCsvFrameStorer":
        """Set overwrite

        :param overwrite_bool: denotes if the table should be overwritten.
        :return:
        """
        java_handler(self.java_pgx_csv_frame_storer.overwrite, [overwrite_bool])
        return self

    def clear_columns(self) -> None:
        """Clear columns

        :return:
        """
        java_handler(self.java_pgx_csv_frame_storer.clearColumns, [])

    def columns(self, column_descriptors: List[Tuple[str, str]]) -> "PgxCsvFrameStorer":
        """Set columns

        :param column_descriptors: List of tuples (columnName, columnType)
        :return:
        """
        java_column_descriptors = []
        for col_desc in column_descriptors:
            column_name = col_desc[0]
            column_type = get_data_type(col_desc[1])

            java_column_descriptors.append(
                _ColumnDescriptor.columnDescriptor(column_name, column_type)
            )

        java_handler(self.java_pgx_csv_frame_storer.columns, java_column_descriptors)

        return self

    def store(self) -> None:
        """Store PgxFrame

        :return: PgxFrame instance
        """
        java_handler(self.java_pgx_csv_frame_storer.store, [])

    def store_async(self):
        """Store PgxFrame

        :return: PgxFrame instance
        """
        java_handler(self.java_pgx_csv_frame_storer.storeAsync, []).get()

    def table_name(self, table_name: str) -> "PgxCsvFrameStorer":
        """Set the table name in the database.

        :param table_name: nodes table name.
        :return: this storer
        """
        java_handler(self.java_pgx_csv_frame_storer.tablename, [table_name])
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(UNHASHABLE_TYPE.format(type_name=self.__class__))


class PgxPgbFrameStorer:
    """Class for configuring the storing operation of a :class:`PgxFrame` to a PGB file
    and then triggering it.
    """

    _java_class = 'oracle.pgx.api.frames.PgxPgbFrameStorer'

    def __init__(self, java_pgx_pgb_frame_storer) -> None:
        self.java_pgx_pgb_frame_storer = java_pgx_pgb_frame_storer

    def name(self, frame_name: str) -> "PgxPgbFrameStorer":
        """Set the frame name.

        :param frame_name: frame name.
        :return: this storer
        """
        self.java_pgx_pgb_frame_storer = java_handler(
            self.java_pgx_pgb_frame_storer.name, [frame_name]
        )
        return self

    def store(self) -> None:
        """Store PgxFrame

        :return: PgxFrame instance
        """
        java_handler(self.java_pgx_pgb_frame_storer.store, [])

    def store_async(self):
        """Store PgxFrame

        :return: PgxFrame instance
        """
        java_handler(self.java_pgx_pgb_frame_storer.storeAsync, []).get()

    def overwrite(self, overwrite_bool: bool) -> "PgxPgbFrameStorer":
        """Set overwrite

        :param overwrite_bool: denotes if the table should be overwritten.
        :return:
        """
        java_handler(self.java_pgx_pgb_frame_storer.overwrite, [overwrite_bool])
        return self
