#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius.jnius import cast
from pypgx._utils.error_handling import java_handler
from pypgx._utils.error_messages import MODEL_NOT_FITTED

from typing import Callable, Optional, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Don't import at runtime, to avoid circular imports.
    from pypgx.api._analyst import Analyst
    from pypgx.api.mllib._deepwalk_model import DeepWalkModel
    from pypgx.api.mllib._pg2vec_model import Pg2vecModel
    from pypgx.api.mllib._supervised_graphwise_model import SupervisedGraphWiseModel
    from pypgx.api.mllib._unsupervised_graphwise_model import UnsupervisedGraphWiseModel


def _input_db_params(
    tmp, username, password, model_store, model_name, jdbc_url, keystore_alias, schema
):
    if model_name:
        tmp = java_handler(tmp.modelname, [model_name])
    if model_store:
        tmp = java_handler(tmp.modelstore, [model_store])
    if username:
        tmp = java_handler(tmp.username, [username])
    if password:
        tmp = java_handler(tmp.password, [password])
    if jdbc_url:
        tmp = java_handler(tmp.jdbcUrl, [jdbc_url])
    if keystore_alias:
        tmp = java_handler(tmp.keystoreAlias, [keystore_alias])
    if schema:
        tmp = java_handler(tmp.schema, [schema])
    return tmp


class ModelStorer:
    """ModelStorer object."""

    _java_class = 'oracle.pgx.api.mllib.ModelStorer'

    def __init__(
        self,
        model: Union[
            "SupervisedGraphWiseModel", "Pg2vecModel", "UnsupervisedGraphWiseModel", "DeepWalkModel"
        ],
    ) -> None:
        self.model = model

    def __repr__(self) -> str:
        return "{}(model: {})".format(self.__class__.__name__, repr(self.model))

    def __str__(self) -> str:
        return repr(self)

    def db(
        self,
        model_store: str,
        model_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        jdbc_url: Optional[str] = None,
        model_description: Optional[str] = None,
        overwrite: bool = False,
        keystore_alias: Optional[str] = None,
        schema=None,
    ) -> None:
        """Store a model to a database.

        :param username: username in database
        :param password: password of username in database
        :param model_store: model store in database
        :param model_name: name of the model to store
        :param jdbc_url: jdbc url of database
        :param model_description: description of model
        :param overwrite: boolean value for overwriting or not
        :param keystore_alias: the keystore alias to get the password in the keystore
        :param schema: the schema of the model store in database
        """
        if not self.model._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        tmp = java_handler(self.model._model.export, [])
        tmp = java_handler(tmp.db, [])
        tmp = _input_db_params(
            tmp, username, password, model_store, model_name, jdbc_url, keystore_alias, schema
        )
        if model_description:
            tmp = java_handler(tmp.description, [model_description])
        if overwrite:
            tmp = java_handler(tmp.overwrite, [overwrite])
        java_handler(tmp.store, [])

    def file(self, path: str, key: str, overwrite: bool = False) -> None:
        """Store an encrypted model to a file.

        :param path: path to store model
        :param key: key used for encryption
        :param overwrite: boolean value for overwriting or not
        """
        if not self.model._is_fitted:
            raise RuntimeError(MODEL_NOT_FITTED)
        java_handler(self.model._model.store, [path, key, overwrite])


class ModelLoader:
    """ModelLoader object."""

    _java_class = 'oracle.pgx.api.mllib.ModelLoader'

    def __init__(
        self,
        analyst: "Analyst",
        java_model_loader: Any,
        wrapper: Callable,
        java_class: str,
    ) -> None:
        self.analyst = analyst
        self._model_loader = java_model_loader
        self.wrapper = wrapper
        self.java_class = java_class

    def __repr__(self) -> str:
        return "{}(analyst: {})".format(self.__class__.__name__, repr(self.analyst))

    def __str__(self) -> str:
        return repr(self)

    def db(
        self,
        model_store: str,
        model_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        jdbc_url: Optional[str] = None,
        keystore_alias: Optional[str] = None,
        schema=None,
    ) -> Union[
        "SupervisedGraphWiseModel", "Pg2vecModel", "UnsupervisedGraphWiseModel", "DeepWalkModel"
    ]:
        """Return a model stored in a database.

        :param username: username in database
        :param password: password of username in database
        :param model_store: model store in database
        :param model_name: name of the model to load
        :param jdbc_url: jdbc url of database
        :param keystore_alias: the keystore alias to get the password in the keystore
        :param schema: the schema of the model store in database
        :returns: model stored in database.
        """
        tmp = java_handler(self._model_loader, [])
        tmp = java_handler(tmp.db, [])
        tmp = _input_db_params(
            tmp, username, password, model_store, model_name, jdbc_url, keystore_alias, schema
        )
        model = java_handler(tmp.load, [])
        return self.wrapper(cast(self.java_class, model))

    def file(
        self, path: str, key: str
    ) -> Union[
        "SupervisedGraphWiseModel", "Pg2vecModel", "UnsupervisedGraphWiseModel", "DeepWalkModel"
    ]:
        """Return an encrypted model stored in a file.

        :param path: path of stored model
        :param key: used for encryption
        :returns: model stored in file.
        """
        tmp = java_handler(self._model_loader, [])
        tmp = java_handler(tmp.file, [])
        tmp = java_handler(tmp.path, [path])
        tmp = java_handler(tmp.key, [key])
        model = java_handler(tmp.load, [])
        return self.wrapper(cast(self.java_class, model))
