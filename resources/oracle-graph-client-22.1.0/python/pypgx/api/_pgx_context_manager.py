#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#


class PgxContextManager:
    """Base class that implements context manager for PGX objects.

    see https://docs.python.org/3/reference/datamodel.html#context-managers
    """

    def destroy(self) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def __enter__(self) -> "PgxContextManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Most objects will define a destroy() method, but some will define a close()
        # method instead.
        try:
            self.destroy()
        except NotImplementedError:
            self.close()
