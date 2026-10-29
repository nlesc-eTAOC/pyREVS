"""A base class for pyREVS SQL databases."""

from __future__ import annotations
import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from pyrevs.utils.utils import is_windows_os

if TYPE_CHECKING:
    from collections.abc import Generator
    from sqlalchemy import MetaData
    from sqlalchemy.engine import Engine

_logger = logging.getLogger(__name__)


class BaseSQLManager:
    """A base class to handle SQLAlchemy engine and session boilerplate."""

    def __init__(self, file_name: str, base_metadata: MetaData, in_memory: bool = False, ro_mode: bool = False) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
            base_metadata: the table metadata to initialize
            in_memory: a bool to trigger in-memory creation
            ro_mode: a bool to trigger read-only access to the database
        """
        self._file_name = "" if in_memory else file_name

        # URI mode requires absolute path
        file_path = Path(file_name).absolute().as_posix()
        if in_memory:
            self._engine = create_engine("sqlite:///:memory:", echo=False, connect_args={"timeout": 120.0})
        else:
            uri = f"sqlite:///file:{file_path}?mode=ro&uri=true" if ro_mode else f"sqlite:///{file_path}"
            self._engine = create_engine(uri, echo=False, connect_args={"timeout": 120.0})

        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._init_db(base_metadata)

    def _init_db(self, base_metadata: MetaData) -> None:
        """Initialize the tables of the file.

        Raises:
            RuntimeError : If a connection to the DB could not be acquired
        """
        try:
            base_metadata.create_all(self._engine)
        except SQLAlchemyError:
            _logger.exception("Failed to initialize DB schema")
            raise

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def name(self) -> str:
        """Access the DB file name.

        Returns:
            the database name, empty string if in-memory
        """
        return self._file_name

    def engine(self) -> Engine:
        """Access the DB engine.

        Returns:
            the database engine
        """
        return self._engine

    def close(self) -> None:
        """Dispose of the engine and clear connections."""
        if self._engine:
            self._engine.dispose()

    def __del__(self) -> None:
        """Explicit delete function.

        On windows, the SQL file might be locked: force garbage collection
        to release it. This is skipped on other platforms as it is costly.
        """
        del self._Session
        self.close()
        if is_windows_os():
            gc.collect()
