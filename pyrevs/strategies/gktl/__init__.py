from pyrevs.database import Database
from .config import GKTLConfig
from .extension import GKTLDatabaseExtension
from .gktl import GKTL

__all__ = ["GKTL", "GKTLConfig", "GKTLDatabaseExtension", "load_database_extension"]


def load_database_extension(tdb: Database) -> GKTLDatabaseExtension:
    """A factory function to instanciate the extension from the database."""
    ext = GKTLDatabaseExtension()
    ext.initialize_from_database(tdb)
    return ext
