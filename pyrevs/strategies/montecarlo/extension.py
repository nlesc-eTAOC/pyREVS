"""An extension class for the MonteCarlo strategy."""

import logging
from typing import TypeVar
from pyrevs.database import Database
from pyrevs.database import StrategyDatabaseExtension

_logger = logging.getLogger(__name__)

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class MCDatabaseExtension(StrategyDatabaseExtension):
    """An extension class for the MonteCarlo strategy.

    The Monte-Carlo strategy has minimal extension requirements.
    It mostly provides a concrete implementation of the
    event probability computation method.

    It only holds a reference to the core trajectory database.

    Attributes:
        _tdb: the core trajectory database
    """

    def initialize(self, tdb: Database) -> None:
        """Initialize the MC database extension.

        Args:
            tdb: the core trajectory database
        """
        self._tdb = tdb

    def initialize_from_database(self, tdb: Database) -> None:
        """Initialize the MC database extension.

        Args:
            tdb: the core trajectory database
        """

    def serialize(self) -> None:
        """Serialize the extension."""

    def deserialize(self) -> None:
        """Serialize the extension."""

    def get_event_probability(self) -> float:
        """Return the event probability."""
        return self._tdb.count_converged_traj() / self._tdb.n_traj()
