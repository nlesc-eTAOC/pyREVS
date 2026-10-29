"""A protocol database extension class for specific strategies."""

from typing import Protocol


class StrategyDatabaseExtension(Protocol):
    """A base extension class for specific strategies.

    The core database only holds data relative to the model trajectories,
    as well as accessor and convienience methods to extract trajectory
    data.
    This extension class aims at gathering strategy-specific
    data and methods that belong to a given sampling run database.
    For instance, the (T)AMS strategy requires keeping track of the
    weight of the ensemble at each iteration, as well as information
    about score levels or number of selection/mutations.

    The extension can store some data in-memory and the serialize/
    deserialize methods are used while checkpointing a sampling run.
    In addition, if the strategy requires long tables, one can use
    SQL to store the data (see the AMS strategy for example).

    Currently an empty type definition might be extended
    in the future for pickling purposes.
    """

    def serialize(self) -> None:
        """Serialize the extension."""
        raise NotImplementedError

    def deserialize(self) -> None:
        """Deserialize the extension."""
        raise NotImplementedError

    def get_event_probability(self) -> float:
        """Return the event probability."""
        raise NotImplementedError
