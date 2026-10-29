"""Defines the generic interface for sampling strategies."""

from __future__ import annotations
import datetime as dt
import logging
from abc import ABC
from abc import abstractmethod
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeVar
from typing import final

if TYPE_CHECKING:
    from collections.abc import Callable
    from pyrevs.core import Config
    from pyrevs.database import Database

_logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseSamplingStrategy")


class BaseSamplingStrategy(ABC):
    """An interface for all rare-event algorithms.

    Define the common interface for sampling strategies within
    the sampler object.

    A registry is used to store all available strategies.
    It is managed using a decorator and entry_points.

    Subclasses must implement the following methods:
    - :meth:`sample`
    - :meth:`out_of_time`
    """

    # Registry, loaded on first use
    _registry: ClassVar[dict[str, type[BaseSamplingStrategy]]] = {}
    _strategies_loaded: ClassVar[bool] = False

    @classmethod
    def _load_strategies(cls) -> None:
        """Load all available strategies."""
        if cls._strategies_loaded:
            return

        eps = entry_points(group="pyrevs.strategies")
        for ep in eps:
            try:
                ep.load()
            except (ImportError, AttributeError) as exc:  # noqa: PERF203
                wrn_msg = f"Failed to load strategy {ep.name}: {exc}"
                _logger.warning(wrn_msg)

        cls._strategies_loaded = True

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Register a new strategy.

        Args:
            name: the strategy name
        """
        key = name.lower()

        def decorator(subclass: type[T]) -> type[T]:
            if key in cls._registry:
                err_msg = f"Strategy {key} already registered"
                raise ValueError(err_msg)
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> BaseSamplingStrategy:
        """Instantiate a strategy out of the registry.

        Args:
            name: the strategy name
            *args: positional arguments
            **kwargs: keyword arguments
        """
        cls._load_strategies()
        key = name.lower()
        try:
            return cls._registry[key](*args, **kwargs)
        except KeyError as err:
            err_msg = f"Unknown strategy type: {key}"
            raise ValueError(err_msg) from err

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return list of registered strategy names."""
        cls._load_strategies()
        return sorted(cls._registry.keys())

    # Time management uses UTC date
    _start_date: dt.datetime
    _end_date: dt.datetime
    _min_remaining_time: float
    _MIN_REMAINING_TIME_RATIO: ClassVar[float] = 0.05

    @final
    def sample(self, database: Database, walltime: float, plot_diags: bool) -> None:
        """Run the sampling lifecycle.

        This method handles walltime bookkeeping and delegates the
        algorithm implementation to ``_execute_sampling``.

        Subclasses must implement ``_execute_sampling`` and should
        regularly check ``out_of_time()`` to terminate gracefully.

        Args:
            database: Database used for storing results.
            walltime: Maximum allowed runtime in seconds.
            plot_diags: Whether to enable diagnostic plotting.
        """
        self._start_date = self._now()
        self._end_date = self._start_date + dt.timedelta(seconds=walltime)
        self._min_remaining_time = self._MIN_REMAINING_TIME_RATIO * walltime
        self._execute_sampling(database, plot_diags)

    def _now(self) -> dt.datetime:
        """Return the current time."""
        return dt.datetime.now(tz=dt.timezone.utc)

    def remaining_time(self) -> float:
        """Return the remaining wallclock time."""
        if not hasattr(self, "_end_date"):
            err_msg = "Sampling has not been started. Call 'sample()' first."
            raise RuntimeError(err_msg)
        return (self._end_date - self._now()).total_seconds()

    def out_of_time(self) -> bool:
        """Return true if insufficient walltime remains."""
        return self.remaining_time() <= self._min_remaining_time

    def elapsed_time(self) -> float:
        """Return the elapsed wallclock time."""
        if not hasattr(self, "_start_date"):
            err_msg = "Sampling has not been started. Call 'sample()' first."
            raise RuntimeError(err_msg)
        return (self._now() - self._start_date).total_seconds()

    @abstractmethod
    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Implement the core sampling algorithm.

        This method is called by :meth:`sample` after time bookkeeping
        has been initialized. Concrete implementations should
        implement the actual rare event sampling algorithm.

        Implementations should:
        - Regularly call :meth:`out_of_time` to respect walltime limits
        - Store intermediate and final results in ``database``
        - Optionally produce diagnostics if ``plot_diags`` is True

        Args:
            database: Database used for storing results.
            plot_diags: Whether to enable diagnostic plotting.

        Notes:
            This method should terminate gracefully when time is exhausted.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
        """Initialize the schema of the database."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}()"
