.. highlight:: rst

.. _sec:extension

Extension
=========

`pyREVS` was designed to facilitate the implementation/testing of new rare-events sampling algorithms.
As such, sampling algorithms can be implemented as plugins, relying on `pyREVS` core data structures.
Here we will go over the few steps required to add a new sampling algorithm, AKA ``Strategy`` to
`pyREVS`.

Sampling algorithms are located in the ``pyrevs/strategies`` directory, each with its own
subdirectory. Strategies are viewed as plugins for `pyREVS`, i.e. the package will load these plugins at
runtime to make the sampling algorithms available to the user. To do enable the a new sampling
algorithm, let's update the ``pyproject.toml`` file (in the ``<pyREVS_root>`` directory),
specifically the ``entry-points`` section:

.. code-block:: toml

    # Add your own strategy in the block below
    [project.entry-points."pyrevs.strategies"]
    ams = "pyrevs.strategies.ams"
    montecarlo = "pyrevs.strategies.montecarlo"
    mynewstrategy = "pyrevs.strategies.mynewstrategy"

And let's initiate a new directory for a new sampling algorithm, e.g.:

.. code-block:: shell

    cd pyrevs/strategies
    mkdir mynewstrategy

At minima, your new sampling algorithm will require creating a new subclass of the
:py:class:`pyrevs.strategies.BaseSamplingStrategy` class (e.g. in ``mynewstrategy/mynewstrategy.py``),
an implementation of the :py:class:`pyrevs.database.StrategyDatabaseExtension` Protocol
(e.g. in ``mynewstrategy/extension.py``), and a configuration dataclass (e.g. in
``mynewstrategy/config.py``). Let's look at the concrete strategy implementation first:

.. code-block:: shell

    cd mynewstrategy

And using your favorite text editor (or an IDE), let's create a new ``mynewstrategy.py`` file with:

.. code-block:: python

    from pyrevs.core import Config
    from pyrevs.database import Database
    from pyrevs.strategies.base import BaseSamplingStrategy 

    @BaseSamplingStrategy.register("mynewstrategy")
    class MyNewStrategy(BaseSamplingStrategy):
        """An awesome new sampling strategy."""

        def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
            """..."""

        def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
            """..."""

The two methods are the only two ``abstractmethod`` in the ``BaseSamplingStrategy`` class,
and thus must be implemented by the new strategy. The first one effectively implements the algorithm,
while the second one is used to initialize strategy-specific database content. Developers should fitst
look at the ``montecarlo`` strategy files as an example of a very simple strategy to get familiar with the
structure of ``BaseSamplingStrategy``. The ``ams`` strategy is a more complex example of a strategy that
can be used as a template.

The ``BaseSamplingStrategy`` class will perform the time management of the sampling process such that concrete
implementations can relies on ``self.elapsed_time()`` and other base methods for time management.

The next ingredient is the ``extension.py`` file, which implements the
:py:class:`pyrevs.database.StrategyDatabaseExtension` Protocol. This protocol is used to define the
database extension class for a specific strategy. Again with your favorite editor:

.. code-block:: python

    from pyrevs.database import Database
    from pyrevs.database import StrategyDatabaseExtension

    class MyNewStrategyDatabaseExtension(StrategyDatabaseExtension):
        """An extension class for the MyNewStrategy strategy."""
        def serialize(self) -> None:
            """Serialize the extension."""

        def deserialize(self) -> None:
            """Deserialize the extension."""

        def get_event_probability(self) -> float:
            """Return the event probability."""

These three methods are needed, for writting/reading the extension to/from the database and
computing the event probability from the content of the core database (i.e. ensemble of trajectories).
That object should be initialized in the ``initialize_database_schema``  method and tranfered 
to the core database, e.g.:

.. code-block:: python

    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:

        self._db_ext = MyNewStrategyDatabaseExtension()
        self._db_ext.initialize(database)
        database.attach_extension(self._db_ext)

Once again, developers are encouraged to first look at the Monte-Carlo implementation as a baseline example
first, then the AMS one for a more complex one.

Next, the ``config.py`` is intended to hold the new strategy parameters, as a dataclass with
in-code documentation. Each parameter should be annotated as follows in order to be picked up the
``pyrevs_help`` command:

.. code-block:: python

    from dataclasses import dataclass
    from dataclasses import field
    from pyrevs.core import MergePolicy
    
    
    @dataclass(frozen=True)
    class MyNewStrategyConfig:
        """Configuration for the MyNew strategy configuration."""
    
        __section__ = "mynewstrategy"
        __merge_policy__ = MergePolicy.IMMUTABLE
    
        ntrajectories: int = field(
            default=-1,
            metadata={
                "doc": "Number of trajectories to generate",
            },
        )

The ``__section__`` class attribute is used to define the section header in the input TOML file,
and the ``__merge_policy__`` attribute is used to define allows or not updating input parameters
when restarting a simpling run from a database. By default, the strategy parameters are immutable
once the database is created, but this can be changed to ``MergePolicy.REPLACE`` if needed.

Finally, let's add a ``__init__.py`` to the new strategy directory in order to make a plugin:

.. code-block:: python

    from pyrevs.database import Database
    from .mynewstrategy import MyNewStrategy
    from .config import MyNewStrategyConfig
    from .extension import MyNewStrategyDatabaseExtension

    __all__ = [
        "MyNewStrategy",
        "MyNewStrategyConfig",
        "MyNewStrategyDatabaseExtension",
    ]

    def load_database_extension(tdb: Database) -> MyNewStrategyDatabaseExtension:
        """A factory function to instanciate the extension from the database."""
        return MyNewStrategyDatabaseExtension()

This is all that is needed to add a new sampling algorithm to `pyREVS`, well in addition to
implementating an actual sampling algorithm in the ``mynewstrategy/mynewstrategy.py`` file !

.. Note::
    For persistence, the algorithm-specific data should be stored in an SQL addition to
    the core database. Refers to the AMS strategy, specifically the sql.py file, for an example.
    While prototyping, one can simply store the data in-memory within the ``BaseSamplingStrategy`` class.
