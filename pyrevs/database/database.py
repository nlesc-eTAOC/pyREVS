"""A database class for the pyREVS sampler."""

from __future__ import annotations
import copy
import datetime as dt
import importlib
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import toml
from pyrevs.core import Config
from pyrevs.core import CoreDB
from pyrevs.diagnostics import DiagDB
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.trajectory.trajectory import form_trajectory_id
from pyrevs.utils.utils import get_module_local_import
from pyrevs.utils.xmlutils import new_element
from pyrevs.utils.xmlutils import xml_to_dict
from .config import DatabaseConfig

if TYPE_CHECKING:
    from pyrevs.core import ForwardModelBaseClass
    from .extension import StrategyDatabaseExtension

_logger = logging.getLogger(__name__)

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


@dataclass
class DatabaseCoreSpec:
    """A dataclass for the database core specification."""

    ntraj: int
    strategy: str
    deterministic: bool
    diag_configs: dict[str, Config] | None


class Database(Generic[T_Noise, T_State]):
    """A database class for the pyREVS sampler.

    The database class is a container for
    all the trajectory (and splitting data). When the
    user provides a path to store the database, a local folder is
    created holding a number of readable files, any output
    from the model and SQL files used to lock/release
    trajectories as the sampling algorithm proceeds.

    The readable files are currently in an XML format.

    A database can be loaded independently from the sampling
    algorithm and used for post-processing.

    Attributes:
        _fmodel_t: the forward model type
        _database_cfg: the database configuration dataclass
        _parameters: the dictionary of parameters
        _trajs_db: the list of trajectories
    """

    def __init__(
        self,
        fmodel_t: type[ForwardModelBaseClass[T_Noise, T_State]],
        config: Config,
        read_only: bool = True,
    ) -> None:
        """Initialize a pyREVS database.

        Initialize pyREVS database object, bare in-memory or on-disk.

        On-disk database trigger if a path is provided in the
        parameters dictionary. The user can chose to not append/override
        the existing database in which case the existing path
        will be copied to a new random name.

        Args:
            fmodel_t: the forward model type
            config: a Config object with the full configuration
            read_only: the database access mode
        """
        # Access mode
        self._read_only = read_only

        # Metadata and immutable parameters
        self._creation_date = dt.datetime.now(tz=dt.timezone.utc)
        self._version = version("pyrevs")
        self._name = "pyREVS_" + fmodel_t.name()
        self._strategy = "unknown"
        self._fmodel_t = fmodel_t
        self._config = config
        self._database_cfg = config.load(DatabaseConfig)
        self._model_params = config.section_dict("model")

        # Trajectory ensemble parameters
        self._deterministic = False
        self._ntraj: int = -1

        # Database format/storage parameters
        # Update the database name and create an absolute path
        if self._database_cfg.path:
            if self._database_cfg.format != "XML":
                err_msg = f"Unsupported pyREVS database format: {self._database_cfg.format} !"
                _logger.error(err_msg)
                raise ValueError(err_msg)
            self._name = f"{self._database_cfg.path}"
            self._abs_path: Path = Path.cwd() / self._name

        # Trajectory ensembles: in-memory
        # - one for active trajectories
        # - one for archived (discarded) trajectories
        self._trajs_db: list[Trajectory[T_Noise, T_State]] = []
        self._archived_trajs_db: list[Trajectory[T_Noise, T_State]] = []

        # Trajectory ensemble: persistent SQL database
        self._sql_name: str = ""
        self._sql_db: CoreDB | None = None

        # Diagnostics
        self._diag_configs: dict[str, Config] | None = None

        # Strategy-specific database extension
        self._strategy_extension: StrategyDatabaseExtension | None = None

    @classmethod
    def create(
        cls,
        fmodel_t: type[ForwardModelBaseClass[T_Noise, T_State]],
        config: Config,
    ) -> Database[T_Noise, T_State]:
        """Create a new pyREVS database.

        If no path is provided in the TOML configuration,
        the database is created in-memory and this is equivalent
        to calling the constructor.

        Args:
            fmodel_t: the forward model type
            config: a Config object with the full configuration

        Return:
            a pyREVS database object
        """
        db = cls(fmodel_t, config, read_only=False)
        if not db.to_disk():
            db.init_traj_pool()
            return db

        db._setup_folder()
        db._write_metadata()
        db._serialize_fmodel()
        db.init_traj_pool()

        return db

    @classmethod
    def load(cls, a_path: Path, read_only: bool = True) -> Database[T_Noise, T_State]:
        """Instantiate a pyREVS database from disk.

        Args:
            a_path: the path to the database
            read_only: the database access mode

        Return:
            a pyREVS database object
        """
        if not a_path.exists():
            err_msg = f"Database {a_path} does not exist !"
            _logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        # Load Config
        cfg_path = a_path / "input_params.toml"
        if not cfg_path.exists():
            err_msg = f"Database {cfg_path} does not exist !"
            _logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        with cfg_path.open("r") as f:
            config = Config(toml.load(f))

        # Load picked forward model
        model_file = Path(a_path / "fmodel.pkl")
        if not model_file.exists():
            err_msg = f"Picked forward model {model_file} is missing from the database !"
            _logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        with model_file.open("rb") as f:
            path = cloudpickle.load(f)
            module_name, cls_name = path.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            fmodel_t = getattr(mod, cls_name)

        db = cls(fmodel_t, config, read_only)
        db._load_metadata()
        db.init_traj_pool()

        return db

    def _setup_folder(self) -> None:
        """Initialize the trajectory database folders on disk."""
        if self.to_disk():
            # Create the top-level folder
            Path(self._name).mkdir()

            # Empty trajectories subfolder
            Path(self._abs_path / "trajectories").mkdir(parents=True)

    def _serialize_fmodel(self) -> None:
        # Serialize the model
        # We need to pickle by value the local modules
        # which might not be available if we move the database
        # Note: only one import depth is handled at this point, we might
        #       want to make this recursive in the future
        model_file = Path(self._abs_path / "fmodel.pkl")
        cloudpickle.register_pickle_by_value(sys.modules[self._fmodel_t.__module__])
        for mods in get_module_local_import(self._fmodel_t.__module__):
            cloudpickle.register_pickle_by_value(sys.modules[mods])
        with model_file.open("wb") as f:
            cloudpickle.dump(self._fmodel_t.__module__ + "." + self._fmodel_t.__name__, f)

    def _write_metadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._database_cfg.format == "XML":
            header_file = self.header_file()
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyREVS_version", version("pyrevs")))
            mdata.append(new_element("date", self._creation_date))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            mdata.append(new_element("strategy", self._strategy))
            mdata.append(new_element("ntraj", self.n_traj()))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(header_file)
        else:
            err_msg = f"Unsupported pyREVS database format: {self._database_cfg.format} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def _load_metadata(self) -> None:
        """Read the database Metadata from the header."""
        if self.to_disk():
            if self._database_cfg.format == "XML":
                tree = ET.parse(self.header_file())
                root = tree.getroot()
                mdata = root.find("metadata")
                datafromxml = xml_to_dict(mdata)
                self._ntraj = datafromxml["ntraj"]
                self._strategy = datafromxml["strategy"]
                self._version = datafromxml["pyREVS_version"]
                if self._version != version("pyrevs"):
                    warn_msg = f"Database pyREVS version {self._version} is different from {version('pyrevs')}"
                    _logger.warning(warn_msg)
                self._creation_date = datafromxml["date"]
                db_model = datafromxml["model_t"]
                if db_model != self._fmodel_t.name():
                    err_msg = f"Database model {db_model} is different from call {self._fmodel_t.name()}"
                    _logger.error(err_msg)
                    raise RuntimeError(err_msg)
            else:
                err_msg = f"Unsupported pyREVS database format: {self._database_cfg.format} !"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def initialize_core_state(self, spec: DatabaseCoreSpec) -> None:
        """Initialize the core state of the database.

        From a specification dataclass. It also updates the metadata
        on disk if the database is on-disk.

        Args:
            spec: the database core specification
        """
        self._ntraj = spec.ntraj
        self._strategy = spec.strategy
        self._deterministic = spec.deterministic
        self._diag_configs = spec.diag_configs

        self.ping_diag_database()

        if self.to_disk():
            self._write_metadata()

    def init_traj_pool(self) -> None:
        """Initialize the trajectory pool."""
        # If an in-memory database is requested, a temporary (hidden)
        # sql database is still created
        if self.to_disk():
            self._sql_name = f"{self._name}/trajPool.db"
        else:
            self._sql_name = f".sqldb_pyrevs_{np.random.default_rng().integers(0, 999999):06d}.db"

        if self._read_only:
            self._sql_db = CoreDB(self.pool_file(), ro_mode=True)
        else:
            self._sql_db = CoreDB(self.pool_file())

    def attach_extension(self, ext: StrategyDatabaseExtension) -> None:
        """Attach an extension to the database."""
        self._strategy_extension = ext
        if self.to_disk():
            self._strategy_extension.serialize()

    def extension(self) -> StrategyDatabaseExtension:
        """Return the extension attached to the database."""
        if self._strategy_extension is None:
            err_msg = "No strategy extension attached to the database !"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._strategy_extension

    def init_active_ensemble(self) -> None:
        """Initialize the requested number of trajectories."""
        traj_cfg = self._config.load(TrajectoryConfig)
        traj_cfg.validate()
        for n in range(self.n_traj()):
            if not self._require_pool_db().check_trajectory_exist(n):
                workdir = Path(self._abs_path / f"trajectories/{form_trajectory_id(n)}") if self.to_disk() else None
                t: Trajectory[T_Noise, T_State] = Trajectory(
                    traj_id=n,
                    weight=1.0,
                    fmodel_t=self._fmodel_t,
                    traj_cfg=traj_cfg,
                    diag_configs=self._diag_configs,
                    model_params=self._model_params,
                    workdir=workdir,
                    deterministic=self._deterministic,
                )
                self.append_traj(t, True)

    def save_trajectory(self, traj: Trajectory[T_Noise, T_State]) -> None:
        """Save a trajectory to disk in the database.

        Args:
            traj: the trajectory to save
        """
        if not self.to_disk():
            return

        traj.store()

    def load_data(self, load_archived_trajectories: bool = False) -> None:
        """Load data stored into the database.

        The initialization of the database only populate the metadata
        but not the full trajectories data.

        Args:
            load_archived_trajectories: whether to load archived trajectories
        """
        if not self.to_disk():
            return

        # Counter for number of trajectory loaded
        n_traj_restored = 0
        n_traj_initialized = 0

        load_frozen = self._read_only
        traj_cfg = self._config.load(TrajectoryConfig)
        traj_cfg.validate()

        ntraj_in_db = self._require_pool_db().get_trajectory_count()
        for n in range(ntraj_in_db):
            checkpath, metadata = self._require_pool_db().fetch_trajectory(n)
            traj_checkfile = Path(self._abs_path) / checkpath
            workdir = Path(self._abs_path / f"trajectories/{traj_checkfile.stem}")
            if traj_checkfile.exists():
                n_traj_restored += 1
                self.append_traj(
                    Trajectory.restore_from_checkfile(
                        traj_checkfile,
                        metadata,
                        fmodel_t=self._fmodel_t,
                        traj_cfg=traj_cfg,
                        diag_configs=self._diag_configs,
                        model_params=self._model_params,
                        workdir=workdir,
                        frozen=load_frozen,
                    ),
                    False,
                )
            else:
                n_traj_initialized += 1
                self.append_traj(
                    Trajectory.init_from_metadata(
                        metadata,
                        fmodel_t=self._fmodel_t,
                        traj_cfg=traj_cfg,
                        diag_configs=self._diag_configs,
                        model_params=self._model_params,
                        workdir=workdir,
                    ),
                    False,
                )

        if n_traj_restored > 0:
            inf_msg = f"{n_traj_restored} active trajectories loaded"
            _logger.info(inf_msg)
            inf_msg = f"{n_traj_initialized} active trajectories initialized"
            _logger.info(inf_msg)

        # Load the archived trajectories if requested.
        # Those are loaded as 'frozen', i.e. the internal model
        # is not available and advance function disabled.
        if load_archived_trajectories:
            self.load_archived_trajectories()

        self.info()

    def load_archived_trajectories(self) -> None:
        """Load the archived trajectories data."""
        if not self.to_disk():
            return

        n_traj_restored = 0
        traj_cfg = self._config.load(TrajectoryConfig)
        traj_cfg.validate()

        archived_ntraj_in_db = self._require_pool_db().get_archived_trajectory_count()
        for n in range(archived_ntraj_in_db):
            checkpath, metadata_str = self._require_pool_db().fetch_archived_trajectory(n)
            traj_checkfile = Path(self._abs_path) / checkpath
            workdir = Path(self._abs_path / f"trajectories/{traj_checkfile.stem}")
            if traj_checkfile.exists():
                n_traj_restored += 1
                self.append_archived_traj(
                    Trajectory.restore_from_checkfile(
                        traj_checkfile,
                        metadata_str,
                        fmodel_t=self._fmodel_t,
                        traj_cfg=traj_cfg,
                        diag_configs=self._diag_configs,
                        model_params=self._model_params,
                        workdir=workdir,
                        frozen=True,
                    ),
                    False,
                )

        inf_msg = f"{n_traj_restored} archived trajectories loaded"
        _logger.info(inf_msg)

    def name(self) -> str:
        """Accessor to DB name.

        Return:
            DB name
        """
        return self._name

    def strategy(self) -> str:
        """Accessor to DB strategy.

        Return:
            DB strategy
        """
        return self._strategy

    def to_disk(self) -> bool:
        """Check if the database is stored on disk."""
        return self._database_cfg.path is not None

    def append_traj(self, a_traj: Trajectory[T_Noise, T_State], update_db: bool) -> None:
        """Append a Trajectory to the internal list.

        Args:
            a_traj: the trajectory
            update_db: True to update the SQL DB content
        """
        # Also adds it to the SQL pool file.
        # and set the checkfile
        if self.to_disk():
            checkfile_str = f"./trajectories/{a_traj.idstr()}.xml"
            checkfile = Path(self._abs_path) / checkfile_str
            a_traj.set_checkfile(checkfile)
        else:
            checkfile_str = f"{a_traj.idstr()}.xml"
        if update_db:
            self._require_pool_db().add_trajectory(checkfile_str, a_traj.get_metadata())

        self._trajs_db.append(a_traj)

    def append_archived_traj(self, a_traj: Trajectory[T_Noise, T_State], update_db: bool) -> None:
        """Append an archived Trajectory to the internal list.

        Args:
            a_traj: the trajectory
            update_db: True to update the SQL DB content
        """
        checkfile_str = f"./trajectories/{a_traj.idstr()}.xml"
        checkfile = Path(self._abs_path) / checkfile_str
        a_traj.set_checkfile(checkfile)
        if update_db:
            self._require_pool_db().archive_trajectory(checkfile_str, a_traj.get_metadata())

        self._archived_trajs_db.append(a_traj)

    def traj_list(self) -> list[Trajectory[T_Noise, T_State]]:
        """Access to the trajectory list.

        Return:
            Trajectory list
        """
        return self._trajs_db

    def get_traj(self, idx: int) -> Trajectory[T_Noise, T_State]:
        """Access to a given trajectory.

        Args:
            idx: the index

        Return:
            Trajectory

        Raises:
            ValueError if idx is out of range
        """
        if idx < 0 or idx >= len(self._trajs_db):
            err_msg = f"Trying to access a non existing trajectory {idx} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        return self._trajs_db[idx]

    def overwrite_traj(self, idx: int, traj: Trajectory[T_Noise, T_State]) -> None:
        """Deep copy a trajectory into internal list.

        Args:
            idx: the index of the trajectory to override
            traj: the new trajectory

        Raises:
            ValueError if idx is out of range
        """
        if idx < 0 or idx >= len(self._trajs_db):
            err_msg = f"Trying to override a non existing trajectory {idx} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        self._trajs_db[idx] = copy.deepcopy(traj)

    def ping_diag_database(self) -> None:
        """Initialize the diagDB from the main process.

        To avoid race condition in creating the DB from workers,
        let it be initialized from the main process here.
        """
        if self._diag_configs is not None:
            ddb_path = self._abs_path / DiagDB.default_name() if self.to_disk() else Path(DiagDB.default_name())
            ddb = DiagDB(ddb_path.absolute().as_posix())
            ddb.close()

    def update_diagnostic_weights(self, tweight: float) -> None:
        """Update the weights of all the active trajectories."""
        if self._diag_configs is not None:
            ddb_path = self._abs_path / DiagDB.default_name() if self.to_disk() else Path(DiagDB.default_name())
            ddb = DiagDB(ddb_path.absolute().as_posix())
            ddb.update_all_active_weights(tweight)
            ddb.close()

    def header_file(self) -> str:
        """Helper returning the DB header file.

        Return:
            Header file
        """
        return f"{self._name}/header.xml"

    def pool_file(self) -> str:
        """Helper returning the DB trajectory pool file.

        Return:
            Pool file
        """
        return self._sql_name

    def get_pool_db(self) -> CoreDB | None:
        """Get the pool SQL database handle."""
        return self._sql_db

    def _require_pool_db(self) -> CoreDB:
        """Internal accessor to the SQL database handle.

        Raises:
            RuntimeError if the SQL database has not been initialized
        """
        if self._sql_db is None:
            err_msg = "The SQL database has not been initialized !"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._sql_db

    def is_empty(self) -> bool:
        """Check if list of trajectories is empty.

        Return:
            True if the list of trajectories is empty
        """
        return self._require_pool_db().get_trajectory_count() == 0

    def traj_list_len(self) -> int:
        """Length of the trajectory list.

        Return:
            Trajectory list length
        """
        return len(self._trajs_db)

    def archived_traj_list(self) -> list[Trajectory[T_Noise, T_State]]:
        """Access to the archived trajectory list.

        Return:
            Archived trajectory list
        """
        return self._archived_trajs_db

    def archived_traj_list_len(self) -> int:
        """Length of the archived trajectory list.

        Return:
            Trajectory list length
        """
        return len(self._archived_trajs_db)

    def archive_discarded(self) -> bool:
        """Return True if discarded trajectories should be archived."""
        return self._database_cfg.archive_discarded

    def update_traj_list(self, a_traj_list: list[Trajectory[T_Noise, T_State]]) -> None:
        """Overwrite the trajectory list.

        Args:
            a_traj_list: the new trajectory list
        """
        self._trajs_db = a_traj_list

    def archive_trajectory(self, traj: Trajectory[T_Noise, T_State]) -> None:
        """Archive a trajectory about to be discarded.

        Args:
            traj: the trajectory to archive
        """
        # A branched trajectory will be overwritten by the
        # newly generated one in-place in the _trajs_db list.
        self._archived_trajs_db.append(traj)

        # Update the list of archived trajectories in the SQL DB
        checkfile_str = (
            traj.get_checkfile().relative_to(self._abs_path).as_posix()
            if self.to_disk()
            else traj.get_checkfile().as_posix()
        )
        self._require_pool_db().archive_trajectory(checkfile_str, traj.get_metadata())

    def lock_trajectory(self, tid: int, allow_completed_lock: bool = False) -> bool:
        """Lock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            allow_completed_lock: True if the trajectory can be locked even if it is completed

        Return:
            True if no disk DB and the trajectory was locked

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        return self._require_pool_db().lock_trajectory(tid, allow_completed_lock)

    def unlock_trajectory(self, tid: int, has_terminated: bool) -> None:
        """Unlock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            has_terminated: True if the trajectory has terminated

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        if has_terminated:
            self._require_pool_db().mark_trajectory_as_completed(tid)
        else:
            self._require_pool_db().release_trajectory(tid)

    def update_trajectory(self, traj_id: int, traj: Trajectory[T_Noise, T_State]) -> None:
        """Update a trajectory file in the DB.

        Args:
            traj_id : The trajectory id
            traj : the trajectory to get the data from

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        checkfile_str = traj.get_checkfile().relative_to(self._abs_path).as_posix()
        self._require_pool_db().update_trajectory(traj_id, checkfile_str, traj.get_metadata())

    def n_traj(self) -> int:
        """Return the number of trajectory used for sampling run.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory pool.

        Return:
            number of trajectory
        """
        return self._ntraj

    def path(self) -> str | None:
        """Return the path to the database."""
        if self.to_disk():
            return self._abs_path.absolute().as_posix()
        return None

    def count_terminated_traj(self) -> int:
        """Return the number of trajectories that terminated."""
        return self._require_pool_db().get_terminated_trajectory_count()

    def count_converged_traj(self) -> int:
        """Return the number of trajectories that converged."""
        return self._require_pool_db().get_converged_trajectory_count()

    def all_converged(self) -> bool:
        """Check if all the trajectory converged."""
        return self.count_converged_traj() == self.n_traj()

    def count_computed_steps(self) -> int:
        """Return the total number of steps taken.

        This total count includes both the active and
        discarded trajectories.
        """
        return self._require_pool_db().get_total_computed_steps()

    def get_event_probability(self) -> float:
        """Return the event probability.

        Default to a Monte-Carlo event probability if
        no strategy extension is attached to the database.

        Return:
            the event probability
        """
        if self._strategy_extension is not None:
            return self._strategy_extension.get_event_probability()

        return self.count_converged_traj() / self.n_traj()

    def info(self) -> None:
        """Print database info to logger."""
        db_date_str = str(self._creation_date)
        pretty_line = "####################################################"
        inf_tbl = f"""
            {pretty_line}
            # pyREVS v{self._version:40s} #
            # Creation Date: {db_date_str:33s} #
            # Model: {self._fmodel_t.name():41s} #
            # Sampling strategy: {self._strategy:29s} #
            {pretty_line}
            # Requested # of traj: {self.n_traj():27} #
            # Number of 'Terminated' trajectories: {self.count_terminated_traj():11} #
            # Number of 'Converged' trajectories: {self.count_converged_traj():12} #
            # Current total number of steps: {self.count_computed_steps():17} #
            {pretty_line}
        """
        _logger.info(inf_tbl)

    def print_info(self) -> None:
        """Print database info to screen."""
        db_date_str = str(self._creation_date)
        pretty_line = "####################################################"
        inf_tbl = f"""
            {pretty_line}
            # pyREVS v{self._version:40s} #
            # Creation Date: {db_date_str:33s} #
            # Model: {self._fmodel_t.name():41s} #
            # Sampling strategy: {self._strategy:29s} #
            {pretty_line}
            # Requested # of traj: {self.n_traj():27} #
            # Number of 'Terminated' trajectories: {self.count_terminated_traj():11} #
            # Number of 'Converged' trajectories: {self.count_converged_traj():12} #
            # Current total number of steps: {self.count_computed_steps():17} #
            {pretty_line}
        """
        print(inf_tbl)  # noqa: T201

    def plot_score_functions(self, fname: str | None = None, plot_archived: bool = False) -> None:
        """Plot the score as function of time for all trajectories."""
        pltfile = fname if fname else Path(self._name).stem + "_scores.png"

        plt.figure(figsize=(10, 6))
        for t in self._trajs_db:
            plt.plot(t.get_time_array(), t.get_score_array(), linewidth=0.8)

        if plot_archived:
            for t in self._archived_trajs_db:
                plt.plot(t.get_time_array(), t.get_score_array(), linewidth=0.8)

        plt.xlabel(r"$Time$", fontsize="x-large")
        plt.xlim(left=0.0)
        plt.ylabel(r"$Score \; [-]$", fontsize="x-large")
        plt.xticks(fontsize="x-large")
        plt.yticks(fontsize="x-large")
        plt.grid(linestyle="dotted")
        plt.tight_layout()  # to fit everything in the prescribed area
        plt.savefig(pltfile, dpi=300)
        plt.clf()
        plt.close()

    def __del__(self) -> None:
        """Destructor of the db.

        Delete the hidden SQL database if we do not intend to keep
        the database around.
        """
        # Even if we plan to keep the SQL database around, force
        # deleting the SQL connection
        if hasattr(self, "_sql_db"):
            del self._sql_db
            # Remove the hidden db file
            if not self.to_disk():
                Path(self.pool_file()).unlink(missing_ok=True)

        if hasattr(self, "_strategy_extension"):
            del self._strategy_extension
