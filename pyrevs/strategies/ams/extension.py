"""An extension class for the AMS strategy."""

import gc
import json
import logging
from pathlib import Path
from typing import TypeVar
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyrevs.core import CoreDB
from pyrevs.database import Database
from pyrevs.database import StrategyDatabaseExtension
from pyrevs.trajectory import Trajectory
from .sql import AMSDB

_logger = logging.getLogger(__name__)

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class AMSDatabaseExtension(StrategyDatabaseExtension):
    """An extension class for the AMS strategy.

    Attributes:
        _nsplitting: maximum number of splitting iterations
        _ams_db: an instance of AMSDB, extending the SQL database
    """

    def __init__(self) -> None:
        self._nsplitting: int = -1
        self._ams_db: AMSDB | None = None
        self._init_ensemble_done: bool = False

    def initialize(self, nsplitting: int, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            nsplitting: maximum number of splitting iterations
            tdb: the core trajectory database
        """
        self._nsplitting = nsplitting
        self._tdb = tdb
        self._ams_db = AMSDB(self._req_tdb_db().engine())

    def initialize_from_database(self, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            tdb: the core trajectory database
        """
        self._tdb = tdb
        self._ams_db = AMSDB(self._req_tdb_db().engine())
        self.deserialize()

    def serialize(self) -> None:
        """Serialize the extension."""
        spath = Path(self._tdb.name()) / "ams_metadata.json"
        data = {
            "nsplitting": self._nsplitting,
            "init_ensemble_done": self._init_ensemble_done,
        }
        with spath.open("w") as f:
            json.dump(data, f, indent=2)

    def deserialize(self) -> None:
        """Serialize the extension."""
        spath = Path(self._tdb.name()) / "ams_metadata.json"
        with spath.open("r") as f:
            data = json.load(f)
        self._nsplitting = data["nsplitting"]
        self._init_ensemble_done = data["init_ensemble_done"]

    def _req_ams_db(self) -> AMSDB:
        if self._ams_db is None:
            err_msg = "AMSDB has not been initialized ! Call initialize()"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._ams_db

    def _req_tdb_db(self) -> CoreDB:
        pool_db = self._tdb.get_pool_db()
        if pool_db is None:
            err_msg = "CoreDB has not been initialized ! Database is not ready"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return pool_db

    def k_split(self) -> int:
        """Get the current splitting iteration index.

        The current splitting iteration index is equal to the
        ksplit + bias (number of branching event in the last iteration)
        entries of last entry in the SQL db table

        Returns:
            Internal splitting iteration index
        """
        return self._req_ams_db().get_k_split()

    def init_ensemble_done(self) -> bool:
        """Get the initial ensemble status flag.

        Returns:
            the flag indicating that the initial ensemble is finished
        """
        return self._init_ensemble_done

    def set_init_ensemble_flag(self, status: bool) -> None:
        """Change the initial ensemble status flag.

        Args:
            status: the new status
        """
        self._init_ensemble_done = status
        if self._tdb.to_disk():
            self.serialize()

    def get_ongoing(self) -> list[int] | None:
        """Get the list of ongoing trajectories if any.

        Returns:
            Either a list trajectories or None if nothing was left to do
        """
        return self._req_ams_db().get_ongoing()

    def weights(self) -> npt.NDArray[np.number]:
        """Splitting iterations weights."""
        return self._req_ams_db().get_weights()

    def mark_last_splitting_iteration_as_done(self) -> None:
        """Flag the last splitting iteration as done."""
        self._req_ams_db().mark_last_iteration_as_completed()

    def get_iteration_count(self) -> int:
        """Get the number of splitting iteration stored.

        Returns:
            The length of the SplittingIterations table
        """
        return self._req_ams_db().get_iteration_count()

    def append_splitting_iteration_data(
        self,
        ksplit: int,
        bias: int,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Append a set of splitting data to internal list.

        Args:
            ksplit : The splitting iteration index
            bias : The number of restarted trajectories, also ref. to as bias
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart (ancestors)
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values

        Raises:
            ValueError if the provided ksplit is incompatible with the db state
        """
        # Compute the weight of the ensemble at the current iteration
        # Insert 1.0 at the front of the weight array
        weights = np.insert(self._req_ams_db().get_weights(), 0, 1.0)
        new_weight = weights[-1] * (1.0 - float(bias) / float(self._tdb.n_traj()))

        # Check the splitting iteration index. If the incoming split is not
        # equal to the one in the database, something is wrong.
        if ksplit != self.k_split():
            self._req_ams_db().dump_file_json()
            err_msg = f"Attempting to add splitting iteration with splitting index {ksplit} \
                    incompatible with the last entry of the database {self.k_split()} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        # Check that the new min of maxes is larger than
        # at the previous step
        self._req_ams_db().check_new_min_of_maxes(min_max[0])

        self._req_ams_db().add_splitting_data(ksplit, bias, new_weight, discarded_ids, ancestor_ids, min_vals, min_max)

    def update_splitting_iteration_data(
        self,
        ksplit: int,
        bias: int,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Update the last set of splitting data to internal list.

        Args:
            ksplit : The splitting iteration index
            bias : The number of restarted trajectories, also ref. to as bias
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart (ancestors)
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values

        Raises:
            ValueError if the provided ksplit is incompatible with the db state
        """
        # Compute the weight of the ensemble at the current iteration
        # Insert 1.0 at the front of the weight array
        weights = np.insert(self._req_ams_db().get_weights(), 0, 1.0)
        new_weight = weights[-1] * (1.0 - bias / self._tdb.n_traj())

        # Check the splitting iteration index. If the incoming split is not
        # equal to the one in the database, something is wrong.
        if (ksplit + bias) != self.k_split():
            self._req_ams_db().dump_file_json()
            err_msg = f"Attempting to update splitting iteration with splitting index {ksplit + bias} \
                    incompatible with the last entry of the database {self.k_split()} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        self._req_ams_db().update_splitting_data(
            ksplit, bias, new_weight, discarded_ids, ancestor_ids, min_vals, min_max
        )

    def update_trajectories_weights(self) -> None:
        """Update the weights of all the trajectories.

        Using the the current splitting iteration weight.
        """
        tweight = self.weights()[-1]
        for t in self._tdb.traj_list():
            t.set_weight(float(tweight))
            if self._tdb.to_disk():
                self._req_tdb_db().update_trajectory_weight(t.id(), float(tweight))

        # Update the diagDB in the core database
        self._tdb.update_diagnostic_weights(tweight)

    def get_event_probability(self) -> float:
        """Return the event probability."""
        if self._tdb.count_terminated_traj() < self._tdb.n_traj():
            return 0.0

        # Insert a first element to the weight array
        weights = np.insert(self._req_ams_db().get_weights(), 0, 1.0)
        biases = self._req_ams_db().get_biases()

        w = self._tdb.n_traj() * weights[-1]
        for i in range(biases.shape[0]):
            w += biases[i] * weights[i]

        return float(self._tdb.count_converged_traj() * weights[-1] / w)

    def plot_min_max_span(self, fname: str | None = None) -> None:
        """Plot the evolution of the ensemble min/max during iterations."""
        pltfile = fname if fname else Path(self._tdb.name()).stem + "_minmax.png"

        plt.figure(figsize=(6, 4))

        min_max_data = self._req_ams_db().get_minmax()
        plt.plot(min_max_data[:, 0], min_max_data[:, 1], linewidth=1.0, label="min of maxes")
        plt.plot(min_max_data[:, 0], min_max_data[:, 2], linewidth=1.0, label="max of maxes")
        plt.grid(linestyle="dotted")
        ax = plt.gca()
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, np.max(min_max_data[:, 0]))
        ax.legend()
        plt.tight_layout()
        plt.savefig(pltfile, dpi=300)
        plt.clf()
        plt.close()

    def _get_location_and_indices_at_k(self, k_in: int) -> list[tuple[str, int]]:
        """Return the location and indices of active trajectory at k_in.

        Location here can be either 'active' or "archive" depending on
        whether the trajectory we are interested in is still in the current
        active list or in the archived list.

        Args:
            k_in : the index of the splitting iteration

        Returns:
            A list of tuple with the location and index of each trajectory
            active at iteration k
        """
        # Initialize active @k list with current active list
        # For now handle tuple with (active/archived, index)
        # The actual trajectory list will be filled later
        active_list_index = [("active", i) for i in range(self._tdb.n_traj())]

        # Traverse in reverse the splitting iteration table
        idx_in_archive = self._tdb.archived_traj_list_len() - 1
        for k in range(self._req_ams_db().get_iteration_count() - 1, k_in - 1, -1):
            splitting_data = self._req_ams_db().fetch_splitting_data(k)
            if splitting_data:
                _, nbranch, _, discarded, _, _, _, status = splitting_data
                if status == "locked":
                    continue
                for discarded_idx in discarded:
                    for i in range(idx_in_archive, idx_in_archive - nbranch, -1):
                        if self._tdb.archived_traj_list()[i].id() == discarded_idx:
                            active_list_index[discarded_idx] = ("archive", i)
            idx_in_archive = idx_in_archive - nbranch

        return active_list_index

    def get_trajectory_active_at_k(self, k_in: int) -> list[Trajectory[T_Noise, T_State]]:
        """Return the list of trajectory active at a given splitting iteration.

        To explore the ensemble evolution during splitting iterations, it is
        useful to reconstruct the list of active trajectories at the beginning
        of any given splitting iteration.

        Note that k here is not the splitting index, but the iteration index.
        Since more than one child can be spawned at each splitting iteration,
        the two might differ.

        Args:
            k_in : the index of the splitting iteration

        Returns:
            The list of trajectories active at the beginning of iteration k
        """
        # Check that the requested index is available in the database
        if k_in >= self._req_ams_db().get_iteration_count():
            err_msg = (
                f"Attempting to read splitting iteration {k_in} data"
                f"larger than stored data {self._req_ams_db().get_iteration_count()}"
            )
            _logger.error(err_msg)
            raise ValueError(err_msg)

        # Check that archived trajectories are stored
        if self._tdb.archived_traj_list_len() == 0:
            err_msg = "Cannot reconstruct active set without stored archives !"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        # First get the location and indices of the trajectories
        # active at iteration k
        active_list_index = self._get_location_and_indices_at_k(k_in)

        # Retrieve the trajectories from the active/archived lists
        active_list_at_k = []
        for location, idx in active_list_index:
            if location == "active":
                active_list_at_k.append(self._tdb.traj_list()[idx])
            elif location == "archive":
                active_list_at_k.append(self._tdb.archived_traj_list()[idx])

        return active_list_at_k

    def __del__(self) -> None:
        """Destructor of the AMS extension.

        Force deletion of SQL accesses for Windows.
        """
        if hasattr(self, "_tdb"):
            del self._tdb
        if hasattr(self, "_ams_db"):
            del self._ams_db
        gc.collect()
