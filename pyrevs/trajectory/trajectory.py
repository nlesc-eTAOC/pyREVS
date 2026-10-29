"""A trajectory class for individual MCMC runs."""

from __future__ import annotations
import copy
import json
import logging
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from dataclasses import replace
from enum import Enum
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast
import numpy as np
import numpy.typing as npt
from pyrevs.core import Snapshot
from pyrevs.diagnostics import DiagDB
from pyrevs.diagnostics import DiagnosticPlugin
from pyrevs.diagnostics import diagnosticfactory
from pyrevs.utils.xmlutils import dict_to_xml
from pyrevs.utils.xmlutils import make_xml_snapshot
from pyrevs.utils.xmlutils import read_xml_snapshot

if TYPE_CHECKING:
    from pyrevs.core import Config
    from pyrevs.core import ForwardModelBaseClass
    from pyrevs.strategies.base import TerminationCriterion
    from .config import TrajectoryConfig

_logger = logging.getLogger(__name__)

# Tolerances used when checking the scores of replayed snapshots
REPLAY_RTOL = 1e-6
REPLAY_ATOL = 1e-12

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class TrajectoryStateType(Enum):
    """Enumeration of trajectory state types.

    Attributes:
        UNINIT : trajectory has not been initialized
        ONGOING : trajectory is running
        INTERRUPTED : trajectory has been interrupted (expected to continue later)
        TERMINATED : trajectory has been terminated
        CONVERGED : trajectory has converged
    """

    UNINIT = 0
    ONGOING = 1
    INTERRUPTED = 2
    TERMINATED = 3
    CONVERGED = 4


class WallTimeLimitError(Exception):
    """Exception for running into wall time limit."""


class UserInterruptError(Exception):
    """Exception for handling user-triggered interruption."""


def form_trajectory_id(n: int, nb: int = 0) -> str:
    """Helper to assemble a trajectory ID string.

    Args:
        n : trajectory index
        nb : number of branching

    Returns:
        trajectory ID
    """
    return f"traj{n:06}_{nb:04}"


def get_index_from_id(identity: str) -> tuple[int, int]:
    """Helper to get trajectory index from ID string.

    Args:
        identity : trajectory ID

    Returns:
        trajectory index and number of branching
    """
    return int(identity[-10:-5]), int(identity[-4:])


class Trajectory(Generic[T_Noise, T_State]):
    """A class defining a stochastic trajectory.

    Each trajectory is a MCMC simulation of the forward model.

    The trajectory class is a container for time-ordered snapshots.
    It contains an instance of the forward model, current and end times, and
    a list of the model snapshots. Note that the class uses a plain list of snapshots
    and not a more computationally efficient data structure such as a numpy array
    for convenience. It is assumed that the computational cost of running the sampling algorithm
    resides in the forward model and the overhead of the trajectory class is negligible.

    It also provide the forward model with the necessary context to advance in time,
    method to move forward in time, methods to save/load the trajectory to/from disk
    as well as accessor to the trajectory history (time, state, score, ...).

    The _computed_steps variable store the number of steps actually taken by the
    trajectory. It differs from the _step variable when a trajectory is branched
    from an ancestor.

    Attributes:
        _tid : the trajectory index
        _checkFile : the trajectory checkpoint file
        _workdir : the model working directory
        _score_max : the maximum score
        _snaps : a list of snapshots
        _tail : a list of known snapshots beyond the model current state, to be replayed
        _step : the current step counter
        _computed_steps : the number of steps explicitly advanced by the trajectory
        _t_cur : the current time
        _t_end : the end time
        _dt : the stochastic time step size
    """

    # Maximum number of Trajectory objects and history length
    # These are used in assembling the unique ID for each trajectory
    max_ntraj = 10000
    max_nbranch = 9999

    def __init__(
        self,
        traj_id: int,
        weight: float,
        fmodel_t: type[ForwardModelBaseClass[T_Noise, T_State]] | None,
        traj_cfg: TrajectoryConfig,
        diag_configs: dict[str, Config] | None = None,
        model_params: dict[str, Any] | None = None,
        workdir: Path | None = None,
        deterministic: bool = False,
        frozen: bool = False,
    ) -> None:
        """Initialize a trajectory.

        Args:
            traj_id: a int for the trajectory index
            weight: the trajectory weight in the ensemble
            fmodel_t: the forward model type
            traj_cfg: the trajectory configuration
            diag_configs: a dict with diagnostic configurations
            model_params: the model configuration
            workdir: an optional working directory
            deterministic: whether the trajectory is deterministic
            frozen: whether the trajectory is frozen (no fmodel)
        """
        # Stash away the configuration(s)
        self._traj_cfg = traj_cfg
        self._model_params = model_params
        self._deterministic = deterministic

        # The workdir is a runtime parameter, not saved in the chkfile.
        self._tid: int = traj_id
        self._workdir: Path = Path.cwd() if workdir is None else workdir
        self._score_max: float = -1.0e12
        self._has_converged: bool = False
        self._has_terminated: bool = False
        self._computed_steps: int = 0
        self._weight: float = weight

        # Sampling is expected to start at t = 0.0, but the forward model
        # itself can have a different internal starting point
        # or an entirely different time scale.
        self._step: int = 0
        self._t_cur: float = 0.0
        self._t_end: float = traj_cfg.end_time
        self._dt: float = traj_cfg.step_size

        # List of snapshots
        self._snaps: list[Snapshot[T_Noise, T_State]] = []

        # When using sparse state (e.g. upon restoring or branching), the snapshots
        # following the last available state might already be known. These form
        # the tail of the trajectory, replayed upon the next call to advance.
        self._tail: list[Snapshot[T_Noise, T_State]] = []

        # Keep track of the branching history for iterative
        # sampling algorithms
        self._branching_history: list[int] = []

        # For large models, the state may not be available at each snapshot due
        # to memory constraint (both in-memory and on-disk). Sparse state can
        # be specified. Finally, writing a chkfile to disk at each step might
        # incur a performance hit and is by default disabled.
        self._checkFile: Path = Path(f"{self.idstr()}.xml")

        # Each trajectory has its own instance of the forward model
        if frozen or fmodel_t is None:
            self._fmodel = None
        else:
            self._fmodel = fmodel_t(
                self._tid * (self.max_nbranch + 1) + self.get_nbranching(), deterministic, model_params, self._workdir
            )

        # Diagnostics
        # They are initialize upon the first call so that
        # the diagnostic database access is not pickled in the task
        self._has_diagnostics = diag_configs is not None
        self._diag_configs: dict[str, Config] | None = diag_configs
        self._diagplugins: list[DiagnosticPlugin] = []
        self._ddb: DiagDB | None = None
        self._initialized_diags = False

    def __getstate__(self) -> dict[str, Any]:
        """Return the trajectory state for pickling.

        The snapshots are stored column-wise, which is much faster
        to pickle than a list of snapshot objects.
        """
        state = self.__dict__.copy()
        snaps = state.pop("_snaps")
        state["_snaps"] = (
            [s.time for s in snaps],
            [s.score for s in snaps],
            [s.noise for s in snaps],
            [s.state for s in snaps],
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the trajectory from its pickled state.

        Args:
            state: the state returned by __getstate__
        """
        times, scores, noises, states = state.pop("_snaps")
        self.__dict__.update(state)
        self._snaps = [
            Snapshot(time=t, score=s, noise=n, state=x)
            for t, s, n, x in zip(times, scores, noises, states, strict=True)
        ]

    def set_checkfile(self, path: Path) -> None:
        """Setter of the trajectory checkFile.

        Args:
            path: the new checkFile
        """
        self._checkFile = path

    def set_workdir(self, path: Path) -> None:
        """Setter of the trajectory working directory.

        And propagate the workdir to the forward model.

        Args:
            path: the new working directory
        """
        self._workdir = path
        if self._fmodel is not None:
            self._fmodel.set_workdir(path)

    def get_workdir(self) -> Path:
        """Get the trajectory working directory.

        Returns:
            the working directory
        """
        return self._workdir

    def id(self) -> int:
        """Return trajectory index.

        This is the index of the trajectory in the ensemble.

        Returns:
            the trajectory id
        """
        return self._tid

    def unique_id(self) -> int:
        """Return trajectory unique Id.

        Combining the index and the number of branching events.
        This makes a unique ID.

        Returns:
            the trajectory unique id
        """
        return self._tid * (self.max_nbranch + 1) + self.get_nbranching()

    def idstr(self) -> str:
        """Return trajectory Id as a padded string.

        Returns:
            the trajectory id as a string
        """
        return form_trajectory_id(self._tid, self.get_nbranching())

    def advance(
        self,
        termination_criteria: list[TerminationCriterion] | None = None,
        nstep_end: int = -1,
        t_end: float = -1.0,
        walltime: float = 1.0e12,
    ) -> None:
        """Advance the trajectory to a prescribed end time.

        This is the main time loop of the trajectory object.
        Unless specified otherwise, the trajectory will advance until
        the end time is reached or the model has converged.

        If the walltime limit is reached, a WallTimeLimitError exception is raised.
        If a 'pyrevs_stop' file is found, a UserInterruptError exception is raised.
        Note that this exception is treated as a warning not an error by the
        pyREVS workers.

        Args:
            nstep_end: the number of steps to advance
            t_end: the end time of the advance
            walltime: a walltime limit to advance the model to t_end
            termination_criteria: a list of termination criterion

        Returns:
            None

        Raises:
            WallTimeLimitError: if the walltime limit is reached
            UserInterruptError: if a 'pyrevs_stop' file is found
            RuntimeError: if the model advance run into a problem
        """
        # Check if the trajectory is frozen
        if not self._fmodel:
            err_msg = f"Trajectory {self.idstr()} is frozen, without forward model. Advance() deactivated."
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        start_time = time.monotonic()
        end_time = self._calculate_end_time(t_end)

        # Termination from sampling strategy termination criteria
        if termination_criteria is None:
            termination_criteria = []

        # or from runtime arguments (end time, nstep_end)
        # (A previous call to advance with other arguments might have switched the boolean)
        # An interruption criterion already met (e.g. when resuming a run) prevents
        # taking any step.
        self._has_terminated, interrupt = self._check_criteria(termination_criteria, end_time, nstep_end)

        while not (self._has_terminated or self._has_converged or interrupt):
            # Do a single model step
            score = self._one_step()

            # Check for termination/convergence/interruption
            self._has_converged = self._fmodel.check_convergence(
                self._step, self._t_cur, score, self._traj_cfg.targetscore
            )
            self._has_terminated, interrupt = self._check_criteria(termination_criteria, end_time, nstep_end)

            # Handle diagnostics
            self._update_diagnostics()

            # Check timeout before continuing
            if (time.monotonic() - start_time) >= walltime:
                interrupt = True

            # Check for user interruption
            if self.interrupt_trigger():
                interrupt = True

        # Wrap-up the advance function
        self._terminate_advance(start_time, walltime)

    def _terminate_advance(self, start_time: float, walltime: float) -> None:
        """Wrap-up the advance function.

        Args:
            start_time: the start time of the advance
            walltime: the walltime limit

        Returns:
            None

        Raises:
            WallTimeLimitError: if the walltime limit is reached
            UserInterruptError: if a 'pyrevs_stop' file is found
        """
        # If the model has converged, terminate
        if self._has_converged:
            self._has_terminated = True

        if self._has_terminated and self._fmodel:
            self._fmodel.clear()

        # Clear the diagnostic
        if self._ddb is not None and self._initialized_diags:
            self._ddb.close()
            self._ddb = None
            self._diagplugins = []
            self._initialized_diags = False

        if self.interrupt_trigger():
            warn_msg = f"{self.idstr()} interrupt during advance()"
            _logger.warning(warn_msg)
            raise UserInterruptError(warn_msg)

        if (time.monotonic() - start_time) >= walltime:
            warn_msg = f"{self.idstr()} ran out of time in advance()"
            _logger.warning(warn_msg)
            raise WallTimeLimitError(warn_msg)

    def interrupt_trigger(self) -> bool:
        """Check for user interrupt trigger.

        pyREVS can be interrupted cleanly by creating a
        pyrevs_stop file in the run folder.
        """
        return Path("./pyrevs_stop").exists()

    def _check_criteria(
        self, termination_criteria: list[TerminationCriterion], end_time: float, nstep_end: int
    ) -> tuple[bool, bool]:
        """Evaluate the termination criteria, each only once.

        Args:
            termination_criteria: a list of termination criterion
            end_time: the end time of the advance
            nstep_end: the number of steps to advance

        Returns:
            a tuple of bool: (should terminate, should interrupt)
        """
        if self._fmodel is None:
            return self._runtime_termination(end_time, nstep_end), False
        states = [c.should_terminate(self._fmodel, self) for c in termination_criteria]
        terminate = TrajectoryStateType.TERMINATED in states or self._runtime_termination(end_time, nstep_end)
        interrupt = TrajectoryStateType.INTERRUPTED in states
        return terminate, interrupt

    def _runtime_termination(self, end_time: float, nstep_end: int) -> bool:
        """Returns True if the trajectory should terminate from the runtime arguments."""
        step_termination = self._step >= nstep_end if nstep_end > 0 else False
        time_termination = (
            (isclose(self._t_cur, end_time, abs_tol=1e-9) or self._t_cur >= end_time) if end_time > 0.0 else False
        )
        return step_termination or time_termination

    def _calculate_end_time(self, t_end: float) -> float:
        """Returns the earliest positive end time, or -1.0 if none exist.

        Args:
            t_end: the end time of the advance

        Returns:
            the earliest positive end time
        """
        valid_times = [t for t in (self._t_end, t_end) if t > 0.0]
        return min(valid_times) if valid_times else -1.0

    def _update_diagnostics(self) -> None:
        """Handle diagnostic initialization and plugin updates."""
        if not self._has_diagnostics:
            return

        # Initialize diagnostic now, access to diagdb
        # no longer needs to be pickled at this point
        if not self._initialized_diags:
            self._setup_diagnostics()

        for plugin in self._diagplugins:
            plugin.update(self._snaps[-2], self._snaps[-1])

    def _one_step(self) -> float:
        """Perform a single step of the forward model.

        Perform a single time step of the forward model. This
        function will also set the noise to use for the next step
        in the forward model, replaying the tail if any.

        Args:
            nstep_end: the maximum number of steps to advance
            t_end: the end time of the advance
        """
        if not self._fmodel:
            err_msg = f"Trajectory {self.idstr()} is frozen, without forward model. Advance() deactivated."
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Add the initial snapshot to the list
        if len(self._snaps) == 0:
            self.setup_noise()
            self._append_snapshot()

        # Trigger storing the end state of the current time step
        # if the next trajectory snapshot needs it
        need_end_state = (self._traj_cfg.sparse_start + self._step + 1) % self._traj_cfg.sparse_freq == 0

        try:
            dt = self._fmodel.advance(self._dt, need_end_state)
        except Exception:
            err_msg = f"ForwardModel advance error at step {self._step:08}"
            _logger.exception(err_msg)
            raise

        self._step += 1
        self._t_cur = self._t_cur + dt
        score = self._fmodel.score()

        # Prepare the noise for the next step
        self.setup_noise(score)

        # Append a snapshot at the beginning of the time step
        self._append_snapshot(score)

        if self._traj_cfg.chkfile_dump_all:
            self.store()

        self._score_max = max(self._score_max, score)

        # Increment the computed step counter
        self._computed_steps += 1

        return score

    def _setup_diagnostics(self) -> None:
        """Setup the diagnostic."""
        if self._fmodel is not None:
            if self._diag_configs is None:
                self._initialized_diags = True
                return

            if self._workdir == Path.cwd():
                self._ddb = DiagDB("./diagDB.db")
            else:
                db_path = self._workdir.parents[1] / "./diagDB.db"
                self._ddb = DiagDB(db_path.absolute().as_posix())
            self._diagplugins = diagnosticfactory(
                self._diag_configs,
                self.unique_id(),
                self._weight,
                self._workdir,
                self._fmodel.diagnostic_hook,
                self._ddb,
            )
            self._initialized_diags = True

    def _branch_diagnostics(
        self, ancestor_id: int, discarded_id: int, child_id: int, child_weight: float, branching_time: float
    ) -> None:
        """Duplicate diagnostics entry while branching.

        Args:
            ancestor_id: the ID of the ancestor traj to duplicate
            discarded_id: the ID of the discarded traj
            child_id: the ID of the child traj
            child_weight: the weight of the child traj
            branching_time: the time up to which copy must be performed
        """
        if self._workdir == Path.cwd():
            ddb = DiagDB("./diagDB.db")
        else:
            db_path = self._workdir.parents[1] / "./diagDB.db"
            ddb = DiagDB(db_path.absolute().as_posix())
        ddb.duplicate_diagnostic_history_from_time(ancestor_id, discarded_id, child_id, child_weight, branching_time)
        ddb.close()

    def setup_noise(self, score: float | None = None) -> None:
        """Prepare the noise for the next step.

        When replaying the tail of the trajectory, the noise is the one of the
        replayed snapshot, otherwise a new noise increment is made.

        Args:
            score: an optional score of the new snapshot, checked against the replayed one
        """
        if self._fmodel:
            if self._tail:
                replayed = self._tail.pop(0)
                if score is not None and not isclose(score, replayed.score, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                    warn_msg = (
                        f"{self.idstr()} replay mismatch at time {self._t_cur}: score {score} instead of"
                        f" {replayed.score}. Is the model deterministic for a given noise ?"
                    )
                    _logger.warning(warn_msg)
                self._fmodel.noise = replayed.noise
            else:
                self._fmodel.noise = self._fmodel.make_noise()

    def _append_snapshot(self, score: float | None = None) -> None:
        """Append the current snapshot to the trajectory list."""
        # Append the current snapshot to the trajectory list
        if self._fmodel:
            need_state = (self._traj_cfg.sparse_start + self._step) % self._traj_cfg.sparse_freq == 0 or self._step == 0
            self._snaps.append(
                Snapshot[T_Noise, T_State](
                    time=self._t_cur,
                    # Cast to builtin float: numpy scalars are slow to pickle
                    score=float(score if score is not None else self._fmodel.score()),
                    noise=self._fmodel.noise,
                    state=self._fmodel.get_current_state() if need_state else None,
                ),
            )

    @classmethod
    def init_from_metadata(
        cls,
        metadata: dict[str, Any],
        fmodel_t: type[ForwardModelBaseClass[T_Noise, T_State]],
        traj_cfg: TrajectoryConfig,
        diag_configs: dict[str, Config] | None = None,
        model_params: dict[str, Any] | None = None,
        workdir: Path | None = None,
        frozen: bool = False,
    ) -> Trajectory[T_Noise, T_State]:
        """Initialize a trajectory from serialized metadata."""
        traj: Trajectory[T_Noise, T_State] = Trajectory(
            traj_id=metadata["id"],
            weight=metadata["weight"],
            fmodel_t=fmodel_t,
            traj_cfg=traj_cfg,
            diag_configs=diag_configs,
            model_params=model_params,
            workdir=workdir,
            deterministic=metadata["deterministic"],
            frozen=frozen,
        )

        traj._t_end = metadata["t_end"]
        traj._t_cur = metadata["t_cur"]
        traj._dt = metadata["dt"]
        traj._score_max = metadata["score_max"]
        traj._has_terminated = metadata["terminated"]
        traj._has_converged = metadata["converged"]
        traj._branching_history = metadata["branching_history"]
        traj._computed_steps = metadata["nstep_compute"]

        return traj

    @classmethod
    def restore_from_checkfile(
        cls,
        checkfile: Path,
        metadata: dict[str, Any],
        fmodel_t: type[ForwardModelBaseClass[T_Noise, T_State]],
        traj_cfg: TrajectoryConfig,
        diag_configs: dict[str, Config] | None = None,
        model_params: dict[str, Any] | None = None,
        workdir: Path | None = None,
        frozen: bool = False,
    ) -> Trajectory[T_Noise, T_State]:
        """Return a trajectory restored from an XML chkfile."""
        if not checkfile.exists():
            err_msg = f"Trajectory {checkfile} does not exist."
            _logger.error(err_msg)
            raise FileNotFoundError

        rest_traj: Trajectory[T_Noise, T_State] = Trajectory.init_from_metadata(
            metadata, fmodel_t, traj_cfg, diag_configs, model_params, workdir, frozen
        )

        # Read in trajectory data
        tree = ET.parse(checkfile.absolute())
        root = tree.getroot()
        snapshots = root.find("snapshots")
        if snapshots is not None:
            for snap in snapshots:
                time, score, noise, state = read_xml_snapshot(snap)
                rest_traj._snaps.append(Snapshot[T_Noise, T_State](time=time, score=score, noise=noise, state=state))

        # If the trajectory is frozen, that is all we need. Otherwise
        # handle sparse state, tail and necessary fmodel initialization
        if rest_traj._fmodel:
            # The snapshots after the last available state form the tail,
            # replayed upon the next call to advance
            state_idx = rest_traj.get_last_state_id() or 0
            rest_traj._tail = rest_traj._snaps[state_idx + 1 :]
            rest_traj._snaps = rest_traj._snaps[: state_idx + 1]

            # Current step with python indexing, so remove 1
            rest_traj.set_current_time_and_step(rest_traj._snaps[-1].time, len(rest_traj._snaps) - 1)

            # Ensure everything is set to start the time stepping loop:
            # the noise in the snapshot is the noise used to reach the next state
            rest_traj._fmodel.noise = rest_traj._snaps[-1].noise
            # mypy is not that bright, need explicit check here
            if rest_traj._snaps[-1].state is not None:
                rest_traj._fmodel.set_current_state(rest_traj._snaps[-1].state)

            # Enable the model to perform tweaks
            # after a trajectory restore
            rest_traj._fmodel.post_trajectory_restore_hook(len(rest_traj._snaps) - 1, rest_traj.current_time())

        return rest_traj

    @classmethod
    def branch_from_trajectory(
        cls,
        from_traj: Trajectory[T_Noise, T_State],
        rst_traj: Trajectory[T_Noise, T_State],
        score: float,
        new_weight: float,
    ) -> Trajectory:
        """Create a new trajectory.

        Loading the beginning of a provided trajectory
        for all entries with score below a given score.
        This effectively branches the trajectory.

        Although the rst_traj is provided as an argument, it is
        only used to set metadata of the branched trajectory.

        Args:
            from_traj: an already existing trajectory to restart from
            rst_traj: the trajectory being restarted
            score: a threshold score
            new_weight: the weight of the child trajectory
        """
        # Initialize the new trajectory object (handles path and ID logic)
        rest_traj = cls._init_branched_trajectory(from_traj, rst_traj, new_weight)

        # Return empty the empty trajectory immediately
        if not from_traj._snaps:
            return rest_traj

        # Locate the branching points in the known history of the ancestor,
        # including its tail if not replayed yet
        known_snaps = from_traj._snaps + from_traj._tail
        high_score_idx, last_snap_with_state = cls._find_branch_indices(known_snaps, score)

        # Transfer the data from the ancestor to the child
        cls._transfer_data(known_snaps, rest_traj, high_score_idx, last_snap_with_state)

        # The child diverges from the ancestor after the first snapshot above the threshold
        cls._renew_last_noise(rest_traj)

        # Finalize branching
        ancestor_workdir = from_traj.get_workdir()
        cls._finalize_branch(from_traj.unique_id(), rst_traj.unique_id(), rest_traj, new_weight, ancestor_workdir)

        return rest_traj

    @classmethod
    def clone_from_trajectory(
        cls,
        from_traj: Trajectory[T_Noise, T_State],
        rst_traj: Trajectory[T_Noise, T_State],
        new_weight: float,
    ) -> Trajectory:
        """Create a new trajectory as a clone of an ancestor at its current time.

        Unlike :meth:`branch_from_trajectory`, the entire history of the ancestor
        is copied. If the last snapshot of the ancestor does not hold a model
        state (sparse state), the snapshots after the last available state are
        put in the tail and replayed upon the next call to advance.

        The noise the ancestor has already drawn for its upcoming step is *not*
        transferred: the clone draws fresh noise beyond the ancestor current time,
        such that clones diverge from their ancestor.

        Although the rst_traj is provided as an argument, it is
        only used to set metadata of the cloned trajectory.

        Args:
            from_traj: an already existing trajectory to clone
            rst_traj: the trajectory being replaced
            new_weight: the weight of the clone trajectory
        """
        # Initialize the new trajectory object (handles path and ID logic)
        rest_traj = cls._init_branched_trajectory(from_traj, rst_traj, new_weight)

        # Return empty the empty trajectory immediately
        if not from_traj._snaps:
            return rest_traj

        state_idx = from_traj.get_last_state_id()
        if state_idx is None:
            err_msg = f"Unable to clone {from_traj.idstr()}: no state available in the trajectory"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Everything up to the last state is a full snapshot, the rest
        # of the known history of the ancestor forms the tail
        known_snaps = from_traj._snaps + from_traj._tail
        rest_traj._snaps = known_snaps[: state_idx + 1]
        rest_traj._tail = known_snaps[state_idx + 1 :]

        # The clone diverges from the ancestor after its last known snapshot
        cls._renew_last_noise(rest_traj)

        # Finalize branching
        ancestor_workdir = from_traj.get_workdir()
        cls._finalize_branch(from_traj.unique_id(), rst_traj.unique_id(), rest_traj, new_weight, ancestor_workdir)

        return rest_traj

    @classmethod
    def _init_branched_trajectory(
        cls, from_traj: Trajectory[T_Noise, T_State], rst_traj: Trajectory[T_Noise, T_State], weight: float
    ) -> Trajectory[T_Noise, T_State]:
        """Initialize a new trajectory for branching.

        Args:
            from_traj: an already existing trajectory to restart from
            rst_traj: the trajectory being restarted
            weight: the weight of the child trajectory
        """
        tid, nb = get_index_from_id(rst_traj.idstr())
        new_name = form_trajectory_id(tid, nb + 1)
        new_workdir = Path(rst_traj.get_workdir().parents[0] / new_name)
        fmodel_t = type(from_traj._fmodel) if from_traj._fmodel else None

        new_traj: Trajectory[T_Noise, T_State] = Trajectory(
            traj_id=rst_traj.id(),
            weight=weight,
            fmodel_t=fmodel_t,
            traj_cfg=from_traj._traj_cfg,
            diag_configs=from_traj._diag_configs,
            model_params=from_traj._model_params,
            deterministic=from_traj._deterministic,
            workdir=new_workdir,
        )

        # Copy history if not empty
        if from_traj._snaps:
            new_traj._branching_history = copy.deepcopy(rst_traj._branching_history)
            new_traj._branching_history.append(from_traj.id())

        # Update the checkfile
        # It needs to be done after copying the history
        # as the unique traj ID uses the length of the history
        xml_name = f"{new_traj.idstr()}.xml"
        new_traj.set_checkfile(Path(rst_traj.get_checkfile().parents[0] / xml_name))

        return new_traj

    @staticmethod
    def _find_branch_indices(snaps: list[Snapshot[T_Noise, T_State]], score_threshold: float) -> tuple[int, int]:
        """Finds the split index and the last index containing a state.

        To ensure that sampling converges, branching occurs on
        the first snapshot with a score *strictly* above the target.

        Args:
            snaps: the known snapshots of the ancestor trajectory
            score_threshold: the score threshold up to which we must copy

        Returns:
            a tuple with the index of the first snapshot above threshold and
            the index of the last snapshot with a state (equal or below high_score_idx).
        """
        high_score_idx = 0
        last_state_idx = 0
        for i, snap in enumerate(snaps):
            if snap.has_state:
                last_state_idx = i
            if snap.score > score_threshold:
                high_score_idx = i
                break
        return high_score_idx, last_state_idx

    @staticmethod
    def _transfer_data(
        snaps: list[Snapshot[T_Noise, T_State]], rest_traj: Trajectory[T_Noise, T_State], high_idx: int, state_idx: int
    ) -> None:
        """Transfer data from the ancestor to the child.

        The child gets the ancestor snapshots up to the last state, and replays
        the following ones up to the snapshot high_idx.

        Args:
            snaps: the known snapshots of the ancestor trajectory
            rest_traj: the child trajectory
            high_idx: the index of the first snapshot above the threshold
            state_idx: the index of the last snapshot with a state (equal or below high_idx)
        """
        # Everything up to the last state is a full snapshot
        rest_traj._snaps = snaps[: state_idx + 1]

        # The snapshots from the last state up to high_idx form the tail
        rest_traj._tail = snaps[state_idx + 1 : high_idx + 1]

    @staticmethod
    def _renew_last_noise(traj: Trajectory[T_Noise, T_State]) -> None:
        """Make a new noise increment for the step following the last known snapshot.

        The snapshot noise is the noise of the next step: renewing it makes a
        branched or cloned trajectory diverge from its ancestor after its last
        known snapshot, and keeps the snapshots consistent with the noise actually
        used, such that restoring from a checkfile replays the trajectory history.

        Args:
            traj: the branched or cloned trajectory
        """
        if traj._fmodel:
            noise = traj._fmodel.make_noise()
            if traj._tail:
                traj._tail[-1] = replace(traj._tail[-1], noise=noise)
            else:
                traj._snaps[-1] = replace(traj._snaps[-1], noise=noise)

    @staticmethod
    def _finalize_branch(
        ancestor_id: int,
        discarded_id: int,
        rest_traj: Trajectory[T_Noise, T_State],
        weight: float,
        ancestor_workdir: Path,
    ) -> None:
        """Finalizes metadata, model state, and diagnostics.

        Args:
            ancestor_id: the unique ID of the ancestor traj
            discarded_id: the unique ID of the discarded traj
            rest_traj: the child trajectory
            weight: the weight of the new child trajectory
            ancestor_workdir: the workdir of the ancestor trajectory
        """
        rest_traj._t_cur = rest_traj._snaps[-1].time
        rest_traj._step = len(rest_traj._snaps) - 1

        if rest_traj._fmodel:
            # The noise in the snapshot is the noise used to reach the next state
            last_snap = rest_traj._snaps[-1]
            rest_traj._fmodel.noise = last_snap.noise
            if last_snap.state is not None:
                rest_traj._fmodel.set_current_state(last_snap.state)

            rest_traj.update_metadata()
            rest_traj._fmodel.post_trajectory_branching_hook(
                rest_traj._step, rest_traj._t_cur, ancestor_workdir.absolute().as_posix()
            )

        if rest_traj._has_diagnostics:
            rest_traj._branch_diagnostics(ancestor_id, discarded_id, rest_traj.unique_id(), weight, rest_traj._t_cur)

    def store(self, traj_file: Path | None = None, write_metadata_json: bool = False) -> None:
        """Store the trajectory data to an XML chkfile.

        The default behavior is to store the trajectory into the
        file specified by the attribute self._checkFile unless a
        different path is provided.
        The metadata are not writen to file by default, as the pyREVS database
        handle metadata in an SQL file. It can be triggered when using trajectories
        in stand-alone.

        Args:
            traj_file: an optional path file to store the trajectory to
            write_metadata_json: an optional boolean to also write the metadata json dict
        """
        root = ET.Element(self.idstr())
        root.append(dict_to_xml("params", asdict(self._traj_cfg)))
        snaps_xml = ET.SubElement(root, "snapshots")
        # The tail is part of the known trajectory history
        for k, snap in enumerate(self._snaps + self._tail):
            snaps_xml.append(make_xml_snapshot(k, snap.time, snap.score, snap.noise, snap.state))
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)

        if traj_file is not None:
            tree.write(traj_file.as_posix())
        else:
            tree.write(self._checkFile.as_posix())

        # Separately dumps the metadata if requested
        if write_metadata_json:
            json_path = (
                traj_file.parent / Path(traj_file.stem + ".json")
                if traj_file is not None
                else self._checkFile.parent / Path(self._checkFile.stem + ".json")
            )
            with json_path.open("w") as f:
                f.write(self.serialize_metadata_json())

    def set_weight(self, weight: float) -> None:
        """Set the trajectory weight.

        Args:
            weight: the (new) trajectory weight in the ensemble
        """
        self._weight = weight

    def weight(self) -> float:
        """Return the trajectory weight in the ensemble."""
        return self._weight

    def update_metadata(self) -> None:
        """Update trajectory score/ending metadata.

        Update the maximum of the score function over the trajectory
        as well as the bool values for has_converged and has_terminated
        """
        # Scores can be negative: same default as upon initialization
        self._score_max = max((snap.score for snap in self._snaps + self._tail), default=-1.0e12)
        if self._fmodel:
            self._has_converged = self._fmodel.check_convergence(
                self._step, self._t_cur, self._score_max, self._traj_cfg.targetscore
            )
            if self._has_converged:
                self._has_terminated = True
            else:
                self._has_terminated = self._fmodel.check_termination(
                    self._step,
                    self._t_cur,
                )
        else:
            self._has_converged = False
            self._has_terminated = False

    def set_current_time_and_step(self, time: float, step: int) -> None:
        """Set the current time and step."""
        self._t_cur = time
        self._step = step

    def current_time(self) -> float:
        """Return the current trajectory time."""
        return self._t_cur

    def current_step(self) -> int:
        """Return the current trajectory step."""
        return self._step

    def step_size(self) -> float:
        """Return the time step size."""
        return self._dt

    def score_max(self) -> float:
        """Return the maximum of the score function."""
        return self._score_max

    def is_converged(self) -> bool:
        """Return True for converged trajectory."""
        return self._has_converged

    def is_terminated(self) -> bool:
        """Return True for terminated trajectory."""
        return self._has_terminated

    def has_started(self) -> bool:
        """Return True if computation has started."""
        return self._t_cur > 0.0

    def get_checkfile(self) -> Path:
        """Return the trajectory check file name."""
        return self._checkFile

    def get_time_array(self) -> npt.NDArray[np.number]:
        """Return the trajectory time instants, including the tail."""
        return np.array([snap.time for snap in self._snaps + self._tail], dtype=np.float64)

    def get_score_array(self) -> npt.NDArray[np.number]:
        """Return the trajectory scores, including the tail."""
        return np.array([snap.score for snap in self._snaps + self._tail], dtype=np.float64)

    def get_noise_array(self) -> npt.NDArray[Any]:
        """Return the trajectory noises, including the tail."""
        if not self._snaps:
            return np.array([])

        return np.array([snap.noise for snap in self._snaps + self._tail])

    def get_state_list(self) -> list[tuple[int, T_State | None]]:
        """Return a list of states and associated indices.

        Returns:
            A list of tuples with index and states
        """
        return [(k, self._snaps[k].state) for k in range(len(self._snaps)) if self._snaps[k].has_state]

    def get_length(self) -> int:
        """Return the trajectory length, including the tail."""
        return len(self._snaps) + len(self._tail)

    def get_nbranching(self) -> int:
        """Return the number of branching events."""
        if len(self._branching_history) > self.max_nbranch:
            err_msg = f"Branching history size exceeds maximum of {self.max_nbranch}!"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return len(self._branching_history)

    def get_computed_steps_count(self) -> int:
        """Return the number of compute steps taken."""
        return self._computed_steps

    def get_last_state(self) -> T_State | None:
        """Return the last state in the trajectory."""
        for snap in reversed(self._snaps):
            if snap.has_state:
                return snap.state

        return None

    def get_last_state_id(self) -> int | None:
        """Return the id of the last state in the trajectory."""
        for idx, snap in reversed(list(enumerate(self._snaps))):
            if snap.has_state:
                return idx

        return None

    def get_metadata(self) -> dict:
        """Return a dict with traj metadata.

        Returns:
            The trajectory metadata in a dict.
        """
        return {
            "id": self.id(),
            "weight": float(self._weight),
            "t_end": self._t_end,
            "t_cur": self._t_cur,
            "dt": self._dt,
            "terminated": bool(self._has_terminated),
            "converged": bool(self._has_converged),
            "score_max": float(self._score_max),
            "length": self.get_length(),
            "nstep_compute": self._computed_steps,
            "deterministic": self._deterministic,
            "branching_history": self._branching_history,
        }

    def serialize_metadata_json(self) -> str:
        """Return a json string with metadata.

        Returns:
            A json string with the trajectory metadata
        """
        return json.dumps(
            self.get_metadata(),
            default=str,
        )

    @classmethod
    def deserialize_metadata(cls, json_str: str) -> dict[str, Any]:
        """Load a json string into a properly typed metadata dict.

        Returns:
            A dictionary with the metadata
        """
        mstr = json.loads(json_str)
        return {
            "id": int(mstr["id"]),
            "weight": float(mstr["weight"]),
            "t_end": float(mstr["t_end"]),
            "t_cur": float(mstr["t_cur"]),
            "dt": float(mstr["dt"]),
            "terminated": mstr["terminated"],
            "converged": mstr["converged"],
            "score_max": float(mstr["score_max"]),
            "length": int(mstr["length"]),
            "nstep_compute": int(mstr["nstep_compute"]),
            "deterministic": mstr["deterministic"],
            "branching_history": cast("list[int]", mstr["branching_history"]),
        }

    def delete(self) -> None:
        """Clear the trajectory on-disk data."""
        self._checkFile.unlink()
        if self._workdir.exists():
            shutil.rmtree(self._workdir)
