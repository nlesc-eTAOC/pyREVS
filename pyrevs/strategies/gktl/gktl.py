"""The GiardinaKurchanLecomteTailleur sampling strategy."""

import logging
from math import ceil
from math import isclose
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from pyrevs.core import Config
from pyrevs.core import RuntimeConfig
from pyrevs.database import Database
from pyrevs.database import DatabaseCoreSpec
from pyrevs.runner import BaseRunner
from pyrevs.runner import RunnerConfig
from pyrevs.runner import make_runner
from pyrevs.runner import pool_worker
from pyrevs.runner.worker import update_trajectory_in_sql
from pyrevs.strategies.base import BaseSamplingStrategy
from pyrevs.strategies.base import ModelTerminationCriterion
from pyrevs.strategies.base import TerminationCriterion
from pyrevs.strategies.base import TimeInterruptionCriterion
from pyrevs.strategies.base import TimeTerminationCriterion
from pyrevs.trajectory import Trajectory
from .config import GKTLConfig
from .extension import GKTLDatabaseExtension

_logger = logging.getLogger(__name__)

# Tolerance used when comparing trajectory times
TIME_TOL = 1e-9


def select_clones(log_w: npt.NDArray[np.number], rng: np.random.Generator) -> list[int]:
    """Select the clones of the next generation.

    Each clone i gets floor(W_i / R + U_i) copies, with W_i = exp(log_w_i),
    R = mean(W) and U_i uniform in [0, 1). Copies are then randomly removed
    or added to maintain the ensemble size. Surviving clones keep their own
    slot, while the extra copies take the slots of the killed clones.

    Args:
        log_w: the log-weight of each clone
        rng: a random number generator

    Returns:
        the parent index of each slot of the new generation
    """
    ntraj = len(log_w)

    # Create the pool of trajectories to pick from
    # Each trajectory of the initial pool gets ncopies in the
    # new pool
    w = np.exp(log_w - np.max(log_w))
    ncopies = np.floor(w / np.mean(w) + rng.uniform(size=ntraj)).astype(int)
    pool = np.repeat(np.arange(ntraj), ncopies)

    # All trajectories were killed
    if len(pool) == 0:
        pool = rng.choice(ntraj, size=ntraj, p=w / np.sum(w))

    # Choose nTraj at random (no repetition)
    elif len(pool) > ntraj:
        pool = rng.choice(pool, size=ntraj, replace=False)

    # Choose nTraj at random (repetition allowed)
    elif len(pool) < ntraj:
        pool = np.concatenate([pool, rng.choice(pool, size=ntraj - len(pool), replace=True)])

    # Assemble a list of parents
    counts = np.bincount(pool, minlength=ntraj)
    extras = [i for i in range(ntraj) for _ in range(max(counts[i] - 1, 0))]
    parents = []
    for i in range(ntraj):
        if counts[i] > 0:
            parents.append(i)
        else:
            parents.append(extras.pop())
    return parents


def observable_increment(traj: Trajectory, t_start: float, observable: str) -> float:
    """Compute the observable increment of a trajectory since a given time.

    Args:
        traj: the trajectory
        t_start: the start time of the interval
        observable: either 'integral' (time integral of the score) or 'increment'

    Returns:
        the observable increment over [t_start, current time]
    """
    times = traj.get_time_array()
    scores = traj.get_score_array()
    mask = times >= t_start - TIME_TOL
    t, s = times[mask], scores[mask]
    if len(t) < 2:  # noqa: PLR2004
        return 0.0
    if observable == "integral":
        return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(t)))
    return float(s[-1] - s[0])


@BaseSamplingStrategy.register("gktl")
class GKTL(BaseSamplingStrategy):
    """A strategy class implementing GKTL.

    The Giardina-Kurchan-Lecomte-Tailleur algorithm evolves an ensemble of
    trajectories and, at fixed resampling intervals, kills or clones each of them
    according to a weight exp(k * dA), where dA is an observable increment
    over the interval. This biases the ensemble toward large values of the
    observable. The bias is accounted for when computing estimates (see
    :class:`GKTLDatabaseExtension`).

    Trajectories that converge (reach the target score) before the end time
    are frozen: their observable increment is zero over later intervals.

    Notes:
        This strategy relies on time management provided by
        BaseSamplingStrategy (e.g. ``self._end_date``, ``elapsed_time()``).
    """

    def __init__(
        self,
        fmodel_t: Any,
        runtime_cfg: RuntimeConfig,
        runner_cfg: RunnerConfig,
        strategy_cfg: GKTLConfig,
        deterministic: bool,
    ) -> None:
        """Initialize a GKTL object.

        Args:
            fmodel_t: the forward model type
            runtime_cfg: the runtime config
            runner_cfg: the runner config
            strategy_cfg: the gktl config
            deterministic: the deterministic flag

        Raises:
            ValueError: if necessary config parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._gktl_cfg = strategy_cfg
        self._gktl_cfg.validate()
        self._runner_cfg = runner_cfg
        self._loglevel = runtime_cfg.loglevel
        self._logfile = runtime_cfg.logfile
        self._deterministic = deterministic
        self._db_ext: GKTLDatabaseExtension | None = None

        # Number of resampling intervals
        self._n_intervals = ceil(self._gktl_cfg.end_time / self._gktl_cfg.resampling_interval - TIME_TOL)

        # Termination criteria common to all intervals. The interval
        # interruption criterion is appended for each interval.
        self._term_crit: list[TerminationCriterion] = [TimeTerminationCriterion(self._gktl_cfg.end_time)]
        if strategy_cfg.use_custom_termination:
            self._term_crit.append(ModelTerminationCriterion())

    def _req_db_ext(self) -> GKTLDatabaseExtension:
        if self._db_ext is None:
            err_msg = "Database extension not initialized"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)
        return self._db_ext

    def _interval_end_time(self, n: int) -> float:
        """Return the end time of the n-th (0-based) resampling interval."""
        return min((n + 1) * self._gktl_cfg.resampling_interval, self._gktl_cfg.end_time)

    def run_gktl(self, database: Database, plot_diags: bool) -> bool:
        """Run the GKTL resampling loop.

        Advance the ensemble of trajectories interval by interval, resampling
        the ensemble at the end of each interval (except the last one).
        The loop resumes from the last completed resampling step when
        restarting from an existing database.

        Args:
            database: the sampling database
            plot_diags: whether to plot diagnostics

        Returns:
            True if the end time was reached (or all trajectories terminated)

        Raises:
            RuntimeError if the runner fails
        """
        ext = self._req_db_ext()
        n = ext.iteration_count()

        inf_msg = (
            f"Running GKTL with {database.n_traj()} clones over {self._n_intervals} intervals (k={self._gktl_cfg.k})"
        )
        _logger.info(inf_msg)

        if n == 0:
            self._update_weights(database)

        with make_runner(
            self._runner_cfg,
            pool_worker,
            loglevel=self._loglevel,
            logfile=self._logfile,
            max_workers=self._gktl_cfg.ntrajectories,
        ) as runner:
            while True:
                t_next = self._interval_end_time(n)
                inf_msg = f"### [{n:5}] GKTL interval - from time {ext.current_time()} to {t_next}"
                _logger.info(inf_msg)

                if plot_diags:
                    self._plot_scores(database, n)

                self._advance_ensemble(database, runner, t_next)

                if not self._ensemble_synchronized(database, t_next):
                    self._log_interruption(n)
                    return False

                if n + 1 >= self._n_intervals or all(t.is_terminated() for t in database.traj_list()):
                    break

                self._resample(database, t_next)
                n += 1

                if self.out_of_time():
                    warn_msg = f"Ran out of time after {n} GKTL resampling steps"
                    _logger.warning(warn_msg)
                    return False

        if plot_diags:
            self._plot_scores(database, n + 1)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)
        return True

    def _advance_ensemble(self, database: Database, runner: BaseRunner, t_next: float) -> None:
        """Advance all the trajectories up to the end of the current interval.

        Args:
            database: the trajectory database
            runner: the runner
            t_next: the end time of the interval
        """
        term_crit = [*self._term_crit, TimeInterruptionCriterion(t_next)]
        for t in database.traj_list():
            task = [t, term_crit, self._end_date, database.pool_file(), database.path()]
            runner.make_promise(task)

        try:
            t_list = runner.execute_promises()
        except Exception as exc:
            err_msg = f"Failed to advance the ensemble of {database.n_traj()} trajectories"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg) from exc

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        database.update_traj_list(t_list)

    @staticmethod
    def _ensemble_synchronized(database: Database, t_next: float) -> bool:
        """Check that all clones are terminated or reached the end of the interval."""
        return all(
            t.is_terminated() or isclose(t.current_time(), t_next, abs_tol=TIME_TOL) or t.current_time() >= t_next
            for t in database.traj_list()
        )

    def _log_interruption(self, n: int) -> None:
        """Log the reason the interval was not completed."""
        if Path("./pyrevs_stop").exists():
            warn_msg = f"User-interrupt triggered during GKTL interval {n}"
            _logger.warning(warn_msg)
            Path("./pyrevs_stop").unlink(missing_ok=True)
        else:
            warn_msg = f"Ran out of time during GKTL interval {n}"
            _logger.warning(warn_msg)

    def _resample(self, database: Database, time: float) -> None:
        """Perform a GKTL resampling step.

        Compute the weight of each clone over the last interval, select the next
        generation and replace the killed clones by copies of the survivors.

        Args:
            database: the trajectory database
            time: the time of the resampling step
        """
        ext = self._req_db_ext()
        window_start = ext.window_start()
        delta_a = np.array(
            [
                observable_increment(database.get_traj(i), window_start[i], self._gktl_cfg.bias_observable)
                for i in range(database.traj_list_len())
            ]
        )

        # Enable deterministic runs by setting a (different) seed
        # for each resampling step
        rng = (
            np.random.default_rng(seed=42 * (ext.iteration_count() + 1))
            if self._deterministic
            else np.random.default_rng()
        )
        parents = select_clones(self._gktl_cfg.k * delta_a, rng)

        n_killed = sum(1 for j, p in enumerate(parents) if p != j)
        inf_msg = f"    Resampling at time {time}: {n_killed} clone(s) killed and replaced"
        _logger.info(inf_msg)

        # Killed slots are only filled with copies of surviving clones,
        # which keep their own slot: no ancestor is overwritten in the loop.
        pool_db = database.get_pool_db()
        for j, p in enumerate(parents):
            if p == j:
                continue
            killed = database.get_traj(j)
            database.archive_trajectory(killed)
            clone = Trajectory.clone_from_trajectory(database.get_traj(p), killed, database.get_traj(p).weight())
            update_trajectory_in_sql(clone, pool_db, database.path())
            if database.to_disk():
                clone.store()
            database.overwrite_traj(j, clone)

        new_window_start = [database.get_traj(i).current_time() for i in range(database.traj_list_len())]
        ext.record_resampling(delta_a, parents, new_window_start, time)
        self._update_weights(database)

    def _update_weights(self, database: Database) -> None:
        """Set the estimator weight of each active trajectory in the ensemble."""
        weights = self._req_db_ext().clone_weights()
        pool_db = database.get_pool_db()
        for i, t in enumerate(database.traj_list()):
            t.set_weight(float(weights[i]))
            if pool_db:
                pool_db.update_trajectory_weight(t.id(), float(weights[i]))

    @staticmethod
    def _plot_scores(database: Database, n: int) -> None:
        """Plot the score functions of the ensemble."""
        pltfile = f"Score_GKTL_{n:06}.png"
        if Path(pltfile).exists():
            wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
            _logger.warning(wrn_msg)
        database.plot_score_functions(pltfile)

    def compute_probability(self, database: Database, plot_diags: bool) -> float:
        """Compute the rare-event probability using GKTL.

        Args:
            database: the trajectory database
            plot_diags: whether to plot diagnostics

        Returns:
            the rare-event probability, or -1.0 if the run did not complete
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using GKTL"
        _logger.info(inf_msg)

        if not self.run_gktl(database, plot_diags):
            return -1.0

        # Note: the killed clones are archived from the main process,
        # no need to reload the archived trajectories from disk.
        database.info()

        return database.get_event_probability()

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        # Initialize the active ensemble
        database.init_active_ensemble()

        # Check that the database extension is initialized
        if self._db_ext is None:
            err_msg = "Database extension is not initialized ! Call initialize_database_schema() first"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        proba = self.compute_probability(database, plot_diags)
        inf_msg = f"Event probability: {proba}"
        _logger.info(inf_msg)

    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
        """Initialize database core state."""
        spec = DatabaseCoreSpec(
            ntraj=self._gktl_cfg.ntrajectories,
            strategy="gktl",
            deterministic=self._deterministic,
            diag_configs=diag_configs,
        )
        database.initialize_core_state(spec)

        # Setup GKTL extension
        self._db_ext = GKTLDatabaseExtension()
        self._db_ext.initialize(database, self._gktl_cfg.k)
        database.attach_extension(self._db_ext)
