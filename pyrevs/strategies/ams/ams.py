"""The (T)AMS sampling strategy."""

import logging
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
from pyrevs.runner import ms_worker
from pyrevs.runner import pool_worker
from pyrevs.strategies.base import BaseSamplingStrategy
from pyrevs.strategies.base import LowScoreTerminationCriterion
from pyrevs.strategies.base import ModelTerminationCriterion
from pyrevs.strategies.base import TerminationCriterion
from pyrevs.strategies.base import TimeTerminationCriterion
from pyrevs.utils.utils import get_min_scored
from .config import AMSConfig
from .extension import AMSDatabaseExtension

_logger = logging.getLogger(__name__)

STALL_TOL = 1e-10


@BaseSamplingStrategy.register("ams")
class AMS(BaseSamplingStrategy):
    """A strategy class implementing (T)AMS.

    The interface to (T)AMS, implementing the main steps of
    the algorithm.

    Initialization of the (T)AMS class requires a forward model
    type which encapsulate all the model-specific code, and
    an optional list of options.

    The algorithm is roughly divided in two steps:
    1. Initialization of the trajectory ensemble
    2. Splitting iterations

    Separate control of the parallelism is provided for
    both steps.

    All the algorithm data are contained in the (T)AMS database.
    For control purposes, a walltime limit is also provided. It is
    passed to working and lead to the termination of the algorithm
    in a state that can be saved to disk and restarted at a later stage.

    Attributes:
        _fmodel_t: the forward model type
        _config: the config object
        _wallTime: the walltime limit
        _init_ensemble_only: whether or not to stop after initializing the trajectory ensemble
    """

    def __init__(
        self,
        fmodel_t: Any,
        runtime_cfg: RuntimeConfig,
        runner_cfg: RunnerConfig,
        strategy_cfg: AMSConfig,
        deterministic: bool,
    ) -> None:
        """Initialize a AMS strategy.

        Args:
            fmodel_t: the forward model type
            runtime_cfg: the runtime config dataclass
            runner_cfg: the runner config dataclass
            strategy_cfg: the AMS config dataclass
            deterministic: the deterministic flag trigger deterministic runs

        Raises:
            ValueError: if necessary parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._ams_cfg = strategy_cfg
        self._ams_cfg.validate()
        self._runner_cfg = runner_cfg
        self._loglevel = runtime_cfg.loglevel
        self._logfile = runtime_cfg.logfile
        self._deterministic = deterministic
        self._db_ext: AMSDatabaseExtension | None = None
        self._term_crit: list[TerminationCriterion] = []

        if strategy_cfg.variant == "ams":
            # Check for None. TODO: validate() above does not satisfy mypy
            if self._ams_cfg.min_score is None:
                err_msg = "AMSConfig.min_score must be set for AMS"
                _logger.error(err_msg)
                raise ValueError(err_msg)
            self._term_crit.append(LowScoreTerminationCriterion(self._ams_cfg.min_score))
        elif strategy_cfg.variant == "tams":
            self._term_crit.append(TimeTerminationCriterion(self._ams_cfg.end_time))
        elif strategy_cfg.variant == "hams":
            # Check for None. TODO: validate() above does not satisfy mypy
            if self._ams_cfg.min_score is None:
                err_msg = "AMSConfig.min_score must be set for HAMS"
                _logger.error(err_msg)
                raise ValueError(err_msg)
            self._term_crit.append(TimeTerminationCriterion(self._ams_cfg.end_time))
            self._term_crit.append(LowScoreTerminationCriterion(self._ams_cfg.min_score))
        else:
            err_msg = f"Unknown variant {self._ams_cfg.variant}"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        # If requested, also add the custom model termination check
        if strategy_cfg.use_custom_termination:
            self._term_crit.append(ModelTerminationCriterion())

    def _req_db_ext(self) -> AMSDatabaseExtension:
        if self._db_ext is None:
            err_msg = "Database extension not initialized"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._db_ext

    def generate_trajectory_ensemble(self, tdb: Database) -> None:
        """Schedule the generation of an ensemble of stochastic trajectories.

        Loop over all the trajectories in the database and schedule
        advancing them to either end time or convergence with the
        runner.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Args:
            tdb: the AMS database

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating the initial ensemble of {tdb.n_traj()} trajectories"
        _logger.info(inf_msg)

        with make_runner(
            self._runner_cfg,
            pool_worker,
            loglevel=self._loglevel,
            logfile=self._logfile,
            max_workers=self._ams_cfg.ntrajectories,
        ) as runner:
            for t in tdb.traj_list():
                task = [t, self._term_crit, self._end_date, tdb.pool_file(), tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except Exception as exc:
                err_msg = f"Failed to generate the initial ensemble of {tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise RuntimeError(err_msg) from exc

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        tdb.update_traj_list(t_list)

        if tdb.count_terminated_traj() == tdb.n_traj():
            self._req_db_ext().set_init_ensemble_flag(True)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def check_exit_splitting_loop(self, tdb: Database, k: int, k_it: int) -> tuple[bool, npt.NDArray[np.number]]:
        """Check for exit criterion of the splitting loop.

        Args:
            tdb: the AMS database
            k: counter of discarded trajectories
            k_it: loop counter

        Returns:
            bool to trigger splitting loop break
            array of maximas across all trajectories
        """
        # Gather max score from all trajectories
        # and check for early convergence
        all_converged = True
        maxes = np.zeros(tdb.traj_list_len(), dtype=float)
        for i in range(tdb.traj_list_len()):
            maxes[i] = tdb.get_traj(i).score_max()
            all_converged = all_converged and tdb.get_traj(i).is_converged()

        # Check for walltime
        if self.out_of_time():
            warn_msg = f"Ran out of time after {k} splitting iterations"
            _logger.warning(warn_msg)
            return True, maxes

        # Exit if our work is done
        if all_converged:
            inf_msg = f"All trajectories converged after {k_it} splitting iterations and {k} discarded trajectories"
            _logger.info(inf_msg)
            return True, maxes

        # Exit if splitting is stalled
        if (np.amax(maxes) - np.amin(maxes)) < STALL_TOL:
            err_msg = f"Splitting is stalling with all trajectories stuck at a score_max: {np.amax(maxes)}"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        return False, maxes

    def finish_ongoing_splitting(self, tdb: Database) -> None:
        """Check and finish unfinished splitting iterations.

        If the run was interrupted during a splitting iteration,
        the branched trajectories might not have terminated yet. In that case,
        a list of trajectories to finish is listed in the database.
        """
        # Check the database for unfinished splitting iteration when restarting.
        # At this point, branching has been done, but advancing to final
        # time is still ongoing.
        ongoing_list = self._req_db_ext().get_ongoing()
        if ongoing_list:
            inf_msg = f"Unfinished splitting iteration detected, traj {ongoing_list} need(s) finishing"
            _logger.info(inf_msg)
            with make_runner(
                self._runner_cfg,
                pool_worker,
                loglevel=self._loglevel,
                logfile=self._logfile,
                max_workers=self._ams_cfg.l_j,
            ) as runner:
                for i in ongoing_list:
                    t = tdb.get_traj(i)
                    task = [t, self._term_crit, self._end_date, tdb.pool_file(), tdb.path()]
                    runner.make_promise(task)

                try:
                    finished_traj = runner.execute_promises()
                except Exception as exc:
                    err_msg = f"Failed to finish branching {len(ongoing_list)} trajectories"
                    _logger.exception(err_msg)
                    raise RuntimeError(err_msg) from exc

                _logger.info("Done with unfinished")

                for t in finished_traj:
                    tdb.overwrite_traj(t.id(), t)

                # Wrap up the iteration by updating its status in the
                # database and incrementing the iteration counter
                self._req_db_ext().mark_last_splitting_iteration_as_done()

    def get_restart_at_random(self, tdb: Database, min_idx_list: list[int]) -> list[int]:
        """Get a list of trajectory index to restart from at random.

        Select trajectories to restart from among the ones not
        in min_idx_list.

        Args:
            tdb: the AMS database
            min_idx_list: list of trajectory index to restart from

        Returns:
            list of trajectory index to restart from
        """
        # Enable deterministic runs by setting a (different) seed
        # for each splitting iteration
        rng = (
            np.random.default_rng(seed=42 * self._req_db_ext().k_split())
            if self._deterministic
            else np.random.default_rng()
        )
        rest_idx = [-1] * len(min_idx_list)
        for i in range(len(min_idx_list)):
            rest_idx[i] = min_idx_list[0]
            while rest_idx[i] in min_idx_list:
                rest_idx[i] = rng.integers(low=0, high=tdb.traj_list_len(), dtype=int)
        return rest_idx

    def do_multilevel_splitting(self, tdb: Database, plot_diags: bool) -> None:
        """Schedule splitting of the initial ensemble of stochastic trajectories.

        Perform the multi-level splitting iterations, possibly restarting multiple
        trajectories at each iterations. All the trajectories in an iterations are
        advanced together, such that each iteration takes the maximum duration among
        the branched trajectories.

        If the walltime is exceeded, the splitting loop is stopped and ongoing
        trajectories are flagged in the database in order to finish them upon
        restart.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Args:
            tdb: the AMS database
            plot_diags: whether or not to plot diagnostics

        Raises:
            Error if the runner fails
        """
        inf_msg = "Using multi-level splitting to get the probability"
        _logger.info(inf_msg)

        # Finish any unfinished splitting iteration
        self.finish_ongoing_splitting(tdb)

        # Initialize number of discarded trajectories counter
        k = self._req_db_ext().k_split()

        # Initialize the splitting iteration counter
        k_it = self._req_db_ext().get_iteration_count()

        with make_runner(
            self._runner_cfg,
            ms_worker,
            loglevel=self._loglevel,
            logfile=self._logfile,
            max_workers=self._ams_cfg.l_j,
        ) as runner:
            while k < self._ams_cfg.nsplititer:
                # Perform a single splitting iteration
                should_stop, n_discarded = self._one_splitting_iteration(
                    tdb,
                    runner,
                    k,
                    k_it,
                    plot_diags,
                )

                # Stop the splitting loop for any reason
                if should_stop:
                    break

                # Wrap up the iteration by updating its status in the
                # database and incrementing the counter of discarded trajectories
                self._req_db_ext().mark_last_splitting_iteration_as_done()
                k = k + n_discarded

                # Update the local splitting iteration counter
                k_it = self._req_db_ext().get_iteration_count()

    def _one_splitting_iteration(
        self, tdb: Database, runner: BaseRunner, k: int, k_it: int, plot_diags: bool
    ) -> tuple[bool, int]:
        """Perform a single splitting iteration.

        Args:
            tdb: the AMS database
            runner: the runner
            k: counter of discarded trajectories
            k_it: loop counter
            plot_diags: whether or not to plot diagnostics

        Returns:
            bool to trigger splitting loop break
            number of discarded trajectories
        """
        inf_msg = (
            f"### [{k_it}] AMS iteration - total # of discarded traj. {k}"
            f" - discarding {self._ams_cfg.l_j} score level(s)"
        )
        _logger.info(inf_msg)

        # Plot trajectory database scores
        if plot_diags:
            pltfile = f"Score_k{k:05}.png"
            if Path(pltfile).exists():
                wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                _logger.warning(wrn_msg)
            tdb.plot_score_functions(pltfile)

        # Get the ensemble maximums and check for early exit conditions
        early_exit, maxes = self.check_exit_splitting_loop(tdb, k, k_it)

        # Get the l_j lower scored trajectories
        # or more if equal score
        min_idx_list, min_vals = get_min_scored(maxes, self._ams_cfg.l_j)

        # Randomly select trajectory to branch from
        ancestor_idx = self.get_restart_at_random(tdb, min_idx_list)
        n_discarded = len(min_idx_list)

        # Update the database with the data of the current
        # iteration
        self._req_db_ext().append_splitting_iteration_data(
            k, n_discarded, min_idx_list, ancestor_idx, min_vals.tolist(), [np.min(maxes), np.max(maxes)]
        )

        # Query the current iteration weight
        # to compute the individual weight of each trajectory in the ensemble
        # at the end of the splitting iteration
        new_traj_weight = self._req_db_ext().weights()[-1]

        # Exit the loop if needed
        if early_exit:
            # If AMS converged, final update of the weights.
            if tdb.all_converged():
                self._req_db_ext().update_trajectories_weights()
            return True, n_discarded

        # Assemble a list of promises
        # and archive the discarded trajectories
        for i in range(n_discarded):
            # Archive
            tdb.archive_trajectory(tdb.get_traj(min_idx_list[i]))

            # Worker task
            task = [
                tdb.get_traj(ancestor_idx[i]),
                tdb.get_traj(min_idx_list[i]),
                np.max(min_vals),
                new_traj_weight,
                self._term_crit,
                self._end_date,
                tdb.pool_file(),
                tdb.path(),
            ]
            runner.make_promise(task)

        try:
            restarted_trajs = runner.execute_promises()
        except Exception as exc:
            err_msg = f"Failed to branch {n_discarded} trajectories at iteration {k_it}"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg) from exc

        # Update the trajectories in the database
        for t in restarted_trajs:
            tdb.overwrite_traj(t.id(), t)

        # Update the weights of all trajectories in the ensemble with the current
        # iteration weight
        self._req_db_ext().update_trajectories_weights()

        # If OOT or interrupted, the database was updated
        # above, but do not increment splitting index yet or
        # mark the iteration as completed.
        # The ongoing trajectories will be completed upon restarting.
        if self.out_of_time():
            warn_msg = f"Ran out of time after {k_it} splitting iterations, and {k} discarded trajectories"
            _logger.warning(warn_msg)
            return True, n_discarded

        if Path("./pyrevs_stop").exists():
            warn_msg = f"User-interupt triggered after {k_it} splitting iterations, and {k} discarded trajectories"
            _logger.warning(warn_msg)

            # Remove the stop file
            Path("./pyrevs_stop").unlink(missing_ok=True)
            return True, n_discarded

        # Return to the main loop
        # to continue splitting
        return False, n_discarded

    def compute_probability(self, tdb: Database, plot_diags: bool) -> float:
        """Compute the probability using AMS.

        Args:
            tdb: the AMS database
            plot_diags: whether or not to plot diagnostics

        Returns:
            the rare-event probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using AMS"
        _logger.info(inf_msg)

        # Generate the initial trajectory ensemble
        init_ensemble_need_work = not self._req_db_ext().init_ensemble_done()
        if init_ensemble_need_work:
            self.generate_trajectory_ensemble(tdb)

        # Check for early convergence
        all_converged = True
        for t in tdb.traj_list():
            if not t.is_converged():
                all_converged = False
                break

        if init_ensemble_need_work and all_converged:
            inf_msg = "All trajectories in the ensemble converged prior to splitting !"
            _logger.info(inf_msg)
            return 1.0

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        if self._ams_cfg.init_ensemble_only:
            warn_msg = "Stopping after the initial ensemble stage !"
            _logger.warning(warn_msg)
            return -1.0

        # Perform multilevel splitting
        if not all_converged:
            self.do_multilevel_splitting(tdb, plot_diags)

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        rare_event_probability = tdb.get_event_probability()

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

        # Load the archived trajectories data since the workers
        # discarded them but the persistent Python process did not
        # kept track
        tdb.load_archived_trajectories()

        tdb.info()

        return rare_event_probability

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        # Initialize the active ensemble
        database.init_active_ensemble()

        # Check that the database extension is initialized
        if self._db_ext is None:
            err_msg = "Database extension is not initialized ! Call initialize_database_schema() first"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        proba = self.compute_probability(database, plot_diags)
        inf_msg = f"Event probability: {proba}"
        _logger.info(inf_msg)

    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
        """Initialize database core state."""
        spec = DatabaseCoreSpec(
            ntraj=self._ams_cfg.ntrajectories,
            strategy="ams",
            deterministic=self._deterministic,
            diag_configs=diag_configs,
        )
        database.initialize_core_state(spec)

        # Setup the AMS database extension
        self._db_ext = AMSDatabaseExtension()
        self._db_ext.initialize(self._ams_cfg.nsplititer, database)
        database.attach_extension(self._db_ext)
