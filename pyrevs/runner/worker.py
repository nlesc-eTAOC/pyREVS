"""A set of functions used by pyREVS workers."""

from __future__ import annotations
import asyncio
import datetime
import functools
import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from pyrevs.core import CoreDB
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory.trajectory import UserInterruptError
from pyrevs.trajectory.trajectory import WallTimeLimitError
from pyrevs.utils.utils import is_windows_os

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Callable
    from pyrevs.strategies.base import TerminationCriterion


_logger = logging.getLogger(__name__)


# CoreDB handles of the worker process, keyed by path
_coredb_cache: dict[str, CoreDB] = {}


def get_coredb(trajdb_path: str) -> CoreDB:
    """Return a handle to the SQL database of trajectories.

    Opening the database is costly compared to the work done by a worker on
    small models: the handle is cached in worker (child) processes, which only
    live for the duration of a runner. It is not cached in the main process,
    which might run several samplers on the same path, nor on Windows where open
    handles might lock the database file.

    Args:
        trajdb_path: the path to the SQL database
    """
    if is_windows_os() or multiprocessing.parent_process() is None:
        return CoreDB(trajdb_path)
    if trajdb_path not in _coredb_cache:
        _coredb_cache[trajdb_path] = CoreDB(trajdb_path)
    return _coredb_cache[trajdb_path]


def update_trajectory_in_sql(traj: Trajectory, trajdb: CoreDB | None = None, db_path: str | None = None) -> None:
    """Wrapper for update SQL trajectory info.

    Args:
        trajdb: the SQL database to update
        traj: the traj to get the information from
        db_path: an optional pyREVS database path
    """
    if trajdb:
        checkfile_str = (
            traj.get_checkfile().relative_to(Path(db_path)).as_posix() if db_path else traj.get_checkfile().as_posix()
        )
        trajdb.update_trajectory(traj.id(), checkfile_str, traj.get_metadata())


def traj_advance_with_exception(
    traj: Trajectory,
    termination_criteria: list[TerminationCriterion],
    walltime: float,
    trajdb: CoreDB | None = None,
    db_path: str | None = None,
) -> Trajectory:
    """Advance a trajectory with exception handling.

    Args:
        traj: a trajectory
        termination_criteria: a list of termination criterion for the advance call
        walltime: the time limit to advance the trajectory
        trajdb: a handle to the SQL database
        db_path: an optional path to the run database

    Returns:
        The updated trajectory
    """
    try:
        traj.advance(walltime=walltime, termination_criteria=termination_criteria)

    except UserInterruptError:
        warn_msg = f"Trajectory {traj.idstr()} advance ran interrupted with file !"
        _logger.warning(warn_msg)

    except WallTimeLimitError:
        warn_msg = f"Trajectory {traj.idstr()} advance ran out of time !"
        _logger.warning(warn_msg)

    except Exception:
        err_msg = f"Trajectory {traj.idstr()} advance ran into an error !"
        _logger.exception(err_msg)
        raise

    finally:
        # Update the SQL database
        if trajdb:
            if traj.is_terminated():
                trajdb.mark_trajectory_as_completed(traj.id())
            else:
                trajdb.release_trajectory(traj.id())
            update_trajectory_in_sql(traj, trajdb, db_path)

        # Trigger a checkfile dump if we are provided with
        # a database path
        if db_path:
            traj.store()

    return traj


def pool_worker(
    traj: Trajectory,
    termination_criteria: list[TerminationCriterion],
    end_date: datetime.date,
    trajdb_path: str | None = None,
    db_path: str | None = None,
) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        termination_criteria: a list of termination criterion for the advance call
        end_date: the time limit to advance the trajectory
        trajdb_path: an optional path to the SQL database
        db_path: an optional path to the run database

    Returns:
        The updated trajectory
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    if wall_time > 0.0 and not traj.is_terminated():
        # Try to lock the trajectory in the DB
        trajdb = None
        if trajdb_path:
            trajdb = get_coredb(trajdb_path)
            get_to_work = trajdb.lock_trajectory(traj.id(), allow_completed_lock=True)
            if not get_to_work:
                return traj

        inf_msg = f"Advancing {traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = traj_advance_with_exception(traj, termination_criteria, wall_time, trajdb, db_path)

    return traj


def ms_worker(
    from_traj: Trajectory,
    rst_traj: Trajectory,
    min_val: float,
    new_weight: float,
    termination_criteria: list[TerminationCriterion],
    end_date: datetime.date,
    trajdb_path: str | None = None,
    db_path: str | None = None,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        from_traj: a trajectory to restart from
        rst_traj: the trajectory being restarted
        min_val: the value of the score function to restart from
        new_weight: the weight of the new child trajectory
        termination_criteria: a list of termination criterion for the advance call
        end_date: the time limit to advance the trajectory
        trajdb_path: a path to the SQL database
        db_path: an optional path to the run database
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    trajdb = None
    if trajdb_path:
        trajdb = get_coredb(trajdb_path)

    if wall_time > 0.0:
        # Try to lock the trajectory in the DB
        if trajdb:
            get_to_work = trajdb.lock_trajectory(rst_traj.id(), allow_completed_lock=True)
            if not get_to_work:
                err_msg = f"Unable to lock trajectory {rst_traj.id()} for branching"
                _logger.error(err_msg)
                raise RuntimeError(err_msg)

        inf_msg = f"    Restarting [{rst_traj.id()}] from {from_traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val, new_weight)

        # The branched trajectory has a new checkfile
        # Update the database to point to the latest one.
        update_trajectory_in_sql(traj, trajdb, db_path)

        return traj_advance_with_exception(traj, termination_criteria, wall_time, trajdb, db_path)

    traj = Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val, new_weight)

    warn_msg = "MS worker ran out of time before advancing trajectory!"
    _logger.warning(warn_msg)

    # The branched trajectory has a new checkfile, even if haven't advanced yet
    # Update the database to point to the latest one.
    update_trajectory_in_sql(traj, trajdb, db_path)

    return traj


async def worker_async(
    queue: asyncio.Queue[tuple[Callable[..., Any], Trajectory, float, bool, str]],
    res_queue: asyncio.Queue[asyncio.Future[Trajectory]],
    executor: concurrent.futures.Executor,
) -> None:
    """An async worker for the asyncio taskrunner.

    It wraps the call to one of the above worker functions
    with access to the queue.

    Args:
        queue: a queue from which to get tasks
        res_queue: a queue to put the results in
        executor: an executor to launch the work in
    """
    while True:
        func, *work_unit = await queue.get()
        loop = asyncio.get_running_loop()
        traj: asyncio.Future[Trajectory] = await loop.run_in_executor(
            executor,
            functools.partial(func, *work_unit),
        )
        await res_queue.put(traj)
        queue.task_done()
