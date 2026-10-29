from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import ntpath
import shutil
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import dask
from dask.distributed import Client
from dask.distributed import Nanny
from dask.distributed import WorkerPlugin
from dask_jobqueue import SLURMCluster
from typing_extensions import Self
from pyrevs.utils import setup_logger
from .worker import worker_async

if TYPE_CHECKING:
    from collections.abc import Callable
    from dask.distributed import Worker
    from .config import RunnerConfig

_logger = logging.getLogger(__name__)


class WorkerLoggerPlugin(WorkerPlugin):
    """A plugin to configure logging on each worker."""

    def __init__(self, loglevel: str, logfile: str | None) -> None:
        """Init function pass in the params dict."""
        self._loglevel = loglevel
        self._logfile = logfile

    def setup(self, worker: Worker) -> None:
        """Configure logging on the worker.

        Args:
            worker: the dask worker
        """
        # Configure logging on each worker
        _ = worker
        setup_logger(self._loglevel, self._logfile)


class RunnerError(Exception):
    """Exception class for the runner."""


class BaseRunner(metaclass=ABCMeta):
    """An ABC for the task runners."""

    @abstractmethod
    def __init__(
        self,
        runner_cfg: RunnerConfig,
        worker_fn: Callable,
        n_workers: int = 1,
        loglevel: str = "INFO",
        logfile: str | None = None,
    ):
        """A dummy init method."""

    @abstractmethod
    def __enter__(self) -> BaseRunner:
        """To enable use of with."""

    @abstractmethod
    def __exit__(self, *args: object) -> None:
        """Executed leaving with scope."""

    @abstractmethod
    def make_promise(self, task: list[Any]) -> None:
        """Log a new task to the list of task to tackle."""

    @abstractmethod
    def execute_promises(self) -> Any:
        """Execute the list of promises."""

    @abstractmethod
    def n_workers(self) -> int:
        """Return the number of workers in the runner."""


class AsIORunner(BaseRunner):
    """A task runner class based on asyncIO.

    An runner that relies on asyncio to schedule
    tasks concurrently in worker processes.
    Tasks are added to an internal queue from
    which worker can take them and put the results
    back into result queue.

    Attributes:
        _queue: an asyncio.Queue() to place the tasks in
        _rqueue: an asyncio.Queue() where the results are returned
        _n_workers: the number of workers in the runner
        _sync_worker: the synchrone worker function
        _async_worker: the asynchrone worker function
        _loop: the event loop associated with the workers
        _executor: an executor for the worker to work in
        _workers: a list of worker tasks
    """

    def __init__(
        self,
        runner_cfg: RunnerConfig,
        worker_fn: Callable,
        n_workers: int = 1,
        loglevel: str = "INFO",
        logfile: str | None = None,
    ):
        """Init the asyncio task runner.

        Args:
            runner_cfg: a RunnerConfig dataclass
            worker_fn: a synchronous worker function
            async_wk: an asynchronous worker function
            n_workers: number of workers
            loglevel: optional logging level
            logfile: optional logging file
        """
        _ = runner_cfg
        self._loglevel = loglevel
        self._logfile = logfile
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._rqueue: asyncio.Queue[Any] = asyncio.Queue()
        self._n_workers: int = n_workers
        self._sync_worker = worker_fn
        self._async_worker = worker_async
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: concurrent.futures.Executor | None = None
        self._workers: list[asyncio.Task[Any]] | None = None

    def __enter__(self) -> Self:
        """To enable use of with."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self

    def __exit__(self, *args: object) -> None:
        """Executed leaving with scope."""
        if self._workers:
            for w in self._workers:
                w.cancel()
        if self._executor:
            self._executor.shutdown()
        if self._loop:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            asyncio.set_event_loop(None)

    async def add_task(self, task: list[Any]) -> None:
        """Append a task to the queue."""
        await self._queue.put([self._sync_worker, *task])

    def make_promise(self, task: list[Any]) -> None:
        """A synchronous wrapper to add_task."""
        asyncio.run(self.add_task(task))

    async def run_tasks(self) -> list[Any]:
        """Create worker tasks and run.

        Initialize the executor and setup the workers (tasks) if not
        already done.
        The join() task is created seperately and awaited with the others in
        order to catch any exception coming from the workers as they are generated
        and stop everything as soon as one task fails.
        """
        if not self._workers:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._n_workers, initializer=setup_logger, initargs=(self._loglevel, self._logfile)
            )
            self._workers = [
                asyncio.create_task(self._async_worker(self._queue, self._rqueue, self._executor))
                for _ in range(self._n_workers)
            ]

        # Create a separate task for the join()
        # and check the tasks status as they are completed
        join_task = asyncio.create_task(self._queue.join())
        done, _ = await asyncio.wait([join_task, *self._workers], return_when=asyncio.FIRST_COMPLETED)

        # If a task raise an exception, cancel all other tasks
        # and re-raise.
        for task in done:
            if task != join_task and task.exception():
                excep = task.exception()
                for t in self._workers:
                    if not t.done():
                        t.cancel()

                join_task.cancel()

                if excep is None:
                    err_msg = "Caught an 'odd' exception in tasks !"
                    raise RunnerError(err_msg)

                raise excep

        # Keep assembling the results list
        res = []
        while not self._rqueue.empty():
            res.append(await self._rqueue.get())

        return res

    def execute_promises(self) -> Any:
        """A synchronous wrapper to run_tasks."""
        if not self._loop:
            err_msg = "AsIORunner has not been initialized."
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        try:
            res = self._loop.run_until_complete(self.run_tasks())
        except Exception:
            err_msg = "Error in AsIORunner while executing promises."
            _logger.exception(err_msg)
            raise
        else:
            return res

    def n_workers(self) -> int:
        """Return the number of workers in the runner."""
        return self._n_workers


# An async function to build and start a worker alongside
# the scheduler when running with SLURMCluster
async def _start_local_worker(scheduler_address: str, nthreads: int = 1, nranks: int = 1) -> Nanny:
    """Start a worker on the same node as the scheduler.

    Args:
        scheduler_address: the address of the scheduler
        nthreads: number of threads
        nranks: number of ranks
    """
    worker = Nanny(
        scheduler_address,
        nthreads=nthreads,
        resources={"mpi_ranks": nranks},
    )
    await worker.start()
    return worker


class DaskRunner(BaseRunner):
    """A task runner class based on Dask.

    An runner that relies on dask to schedule
    a tasks concurrently in workers.
    """

    def __init__(
        self,
        runner_cfg: RunnerConfig,
        worker_fn: Callable,
        n_workers: int = 1,
        loglevel: str = "INFO",
        logfile: str | None = None,
    ):
        """Start the Dask cluster and client.

        Args:
            runner_cfg: a RunnerConfig dataclass
            worker_fn: a synchronous worker function
            async_wk: an asynchronous worker function
            n_workers: number of workers
            loglevel: optional logging level
            logfile: optional logging file
        """
        dask_cfg = runner_cfg.dask_config
        self._n_workers: int = n_workers
        self._sync_worker = worker_fn
        self._tlist: list[Any] = []

        backend = dask_cfg.backend
        if backend == "local":
            self.client = Client(threads_per_worker=1, n_workers=self._n_workers)
            self.cluster = None

        elif backend == "slurm":
            self.slurm_config_file = dask_cfg.slurm_config_file
            if self.slurm_config_file:
                if not Path(self.slurm_config_file).exists():
                    err_msg = f"Specified slurm_config_file do not exists: {self.slurm_config_file}"
                    _logger.error(err_msg)
                    raise FileNotFoundError(err_msg)

                config_file = ntpath.basename(self.slurm_config_file)
                shutil.move(
                    self.slurm_config_file,
                    f"~/.config/dask/{config_file}",
                )
                self.cluster = SLURMCluster()
            else:
                ntasks_per_node = dask_cfg.ntasks_per_node if dask_cfg.ntasks_per_node > 0 else dask_cfg.ntasks_per_job
                self.cluster = SLURMCluster(
                    queue=dask_cfg.queue,
                    cores=dask_cfg.ncores_per_worker,
                    memory="144GB",
                    walltime=dask_cfg.worker_walltime,
                    processes=1,
                    interface="ib0",
                    job_script_prologue=dask_cfg.job_prologue,
                    job_extra_directives=[
                        f"--ntasks={dask_cfg.ntasks_per_job}",
                        f"--tasks-per-node={ntasks_per_node}",
                        "--exclusive",
                    ],
                    job_directives_skip=["--cpus-per-task=", "--mem"],
                )
            self.client = Client(self.cluster)

            # Start a local worker alongside the
            # scheduler slurm job and scale the cluster
            # accordingly
            if dask_cfg.one_worker_with_scheduler:
                cluster_nworkers = max(0, self._n_workers - 1)
                self.cluster.scale(jobs=cluster_nworkers)

                if self.client.loop is None:
                    err_msg = "Dask client loop is None."
                    _logger.error(err_msg)
                    raise RuntimeError(err_msg)

                asyncio_loop = self.client.loop.asyncio_loop  # type: ignore[attr-defined]

                # Submit the async routine to the client's background event loop and wait for it
                future = asyncio.run_coroutine_threadsafe(
                    _start_local_worker(self.cluster.scheduler_address, nthreads=1, nranks=dask_cfg.ntasks_per_job),
                    asyncio_loop,
                )

                # This blocks the main thread for a moment until the local worker is fully up
                self.local_worker = future.result(timeout=30)
            else:
                self.cluster.scale(jobs=self._n_workers)

        else:
            err_msg = f"Unknown [dask] backend: {backend}"
            _logger.error(err_msg)
            raise RunnerError(err_msg)

        # Setup the worker logging
        self.client.register_plugin(WorkerLoggerPlugin(loglevel, logfile))

    def __enter__(self) -> Self:
        """To enable use of with."""
        return self

    def __exit__(self, *args: object) -> None:
        """Executed leaving with scope."""
        if hasattr(self, "local_worker") and self.local_worker:
            if self.client.loop is None:
                err_msg = "Dask client loop is None."
                _logger.error(err_msg)
                raise RuntimeError(err_msg)

            asyncio_loop = self.client.loop.asyncio_loop  # type: ignore[attr-defined]
            try:
                future = asyncio.run_coroutine_threadsafe(self.local_worker.close(), asyncio_loop)
                future.result(timeout=5)
            except Exception as e:
                err_msg = f"Error while closing local worker: {e}"
                _logger.exception(err_msg)
        if self.cluster:
            self.cluster.close()
        self.client.close()

    def make_promise(self, task: list[Any]) -> None:
        """Append a task to the internal task list."""
        self._tlist.append(dask.delayed(self._sync_worker)(*task))

    def just_delay(self, obj: Any) -> Any:
        """Delay an object."""
        return dask.delayed(obj)

    def execute_promises(self) -> Any:
        """Execute a list of promises.

        Args:
            list_of_p: a list of dask promises

        Returns:
            A list with the return argument of each promised task.

        Raises:
            Exception if compute fails (raise internal error)
        """
        try:
            res = list(dask.compute(*self._tlist))
        except Exception:
            err_msg = "Error in DaskRunner while executing promises."
            _logger.exception(err_msg)
            raise
        else:
            self._tlist.clear()
            return res

    def n_workers(self) -> int:
        """Return the number of workers in the runner."""
        return self._n_workers


def make_runner(
    runner_cfg: RunnerConfig,
    worker_fn: Callable,
    loglevel: str = "INFO",
    logfile: str | None = None,
    max_workers: int = -1,
) -> BaseRunner:
    """Factory that instantiates a configured runner.

    Args:
        runner_cfg: a config mapping for the runner
        worker_fn: a worker function
        loglevel: logging level
        logfile: logging file
        max_workers: maximum number of workers
    """
    runner_type = runner_cfg.type.lower()

    runner_map: dict[str, type[BaseRunner]] = {
        "asyncio": AsIORunner,
        "dask": DaskRunner,
    }

    n_workers = min(runner_cfg.nworkers, max_workers) if max_workers > 0 else runner_cfg.nworkers

    if runner_type not in runner_map:
        err_msg = f"Unknown runner type: {runner_type}"
        _logger.error(err_msg)
        raise ValueError(err_msg)

    runner_cls = runner_map[runner_type]

    return runner_cls(
        runner_cfg=runner_cfg,
        worker_fn=worker_fn,
        n_workers=n_workers,
        loglevel=loglevel,
        logfile=logfile,
    )
