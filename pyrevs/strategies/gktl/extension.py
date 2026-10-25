"""An extension class for the GKTL strategy."""

import gc
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from pyrevs.database import Database
from pyrevs.database import StrategyDatabaseExtension

_logger = logging.getLogger(__name__)


class GKTLDatabaseExtension(StrategyDatabaseExtension):
    """An extension class for the GKTL strategy.

    An extension of the database to store GKTL-specific
    data. To be able to checkpoint a sampling run with GKTL,
    and to compute unbiased estimates from the biased ensemble,
    one need to store/read the following attributes.

    At each resampling step n, the clone weights are W_i = exp(k * dA_i)
    and the normalization factor is R_n = mean(W_i). The (unbiased) estimate
    of the expectation of an observable O of the trajectories is then:

        E[O] = 1/N sum_i O_i exp(-log_tilt_i) prod_n R_n

    where log_tilt_i = k * sum_n dA_i,n along the ancestry of clone i.

    Attributes:
        _iteration_count: number of completed resampling steps
        _cur_time: time of the last resampling step
        _k: the bias amplitude
        _log_norm: log of the normalization factor R_n of each resampling step
        _log_tilt: accumulated log-tilt of each clone along its ancestry
        _window_start: start time of the current resampling interval for each clone
        _ancestors: parent index of each clone, for each resampling step
        _times: time of each resampling step
    """

    def __init__(self) -> None:
        self._iteration_count = 0
        self._cur_time = 0.0
        self._k = 0.0
        self._log_norm: list[float] = []
        self._log_tilt: list[float] = []
        self._window_start: list[float] = []
        self._ancestors: list[list[int]] = []
        self._times: list[float] = []

    def _metadata_file(self) -> Path:
        """Return the path to the extension metadata file."""
        return Path(self._tdb.name()) / "gktl_metadata.json"

    def initialize(self, tdb: Database, k: float) -> None:
        """Initialize the GKTL database extension.

        If an on-disk database with GKTL metadata already exists,
        the extension state is read from disk to enable restarts.

        Args:
            tdb: the core trajectory database
            k: the bias amplitude
        """
        self._tdb = tdb
        if tdb.to_disk() and self._metadata_file().exists():
            self.deserialize()
            return

        self._k = k
        self._log_tilt = [0.0] * tdb.n_traj()
        self._window_start = [0.0] * tdb.n_traj()

    def initialize_from_database(self, tdb: Database) -> None:
        """Initialize the GKTL database extension from an existing database.

        Args:
            tdb: the core trajectory database
        """
        self._tdb = tdb
        self.deserialize()

    def serialize(self) -> None:
        """Serialize the extension."""
        if not self._tdb.to_disk():
            return
        data = {
            "iteration_count": self._iteration_count,
            "cur_time": self._cur_time,
            "k": self._k,
            "log_norm": self._log_norm,
            "log_tilt": self._log_tilt,
            "window_start": self._window_start,
            "ancestors": self._ancestors,
            "times": self._times,
        }
        with self._metadata_file().open("w") as f:
            json.dump(data, f, indent=2)

    def deserialize(self) -> None:
        """Deserialize the extension."""
        with self._metadata_file().open("r") as f:
            data = json.load(f)
        self._iteration_count = data["iteration_count"]
        self._cur_time = data["cur_time"]
        self._k = data["k"]
        self._log_norm = data["log_norm"]
        self._log_tilt = data["log_tilt"]
        self._window_start = data["window_start"]
        self._ancestors = data["ancestors"]
        self._times = data["times"]

    def iteration_count(self) -> int:
        """Return the number of completed resampling steps."""
        return self._iteration_count

    def current_time(self) -> float:
        """Return the time of the last resampling step."""
        return self._cur_time

    def window_start(self) -> list[float]:
        """Return the start time of the current resampling interval for each clone."""
        return self._window_start

    def ancestors(self) -> list[list[int]]:
        """Return the parent index of each clone, for each resampling step."""
        return self._ancestors

    def log_normalization(self) -> list[float]:
        """Return the log of the normalization factor of each resampling step."""
        return self._log_norm

    def record_resampling(
        self, delta_a: npt.NDArray[np.number], parents: list[int], window_start: list[float], time: float
    ) -> None:
        """Record a resampling step and update the clones tilt.

        Args:
            delta_a: the observable increment of each clone over the interval (before resampling)
            parents: the parent index of each clone after resampling
            window_start: the start time of the next interval for each clone
            time: the time of the resampling step
        """
        log_w = self._k * np.asarray(delta_a, dtype=float)
        log_w_max = float(np.max(log_w))
        log_r = log_w_max + float(np.log(np.mean(np.exp(log_w - log_w_max))))

        self._log_tilt = [self._log_tilt[p] + float(log_w[p]) for p in parents]
        self._log_norm.append(log_r)
        self._ancestors.append(list(parents))
        self._times.append(time)
        self._window_start = list(window_start)
        self._cur_time = time
        self._iteration_count += 1
        self.serialize()

    def clone_weights(self) -> npt.NDArray[np.number]:
        """Return the estimator weight of each clone in the current ensemble.

        The weights sum to one in expectation (exactly for k = 0).
        """
        log_tilt = np.asarray(self._log_tilt, dtype=float)
        weights: npt.NDArray[np.number] = np.exp(np.sum(self._log_norm) - log_tilt) / len(log_tilt)
        return weights

    def _clones_metadata(self) -> list[dict[str, Any]]:
        """Return the metadata of the clones in the current ensemble."""
        pool_db = self._tdb.get_pool_db()
        if pool_db is None:
            err_msg = "GKTL extension requires the trajectory pool database"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return [pool_db.fetch_trajectory(i)[1] for i in range(len(self._log_tilt))]

    def get_exceedance_probability(self, threshold: float) -> float:
        """Return the probability that the score maximum exceeds a threshold.

        Args:
            threshold: the score threshold

        Returns:
            the unbiased estimate of P(max score >= threshold)
        """
        weights = self.clone_weights()
        exceed = np.array([m["score_max"] >= threshold for m in self._clones_metadata()], dtype=float)
        return float(np.sum(weights * exceed))

    def get_event_probability(self) -> float:
        """Return the event probability.

        The event is the convergence of the trajectory (i.e. reaching the
        target score) within the sampling end time.
        """
        weights = self.clone_weights()
        converged = np.array([m["converged"] for m in self._clones_metadata()], dtype=float)
        return float(np.sum(weights * converged))

    def __del__(self) -> None:
        """Destructor of the GKTL extension.

        Force deletion of SQL accesses for Windows.
        """
        if hasattr(self, "_tdb"):
            del self._tdb
        gc.collect()
