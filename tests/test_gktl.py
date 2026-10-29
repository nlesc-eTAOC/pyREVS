"""Tests for the pyrevs GKTL strategy."""

import json
import shutil
from math import isclose
from pathlib import Path
import numpy as np
import pytest
import toml
from pyrevs.core import Config
from pyrevs.database import Database
from pyrevs.sampler import build_sampler
from pyrevs.strategies.base import TimeInterruptionCriterion
from pyrevs.strategies.gktl import GKTLConfig
from pyrevs.strategies.gktl.gktl import observable_increment
from pyrevs.strategies.gktl.gktl import select_clones
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.utils.utils import is_windows_os
from tests.dwmodel import DoubleWellModel
from tests.models import SimpleFModel


@pytest.mark.parametrize(
    "params",
    [
        {"ntrajectories": 0, "end_time": 1.0, "resampling_interval": 0.1},
        {"ntrajectories": 10, "end_time": -1.0, "resampling_interval": 0.1},
        {"ntrajectories": 10, "end_time": 1.0, "resampling_interval": 2.0},
        {"ntrajectories": 10, "end_time": 1.0, "resampling_interval": 0.1, "k": float("inf")},
        {"ntrajectories": 10, "end_time": 1.0, "resampling_interval": 0.1, "bias_observable": "max"},
    ],
)
def test_gktl_config_invalid(params):
    """Test GKTL config validation."""
    cfg = Config({"gktl": params}).load(GKTLConfig)
    with pytest.raises(ValueError):
        cfg.validate()


def test_select_clones_unbiased():
    """Test that equal weights keep all clones in place."""
    rng = np.random.default_rng(0)
    parents = select_clones(np.zeros(20), rng)
    assert parents == list(range(20))


def test_select_clones_biased():
    """Test selection with very uneven weights."""
    rng = np.random.default_rng(0)
    log_w = np.zeros(10)
    log_w[3] = 10.0
    parents = select_clones(log_w, rng)
    assert len(parents) == 10
    assert all(p == 3 for p in parents)


def test_select_clones_survivors_keep_slot():
    """Test that surviving clones keep their own slot."""
    rng = np.random.default_rng(1)
    log_w = rng.normal(size=50)
    parents = select_clones(log_w, rng)
    assert len(parents) == 50
    survivors = set(parents)
    for j, p in enumerate(parents):
        if j in survivors:
            assert p == j
        else:
            assert p in survivors


def test_observable_increment():
    """Test observable increment on the simple model (score = 10 * t)."""
    cfg = Config({"trajectory": {"step_size": 0.001, "targetscore": 0.95}})
    traj = Trajectory(1, 1.0, SimpleFModel, cfg.load(TrajectoryConfig))
    traj.advance(t_end=0.02)
    assert isclose(observable_increment(traj, 0.01, "increment"), 0.1, abs_tol=1e-9)
    assert isclose(observable_increment(traj, 0.01, "integral"), 0.0015, abs_tol=1e-9)
    assert observable_increment(traj, 0.03, "integral") == 0.0


def test_interruption_no_extra_step():
    """Test that an already met interruption criterion prevents stepping."""
    cfg = Config({"trajectory": {"step_size": 0.001, "targetscore": 0.95}})
    traj = Trajectory(1, 1.0, SimpleFModel, cfg.load(TrajectoryConfig))
    crit = [TimeInterruptionCriterion(0.01)]
    traj.advance(termination_criteria=crit)
    assert isclose(traj.current_time(), 0.01, abs_tol=1e-9)
    assert not traj.is_terminated()
    nsteps = traj.get_computed_steps_count()
    traj.advance(termination_criteria=crit)
    assert traj.get_computed_steps_count() == nsteps


@pytest.mark.parametrize("sparse_freq", [1, 7])
def test_clone_diverges_from_ancestor(sparse_freq):
    """Test that a clone holds the ancestor history and then diverges."""
    cfg = Config({"trajectory": {"step_size": 0.01, "targetscore": 2.0, "sparse_freq": sparse_freq}})
    traj_cfg = cfg.load(TrajectoryConfig)
    params = {"noise_amplitude": 0.8}
    ancestor = Trajectory(1, 1.0, DoubleWellModel, traj_cfg, model_params=params, deterministic=True)
    ancestor.advance(t_end=0.5)
    killed = Trajectory(2, 1.0, DoubleWellModel, traj_cfg, model_params=params, deterministic=True)
    clone = Trajectory.clone_from_trajectory(ancestor, killed, 0.5)
    assert clone.id() == 2
    assert clone.weight() == 0.5

    # Advance both: the clone first replays the ancestor history (if sparse)
    ancestor_scores = ancestor.get_score_array().copy()
    ancestor.advance(t_end=1.0)
    clone.advance(t_end=1.0)
    n_hist = len(ancestor_scores)
    assert np.allclose(clone.get_score_array()[:n_hist], ancestor_scores)
    assert not np.allclose(clone.get_score_array()[n_hist:], ancestor.get_score_array()[n_hist:])


def _gktl_input(db_path: str | None = None, k: float = 0.0) -> dict:
    params = {
        "sampler": {"strategy": "gktl"},
        "runtime": {"loglevel": "WARNING"},
        "gktl": {"ntrajectories": 20, "end_time": 0.02, "resampling_interval": 0.005, "k": k},
        "runner": {"type": "asyncio", "nworkers": 2},
        "trajectory": {"step_size": 0.001, "targetscore": 0.15},
    }
    if db_path:
        params["database"] = {"path": db_path}
    return params


def test_simple_model_gktl():
    """Test GKTL with the simple model: all trajectories converge."""
    with Path("input.toml").open("w") as f:
        toml.dump(_gktl_input(k=1.0), f)
    sampler = build_sampler(fmodel_t=SimpleFModel, a_args=[])
    sampler.run()
    assert isclose(sampler.database.get_event_probability(), 1.0, abs_tol=1e-9)
    del sampler
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_gktl_with_db_and_rerun():
    """Test GKTL on disk, then re-running on the completed database."""
    with Path("input.toml").open("w") as f:
        toml.dump(_gktl_input("gktlTest.tdb", k=1.0), f)
    sampler = build_sampler(fmodel_t=SimpleFModel, a_args=[])
    sampler.run()
    del sampler

    with Path("gktlTest.tdb/gktl_metadata.json").open("r") as f:
        data = json.load(f)
    n_resampling = data["iteration_count"]
    assert n_resampling > 0
    assert len(data["ancestors"]) == n_resampling

    # Re-running on the completed database does not resample again
    sampler = build_sampler(fmodel_t=SimpleFModel, a_args=[])
    sampler.run()
    assert isclose(sampler.database.get_event_probability(), 1.0, abs_tol=1e-9)
    del sampler
    with Path("gktlTest.tdb/gktl_metadata.json").open("r") as f:
        assert json.load(f)["iteration_count"] == n_resampling

    tdb = Database.load(Path("gktlTest.tdb"))
    del tdb
    Path("input.toml").unlink(missing_ok=True)
    if not is_windows_os():
        shutil.rmtree("gktlTest.tdb")


def test_doublewell_gktl_deterministic():
    """Test GKTL on the doublewell model: bias, resampling and reproducibility."""
    params = {
        "sampler": {"strategy": "gktl", "deterministic": True},
        "runtime": {"loglevel": "WARNING"},
        "gktl": {"ntrajectories": 30, "end_time": 3.0, "resampling_interval": 0.25, "k": 2.0},
        "runner": {"type": "asyncio", "nworkers": 2},
        "model": {"noise_amplitude": 0.8},
        "trajectory": {"step_size": 0.01, "targetscore": 0.6},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params, f)

    probas = []
    for _ in range(2):
        sampler = build_sampler(fmodel_t=DoubleWellModel, a_args=[])
        sampler.run()
        probas.append(sampler.database.get_event_probability())
        assert sampler.database.archived_traj_list_len() > 0
        del sampler

    assert 0.0 < probas[0] < 1.0
    assert isclose(probas[0], probas[1], rel_tol=1e-12)
    Path("input.toml").unlink(missing_ok=True)
