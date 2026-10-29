"""Tests for the pyrevs.worker functions."""

import datetime
import logging
from math import isclose
from pathlib import Path
import pytest
from pyrevs.core import Config
from pyrevs.core import CoreDB
from pyrevs.core import RuntimeConfig
from pyrevs.runner import ms_worker
from pyrevs.runner import pool_worker
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.utils.utils import setup_logger
from pyrevs.strategies.base import TimeTerminationCriterion
from pyrevs.strategies.base import LowScoreTerminationCriterion
from tests.dwmodel import DoubleWellModel
from tests.models import FailingFModel
from tests.models import SimpleFModel


def test_run_pool_worker():
    """Advance trajectory through pool_worker."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = pool_worker(t_test, [], enddate)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False


def test_run_pool_worker_with_termination():
    """Advance trajectory through pool_worker."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    term_crit = [TimeTerminationCriterion(0.01), LowScoreTerminationCriterion(-0.01)]
    t_test = pool_worker(t_test, term_crit, enddate)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False


def test_run_pool_worker_with_sql():
    """Advance trajectory through pool_worker with SQL."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}})
    poolfile = CoreDB("./test.db")
    t_test = Trajectory(0, 1.0, fmodel, cfg.load(TrajectoryConfig))
    poolfile.add_trajectory("dummy.xml", t_test.get_metadata())
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    _ = pool_worker(t_test, [], enddate, "./test.db")
    _, metadata = poolfile.fetch_trajectory(0)
    assert metadata["terminated"]
    del poolfile
    Path("./test.db").unlink()


def test_run_pool_worker_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
            "runtime": {"loglevel": "DEBUG"},
            "model": {"slow_factor": 0.03},
        }
    )
    model_params = cfg.section_dict("model")
    setup_logger(cfg.load(RuntimeConfig).loglevel)
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig), model_params=model_params)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=0.1)
    _ = pool_worker(t_test, [], enddate)
    assert "advance ran out of time" in caplog.text


def test_run_pool_worker_advanceerror():
    """Advance trajectory through pool_worker running into error."""
    fmodel = FailingFModel
    cfg = Config(
        {
            "trajectory": {"end_time": 1.0, "step_size": 0.01, "targetscore": 0.75},
        }
    )
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=1.0)
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    with pytest.raises(RuntimeError):
        _ = pool_worker(t_test, [], enddate)


def test_run_ms_worker():
    """Branch and advance trajectory through ms_worker."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}})
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance([])
    rst_test = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    b_test = ms_worker(t_test, rst_test, 0.049, 1.0, [], enddate)
    assert b_test.id() == 2
    assert isclose(b_test.score_max(), 0.1, abs_tol=1e-9)
    assert b_test.is_converged() is False


def test_run_ms_worker_with_sql():
    """Branch and advance trajectory through ms_worker."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}})
    poolfile = CoreDB("./test.db")
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = Trajectory(0, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance()
    poolfile.add_trajectory("dummy.xml", t_test.get_metadata())
    rst_test = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    poolfile.add_trajectory("dummy.xml", rst_test.get_metadata())
    _ = ms_worker(t_test, rst_test, 0.049, 1.0, [], enddate, "./test.db")
    _, metadata = poolfile.fetch_trajectory(1)
    assert metadata["terminated"]
    del poolfile
    Path("./test.db").unlink()


def test_run_ms_worker_model_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through ms_worker running out of time."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
            "runtime": {"loglevel": "DEBUG"},
            "model": {"slow_factor": 0.003},
        }
    )
    model_params = cfg.section_dict("model")
    setup_logger(cfg.load(RuntimeConfig).loglevel)
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    t_test = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params)
    t_test.advance()
    rst_test = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=0.01)
    _ = ms_worker(t_test, rst_test, 0.1, 1.0, [], enddate)
    assert "advance ran out of time" in caplog.text


def test_run_ms_worker_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through ms_worker running out of time."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
            "runtime": {"loglevel": "DEBUG"},
            "model": {"slow_factor": 0.003},
        }
    )
    model_params = cfg.section_dict("model")
    setup_logger(cfg.load(RuntimeConfig).loglevel)
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    t_test = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params)
    t_test.advance()
    rst_test = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=0.1)
    _ = ms_worker(t_test, rst_test, 0.1, 1.0, [], enddate)
    assert "MS worker ran out of time" in caplog.text


def test_run_ms_worker_advanceerror():
    """Advance trajectory through ms_worker running into error."""
    fmodel = FailingFModel
    cfg = Config({"trajectory": {"end_time": 1.0, "step_size": 0.001, "targetscore": 0.75}})
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=1.0)
    t_test = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.01)
    rst_test = Trajectory(5, 0.5, fmodel, cfg.load(TrajectoryConfig))
    with pytest.raises(RuntimeError):
        _ = ms_worker(t_test, rst_test, 0.04, 1.0, [], enddate)
