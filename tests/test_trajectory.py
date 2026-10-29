"""Tests for the pyrevs.trajectory class."""

from dataclasses import FrozenInstanceError
from math import isclose
from pathlib import Path
import numpy as np
import pytest
from pyrevs.core import Config
from pyrevs.core import ForwardModelBaseClass
from pyrevs.core import Snapshot
from pyrevs.diagnostics import DiagnosticAnalyst
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.utils.utils import moving_avg
from tests.dwmodel import DoubleWellModel
from tests.models import SimpleFModel


def test_init_snapshot():
    """Test initialization of a snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy", state="State")
    assert snap.time == 0.1
    assert snap.has_state


def test_init_snapshot_nostate():
    """Test initialization of a stateless snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy")
    assert not snap.has_state


def test_init_snapshot_negtime():
    """Test initialization with a negative time."""
    with pytest.raises(ValueError):
        _ = Snapshot(time=-0.1, score=0.1, noise="Noisy")


def test_modifying_snap():
    """Test modifying a snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy", state="State")
    with pytest.raises(FrozenInstanceError):
        snap.state = "OtherState"


def test_init_missing_basic_inputs():
    """Test lack of minimal params in TrajectoryConfig."""
    parameters = {}
    cfg = Config(parameters)
    with pytest.raises(ValueError):
        cfg.load(TrajectoryConfig).validate()


def test_init_baseclasserror():
    """Test using base class fmodel during trajectory creation."""
    fmodel = ForwardModelBaseClass
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    with pytest.raises(TypeError):
        _ = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))


def test_init_blank_traj():
    """Test blank trajectory creation."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    assert t_test.id() == 1
    assert t_test.idstr() == "traj000001_0000"
    assert t_test.current_time() == 0.0
    assert t_test.score_max() == -1000000000000.0


def test_init_parametrized_traj():
    """Test parametrized trajectory creation."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.set_workdir(Path())
    assert t_test.step_size() == 0.01


def test_restart_empty_traj():
    """Test (empty) trajectory restart."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    from_traj = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_traj = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.1, 0.25)
    assert rst_test.current_time() == 0.0


def test_simple_model_traj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.01)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False
    t_test.advance()
    assert t_test.is_converged() is True


def test_simple_model_traj_with_diag():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config(
        {
            "trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25},
            "testd": {"score_min": 0.0, "score_max": 0.25, "n_levels": 11},
        }
    )
    diagdict = {"testd": cfg.section("testd")}
    t_test_1 = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), diag_configs=diagdict)
    t_test_1.advance()
    t_test_2 = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), diag_configs=diagdict)
    t_test_2.advance()
    analyst = DiagnosticAnalyst("./diagDB.db")
    _ = analyst.get_all_diagnostic_data("testd")
    dstat = analyst.compute_weighted_stats("testd")
    assert dstat[0.0]["mean"] == 42.0
    _ = analyst.get_conditional_means("testd")
    analyst = None
    Path("./diagDB.db").unlink(missing_ok=True)


def test_branch_simple_model_traj():
    """Test branching a trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.45}})
    t_ancestor = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_ancestor.advance()
    assert t_ancestor.get_computed_steps_count() == 200
    t_branched = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, 0.1, 0.25)
    assert t_branched.get_computed_steps_count() == 0
    t_branched.advance()
    assert t_branched.get_computed_steps_count() == 149


@pytest.mark.parametrize("sparse_freq", [1, 4])
def test_branch_diverges_at_level_crossing(sparse_freq: int):
    """Test that a branched trajectory diverges right after the first snapshot above the level."""
    fmodel = DoubleWellModel
    cfg = Config({"trajectory": {"end_time": 3.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": sparse_freq}})
    t_ancestor = Trajectory(
        1, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params={"noise_amplitude": 1.0}, deterministic=True
    )
    t_ancestor.advance()
    level = 0.5 * t_ancestor.score_max()
    ancestor_scores = t_ancestor.get_score_array()
    high_idx = int(np.argmax(ancestor_scores > level))
    t_branched = Trajectory(
        2, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params={"noise_amplitude": 1.0}, deterministic=True
    )
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, level, 0.25)

    # The last snapshot holds the noise of the next model step
    assert np.array_equal(t_branched._snaps[-1].noise, t_branched._fmodel.noise)

    t_branched.advance(nstep_end=high_idx + 2)
    branched_scores = t_branched.get_score_array()
    assert np.array_equal(branched_scores[: high_idx + 1], ancestor_scores[: high_idx + 1])
    assert branched_scores[high_idx + 1] != ancestor_scores[high_idx + 1]


def _sparse_ancestor_with_tail_crossing(cfg: Config) -> tuple[Trajectory, float, int]:
    """Return a sparse trajectory and a level first crossed after its last state."""
    for tid in range(1, 50):
        t_ancestor = Trajectory(
            tid,
            0.5,
            DoubleWellModel,
            cfg.load(TrajectoryConfig),
            model_params={"noise_amplitude": 0.8},
            deterministic=True,
        )
        t_ancestor.advance()
        scores = t_ancestor.get_score_array()
        last_state_idx = t_ancestor.get_last_state_id()
        # The level is first crossed after the last state, and before the last snapshot
        level = np.max(scores[: last_state_idx + 1])
        high_idx = int(np.argmax(scores > level))
        if last_state_idx < high_idx < len(scores) - 1:
            return t_ancestor, level, high_idx
    pytest.fail("No trajectory with a level crossing after its last state")


def test_branch_restored_sparse_traj_crossing_in_tail(tmp_path: Path):
    """Test branching a restored sparse trajectory, with the level crossing after its last state."""
    cfg = Config({"trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10}})
    t_ancestor, level, high_idx = _sparse_ancestor_with_tail_crossing(cfg)
    ancestor_scores = t_ancestor.get_score_array()
    chkfile = tmp_path / "ancestor.xml"
    t_ancestor.store(chkfile)
    t_restored = Trajectory.restore_from_checkfile(
        chkfile,
        t_ancestor.get_metadata(),
        DoubleWellModel,
        cfg.load(TrajectoryConfig),
        model_params={"noise_amplitude": 0.8},
    )
    # The restored trajectory holds its complete history
    assert np.allclose(t_restored.get_score_array(), ancestor_scores, rtol=0.0, atol=1e-12)

    t_branched = Trajectory(
        100, 0.5, DoubleWellModel, cfg.load(TrajectoryConfig), model_params={"noise_amplitude": 0.8}, deterministic=True
    )
    t_branched = Trajectory.branch_from_trajectory(t_restored, t_branched, level, 0.25)
    assert t_branched.score_max() > level

    t_branched.advance(nstep_end=high_idx + 2)
    branched_scores = t_branched.get_score_array()
    assert np.allclose(branched_scores[: high_idx + 1], ancestor_scores[: high_idx + 1], rtol=0.0, atol=1e-12)
    assert abs(branched_scores[high_idx + 1] - ancestor_scores[high_idx + 1]) > 1e-9


def test_store_and_restore_branched_traj_before_replay(tmp_path: Path):
    """Test restoring a branched trajectory which has not replayed its ancestor history yet."""
    cfg = Config({"trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10}})
    t_ancestor, level, high_idx = _sparse_ancestor_with_tail_crossing(cfg)
    t_branched = Trajectory(
        100, 0.5, DoubleWellModel, cfg.load(TrajectoryConfig), model_params={"noise_amplitude": 0.8}, deterministic=True
    )
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, level, 0.25)
    chkfile = tmp_path / "branched.xml"
    t_branched.store(chkfile)
    t_restored = Trajectory.restore_from_checkfile(
        chkfile,
        t_branched.get_metadata(),
        DoubleWellModel,
        cfg.load(TrajectoryConfig),
        model_params={"noise_amplitude": 0.8},
    )
    assert t_restored.get_length() == high_idx + 1

    # Both replay the ancestor history up to the crossing, and take the same (new) step from it
    t_branched.advance(nstep_end=high_idx + 1)
    t_restored.advance(nstep_end=high_idx + 1)
    assert np.allclose(t_restored.get_score_array(), t_branched.get_score_array(), rtol=0.0, atol=1e-12)


def test_replay_mismatch_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Test the warning issued when replaying the tail of a trajectory does not reproduce it."""
    cfg = Config({"trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10}})
    t_ancestor, level, _ = _sparse_ancestor_with_tail_crossing(cfg)
    t_branched = Trajectory(
        100, 0.5, DoubleWellModel, cfg.load(TrajectoryConfig), model_params={"noise_amplitude": 0.8}, deterministic=True
    )
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, level, 0.25)
    chkfile = tmp_path / "branched.xml"
    t_branched.store(chkfile)
    # Restore with a different noise amplitude: the replayed history differs
    t_restored = Trajectory.restore_from_checkfile(
        chkfile, t_branched.get_metadata(), DoubleWellModel, cfg.load(TrajectoryConfig), model_params={}
    )
    t_restored.advance(nstep_end=t_restored.current_step() + 1)
    assert "replay mismatch" in caplog.text


class NegativeScoreModel(SimpleFModel):
    """Simple forward model with a negative score."""

    def score(self) -> float:
        """Override the template."""
        return super().score() - 2.0


def test_branch_negative_score_max():
    """Test the score maximum of a branched trajectory with negative scores."""
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.5}})
    t_ancestor = Trajectory(1, 0.5, NegativeScoreModel, cfg.load(TrajectoryConfig))
    t_ancestor.advance()
    assert t_ancestor.score_max() == pytest.approx(-1.6)
    t_branched = Trajectory(2, 0.5, NegativeScoreModel, cfg.load(TrajectoryConfig))
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, -1.8, 0.25)
    assert t_branched.score_max() == pytest.approx(np.max(t_branched.get_score_array()))
    assert t_branched.score_max() < 0.0
    t_branched.advance()
    assert t_branched.score_max() == pytest.approx(-1.6)


def test_simple_model_traj_end_nstep():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"step_size": 0.001, "targetscore": 0.55}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(nstep_end=20)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert t_test.is_terminated() is True
    t_test.advance(t_end=0.03)
    assert isclose(t_test.score_max(), 0.3, abs_tol=1e-9)
    assert t_test.is_converged() is False


def test_store_and_restore_simple_traj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig))
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink(missing_ok=True)


def test_store_and_restore_frozen_simple_traj():
    """Test store and restoring frozen trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig), frozen=True)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    with pytest.raises(RuntimeError):
        rst_test.advance()
    with pytest.raises(RuntimeError):
        rst_test._one_step()
    chkfile.unlink(missing_ok=True)


def test_restart_simple_traj():
    """Test trajectory restart."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    from_traj = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    from_traj.advance(t_end=0.01)
    rst_traj = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.05, 0.25)
    assert rst_test.current_time() == 0.006


def test_access_data_simple_traj():
    """Test trajectory data access."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.01)
    assert t_test.get_length() == 11
    assert isclose(t_test.get_time_array()[-1], 0.01, abs_tol=1e-9)
    assert isclose(t_test.get_score_array()[-1], 0.1, abs_tol=1e-9)


def test_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.012)
    assert isclose(t_test.score_max(), 0.12, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert isclose(t_test.get_last_state(), 0.01, abs_tol=1e-9)
    t_test.advance()
    assert t_test.is_converged() is True
    assert isclose(t_test.get_last_state(), 0.025, abs_tol=1e-9)


def test_sparse_simple_traj_access_states():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance()
    assert len(t_test.get_state_list()) == 26


def test_store_and_restart_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.013)
    assert isclose(t_test.score_max(), 0.13, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig))
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink()


def test_score_moving_average():
    """Test using a moving average on a score array."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.9, "step_size": 0.0001, "targetscore": 0.95}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance()
    score = t_test.get_score_array()
    avg_score = moving_avg(score, 10)
    assert isclose(avg_score[0], 0.0045, abs_tol=1e-9)


def test_sparse_dw_traj_with_restore():
    """Test restore a sparse trajectory with DW model."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
            "model": {"noise_amplitude": 0.8},
        }
    )
    model_params = cfg.section_dict("model")
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True)
    t_test.advance(t_end=4.07)
    chkfile = Path("./test.xml")
    t_test.store(chkfile, write_metadata_json=True)
    assert isclose(t_test.score_max(), 0.5384037112515893, abs_tol=1e-9)
    assert not t_test.is_converged()
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(
        chkfile, metadata, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, frozen=False
    )
    rst_test.advance()
    assert rst_test.score_max() > 0.95
    assert rst_test.is_converged()
    chkfile.unlink(missing_ok=True)
    Path("test.json").unlink(missing_ok=True)


def test_dw_traj_with_restore_and_diags():
    """Test restore a trajectory with DW model and diagnostics."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95},
            "model": {"noise_amplitude": 0.8},
            "testd": {"type": "FirstAndLastEveryCrossing", "score_min": 0.0, "score_max": 0.85, "n_levels": 11},
            "testd2": {"type": "EveryTimeCrossing", "time_start": 0.005, "time_interval": 0.02},
        }
    )
    model_params = cfg.section_dict("model")
    diagdict = {"testd": cfg.section("testd"), "testd2": cfg.section("testd2")}
    t_test = Trajectory(
        1, 1.0, fmodel, cfg.load(TrajectoryConfig), diag_configs=diagdict, model_params=model_params, deterministic=True
    )
    t_test.advance(t_end=4.07)
    chkfile = Path("./test.xml")
    t_test.store(chkfile, write_metadata_json=True)
    with Path("./test.json").open("r") as f:
        metadata_str = f.read()
    metadata = Trajectory.deserialize_metadata(metadata_str)
    rst_test = Trajectory.restore_from_checkfile(
        chkfile,
        metadata,
        fmodel,
        cfg.load(TrajectoryConfig),
        diag_configs=diagdict,
        model_params=model_params,
        frozen=False,
    )
    rst_test.advance()
    analyst = DiagnosticAnalyst("./diagDB.db")
    _ = analyst.get_traj_diagnostic_data("testd", 1)
    _ = analyst.get_traj_diagnostic_data("testd2", 1, time_ordered=True)
    analyst.db.dump_to_json("test_ddb.json")
    analyst = None
    chkfile.unlink(missing_ok=True)
    Path("test.json").unlink(missing_ok=True)
    Path("./diagDB.db").unlink(missing_ok=True)
    Path("test_ddb.json").unlink(missing_ok=True)


def test_sparse_dw_traj_with_branching():
    """Test branching a sparse trajectory with simple model."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
            "model": {"noise_amplitude": 0.3},
        }
    )
    model_params = cfg.section_dict("model")
    t_test = [
        Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True),
        Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True),
    ]
    t_test[0].advance()
    t_test[1].advance()
    if t_test[0].score_max() > t_test[1].score_max():
        rst_idx = 1
        from_idx = 0
        rst_val = t_test[1].score_max()
    else:
        rst_idx = 0
        from_idx = 1
        rst_val = t_test[0].score_max()
    branched_test = Trajectory.branch_from_trajectory(t_test[from_idx], t_test[rst_idx], rst_val, 0.25)
    branched_test.advance()
    assert branched_test.score_max() > t_test[rst_idx].score_max()
