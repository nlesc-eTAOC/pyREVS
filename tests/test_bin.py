"""Tests for the pyrevs.bin functions."""

from pathlib import Path
import pytest
import toml
from pyrevs.bin import alive
from pyrevs.bin import input_help
from pyrevs.bin import sampling_run
from pyrevs.bin import template_model


def test_alive(capsys: pytest.CaptureFixture[str]):
    """Test pyREVS check function."""
    alive()
    assert "rare-event finder tool" in capsys.readouterr().out


def test_help(capsys: pytest.CaptureFixture[str]):
    """Test pyREVS help function."""
    input_help()
    assert "pyREVS input file help" in capsys.readouterr().out


def test_template_model():
    """Test pyREVS new model init function."""
    template_model(a_args=[])
    assert Path("./MyNewClass.py").exists()
    Path("./MyNewClass.py").unlink(missing_ok=True)


def test_template_model_with_name():
    """Test pyREVS new model init function."""
    template_model(a_args=["-n", "MyCustomClass"])
    assert Path("./MyCustomClass.py").exists()
    Path("./MyCustomClass.py").unlink(missing_ok=True)


def test_sampling_run():
    """Test sampling run."""
    params_dict = {
        "sampler": {"strategy": "ams", "deterministic": True},
        "runtime": {"walltime": 20.0},
        "ams": {"target_score": 0.6, "ntrajectories": 20, "nsplititer": 20, "variant": "tams", "end_time": 6.0},
        "runner": {"type": "asyncio", "nworkers": 1},
        "trajectory": {"step_size": 0.01},
        "model": {"slow_factor": 0.00000001, "noise_amplitude": 0.6},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    sampling_run(a_args=["-m", "./tests/dwmodel.py", "-i", "input.toml"])
    Path("input.toml").unlink(missing_ok=True)


def test_sampling_run_fail_two_fmodel():
    """Test sampling run."""
    with pytest.raises(RuntimeError):
        sampling_run(a_args=["-m", "./tests/models.py", "-i", "input.toml"])


def test_sampling_run_fail_nofmodel():
    """Test sampling run."""
    with pytest.raises(RuntimeError):
        sampling_run(a_args=["-m", "./tests/test_xmlutils.py", "-i", "input.toml"])
