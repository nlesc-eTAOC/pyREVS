"""Tests for the pyrevs.database class."""

import shutil
from pathlib import Path
import pytest
import toml
from pyrevs.core import Config
from pyrevs.database import Database
from pyrevs.database import load_database
from pyrevs.sampler import build_sampler
from pyrevs.utils.dbutils import prepare_database_path
from pyrevs.utils.utils import is_windows_os
from tests.dwmodel import DoubleWellModel


def test_wrong_format():
    """Test init of TDB with unsupported format."""
    fmodel = DoubleWellModel
    cfg = Config({"database": {"path": "dwTest.tdb", "format": "WRONG"}})
    with pytest.raises(ValueError):
        _ = Database.create(fmodel, cfg)


def test_load_missing_tdb():
    """Test failed load database."""
    with pytest.raises(FileNotFoundError):
        _ = Database.load(Path("dwTestNonExistent.tdb"))


def test_init_empty_tdb_inmemory():
    """Test init database."""
    fmodel = DoubleWellModel
    cfg = Config({})
    tdb = Database.create(fmodel, cfg)
    assert tdb.name() == "pyREVS_DoubleWellModel"


def test_init_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    cfg = Config({"database": {"path": "dwTest.tdb"}})
    tdb = Database.create(fmodel, cfg)
    assert tdb.name() == "dwTest.tdb"
    # Necessary on Windows
    del tdb
    shutil.rmtree("dwTest.tdb")


def test_reinit_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel

    # First create
    cfg = Config({"database": {"path": "dwTestDouble.tdb"}})
    tdb = Database.create(fmodel, cfg)
    # Necessary on Windows
    del tdb

    # Archive old
    prepare_database_path(Path("dwTestDouble.tdb"), True)

    # Second create
    cfg = Config({"database": {"path": "dwTestDouble.tdb", "restart": True}})
    tdb = Database.create(fmodel, cfg)
    del tdb
    ndb = 0
    for folder in Path("./").iterdir():
        if "dwTestDouble" in str(folder):
            shutil.rmtree(folder)
            ndb += 1
    assert ndb == 2


def test_init_and_load_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel

    # Create: note that when not using the sampler.build_database helper
    # one need to manually create the input_params.toml
    cfg = Config({"database": {"path": "dwTest.tdb"}})
    tdb = Database.create(fmodel, cfg)
    with Path("dwTest.tdb/input_params.toml").open("w") as f:
        toml.dump({"database": {"path": "dwTest.tdb"}}, f)
    tdb_path = Path(tdb.path())
    assert tdb.name() == "dwTest.tdb"
    del tdb
    tdb = Database.load(tdb_path)
    del tdb
    shutil.rmtree("dwTest.tdb")


@pytest.mark.dependency(name="genDB")
def test_generate_and_load_tdb():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "INFO"},
                "ams": {
                    "target_score": 0.51,
                    "ntrajectories": 50,
                    "nsplititer": 200,
                    "variant": "tams",
                    "end_time": 10.0,
                    "l_j": 1,
                },
                "database": {"path": "dwTest.tdb"},
                "runner": {"type": "asyncio", "nworkers": 2},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"step_size": 0.01},
            },
            f,
        )
    sampler = build_sampler(fmodel_t=fmodel, a_args=[])
    sampler.run()

    tdb = Database.load(Path("dwTest.tdb"))
    tdb.info()
    assert tdb
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.dependency(depends=["genDB"])
def test_access_ensemble_length():
    """Test accessing database trajectory ensemble size."""
    tdb = Database.load(Path("dwTest.tdb"))
    assert tdb.is_empty() is False


@pytest.mark.dependency(depends=["genDB"])
def test_access_terminated_count():
    """Test accessing database trajectory metadata."""
    tdb = Database.load(Path("dwTest.tdb"))
    assert tdb.count_terminated_traj() == 50


@pytest.mark.dependency(depends=["genDB"])
def test_access_converged_count():
    """Test accessing database trajectory metadata."""
    tdb = Database.load(Path("dwTest.tdb"))
    assert tdb.count_converged_traj() == 50


@pytest.mark.dependency(depends=["genDB"])
def test_access_pool_content():
    """Test accessing database SQL pool content."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.get_pool_db().dump_file_json("test.json")
    assert Path("./test.json").exists()
    Path("./test.json").unlink()


@pytest.mark.dependency(depends=["genDB"])
def test_copy_and_access():
    """Test copying the database and accessing it."""
    shutil.copytree("dwTest.tdb", "dwTestCopy.tdb")
    tdb = Database.load(Path("dwTestCopy.tdb"))
    assert tdb.count_converged_traj() == 50
    del tdb
    shutil.rmtree("dwTestCopy.tdb")


@pytest.mark.dependency(depends=["genDB"])
def test_replace_traj_in_tdb():
    """Test replacing a trajectory in the database."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data()

    traj_zero = tdb.get_traj(0)
    tdb.overwrite_traj(1, traj_zero)
    assert tdb.get_traj(1).idstr()[:10] == "traj000000"


@pytest.mark.dependency(depends=["genDB"])
def test_unknown_traj_access_in_tdb():
    """Test accessing a trajectory out-of-range."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data()

    with pytest.raises(ValueError):
        _ = tdb.get_traj(10000)


@pytest.mark.dependency(depends=["genDB"])
def test_unknown_traj_overwrite_in_tdb():
    """Test overwriting a trajectory out-of-range."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data()

    traj_zero = tdb.get_traj(0)
    with pytest.raises(ValueError):
        _ = tdb.overwrite_traj(10000, traj_zero)


@pytest.mark.dependency(depends=["genDB"])
def test_access_trajdata_in_tdb():
    """Test accessing a trajectory in the database."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data()

    traj = tdb.get_traj(0)
    times = traj.get_time_array()
    scores = traj.get_score_array()
    noises = traj.get_noise_array()
    assert times.size > 0
    assert scores.size > 0
    assert noises.size > 0


@pytest.mark.dependency(depends=["genDB"])
def test_explore_tdb():
    """Test loading the TDB."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data()
    tdb.plot_score_functions("test.png")
    Path("./test.png").unlink(missing_ok=False)


@pytest.mark.dependency(depends=["genDB"])
def test_explore_minmax_tdb():
    """Test loading the TDB."""
    tdb = load_database(Path("dwTest.tdb"))
    tdb.extension().plot_min_max_span(fname="test_minmax.png")
    Path("./test_minmax.png").unlink(missing_ok=False)


@pytest.mark.dependency(depends=["genDB"])
def test_explore_active_at_k():
    """Test getting the initial active set."""
    tdb = load_database(Path("dwTest.tdb"))
    tdb.load_data(load_archived_trajectories=True)
    act_trajs = tdb.extension().get_trajectory_active_at_k(0)
    assert len(act_trajs) == 50
    assert act_trajs[42].idstr() == "traj000042_0000"
    del tdb
    # Do not try to delete on Windows.
    if not is_windows_os():
        shutil.rmtree("dwTest.tdb")
