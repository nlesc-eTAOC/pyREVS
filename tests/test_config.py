import pytest
from pyrevs.core import Config
from pyrevs.sampler.system_config import SystemConfig


def test_instantiate_sysconfig():
    """Test instantiation of SystemConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "runtime": {"walltime": 20.0},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
        "trajectory": {"step_size": 0.01},
    }
    config = Config(params_dict)
    _ = config.load(SystemConfig)


def test_amscfg_nosplit():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_nontraj():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"nsplititer": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_tams():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_ams():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20, "variant": "ams"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_wrongvariant():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20, "variant": "unknown"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_mccfg_nontraj():
    """Test validate the MCConfig class."""
    params_dict = {
        "sampler": {"strategy": "montecarlo"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


@pytest.mark.parametrize(("strategy", "default"), [("ams", 0.95), ("montecarlo", 0.95), ("gktl", None)])
def test_strategy_target_score(strategy: str, default: float | None):
    """Test the target score of each sampling strategy."""
    syscfg = Config({"sampler": {"strategy": strategy}}).load(SystemConfig)
    assert syscfg.strategy.target_score == default
    syscfg = Config({"sampler": {"strategy": strategy}, strategy: {"target_score": 0.5}}).load(SystemConfig)
    assert syscfg.strategy.target_score == 0.5


def test_amscfg_target_below_min_score():
    """Test validate the AMSConfig class with a target score below the minimum score."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20, "variant": "ams", "min_score": 0.5, "target_score": 0.4},
    }
    syscfg = Config(params_dict).load(SystemConfig)
    with pytest.raises(ValueError, match="target_score"):
        syscfg.strategy.validate()
