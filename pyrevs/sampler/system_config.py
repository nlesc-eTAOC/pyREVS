"""A configuration class to expose limited configuration."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
import toml
from pyrevs.core import Config
from pyrevs.core import RuntimeConfig
from pyrevs.core.config import collect_sections
from pyrevs.core.config import merge_config
from pyrevs.core.config import print_config_help
from pyrevs.database import DatabaseConfig
from pyrevs.runner import RunnerConfig
from pyrevs.strategies.ams import AMSConfig
from pyrevs.strategies.gktl import GKTLConfig
from pyrevs.strategies.montecarlo import MCConfig
from pyrevs.trajectory import TrajectoryConfig
from .config import SamplerConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SystemConfig:
    """Overarching system configuration.

    This is a helper metadata class for the system configuration
    used to IO full configuration (i.e. including defaults)
    and performing configuration merging.
    """

    sampler: SamplerConfig
    runtime: RuntimeConfig
    strategy: AMSConfig | MCConfig | GKTLConfig
    database: DatabaseConfig
    runner: RunnerConfig
    trajectory: TrajectoryConfig

    @classmethod
    def __load__(cls, cfg: Config) -> SystemConfig:
        sampler = cfg.load(SamplerConfig)

        strategy_cls: type[AMSConfig | MCConfig | GKTLConfig]

        if sampler.strategy == "ams":
            strategy_cls = AMSConfig
        elif sampler.strategy == "montecarlo":
            strategy_cls = MCConfig
        elif sampler.strategy == "gktl":
            strategy_cls = GKTLConfig
        else:
            err_msg = f"Unknown strategy '{sampler.strategy}'"
            raise ValueError(err_msg)

        return cls(
            sampler=sampler,
            runtime=cfg.load(RuntimeConfig),
            strategy=cfg.load(strategy_cls),
            database=cfg.load(DatabaseConfig),
            runner=cfg.load(RunnerConfig),
            trajectory=cfg.load(TrajectoryConfig),
        )

    def write_toml(self, path: Path, other_data: dict[str, Any]) -> None:
        """Write the system configuration to a TOML file."""
        data = collect_sections(
            self.sampler,
            self.runtime,
            self.strategy,
            self.database,
            self.runner,
            self.trajectory,
        )

        with path.open("w") as f:
            toml.dump(data | other_data, f)

    @classmethod
    def merge(
        cls,
        old: SystemConfig,
        new: SystemConfig,
    ) -> SystemConfig:
        """Merge two SystemConfig objects.

        Immutable sections must match exactly.
        Replaceable sections are overwritten by `new`.

        Args:
            old: Existing configuration (e.g. from database).
            new: Incoming configuration (e.g. from CLI/TOML).

        Returns:
            A merged SystemConfig instance.

        Raises:
            ValueError: If immutable sections differ.
        """
        return cls(
            sampler=merge_config(old.sampler, new.sampler),
            runtime=merge_config(old.runtime, new.runtime),
            strategy=merge_config(old.strategy, new.strategy),
            database=merge_config(old.database, new.database),
            runner=merge_config(old.runner, new.runner),
            trajectory=merge_config(old.trajectory, new.trajectory),
        )

    @classmethod
    def print_config_help(cls) -> None:
        """Print a help message for the system configuration."""
        print_config_help(SamplerConfig)
        print_config_help(RuntimeConfig)
        print_config_help(AMSConfig)
        print_config_help(MCConfig)
        print_config_help(GKTLConfig)
        print_config_help(DatabaseConfig)
        print_config_help(RunnerConfig)
        print_config_help(TrajectoryConfig)
