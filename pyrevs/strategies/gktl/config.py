from dataclasses import dataclass
from dataclasses import field
from math import isfinite
from pyrevs.core import MergePolicy

GKTL_OBSERVABLES = ("integral", "increment")


@dataclass(frozen=True)
class GKTLConfig:
    """GKTL strategy configuration."""

    __section__ = "gktl"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = field(
        default=-1,
        metadata={
            "doc": "Number of trajectories to sample. Sensible value checked upon initialization.",
        },
    )
    end_time: float = field(
        default=-1.0,
        metadata={
            "doc": "The end time of the trajectories.",
        },
    )
    resampling_interval: float = field(
        default=-1.0,
        metadata={
            "doc": "The resampling interval.",
        },
    )
    k: float = field(
        default=0.01,
        metadata={
            "doc": "Amplitude of the statistical bias.",
        },
    )
    bias_observable: str = field(
        default="integral",
        metadata={
            "doc": "Observable used to compute the clone weights over a resampling interval: "
            "'integral' (time integral of the score) or 'increment' (change in score).",
        },
    )
    use_custom_termination: bool = field(
        default=False,
        metadata={
            "doc": "Trigger use of the fmodel check_termination during sampling",
        },
    )

    def validate(self) -> None:
        """Validate GKTL configuration."""
        if self.ntrajectories <= 0:
            err_msg = " GKTLConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)

        if self.end_time <= 0.0:
            err_msg = " GKTLConfig.end_time must be > 0"
            raise ValueError(err_msg)

        if self.resampling_interval <= 0.0 or self.resampling_interval > self.end_time:
            err_msg = " GKTLConfig.resampling_interval must be > 0 and <= GKTLConfig.end_time"
            raise ValueError(err_msg)

        if not isfinite(self.k):
            err_msg = " GKTLConfig.k must be finite"
            raise ValueError(err_msg)

        if self.bias_observable not in GKTL_OBSERVABLES:
            err_msg = f" GKTLConfig.bias_observable must be one of {GKTL_OBSERVABLES}"
            raise ValueError(err_msg)
