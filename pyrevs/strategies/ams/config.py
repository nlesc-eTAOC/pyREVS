from dataclasses import dataclass
from dataclasses import field
from pyrevs.core import MergePolicy


@dataclass(frozen=True)
class AMSConfig:
    """AMS strategy configuration."""

    __section__ = "ams"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = field(
        default=-1,
        metadata={
            "doc": "Number of trajectories to sample. Sensible value checked upon initialization.",
        },
    )
    nsplititer: int = field(
        default=-1,
        metadata={
            "doc": "Number of splitting iterations. Sensible value checked upon initialization.",
        },
    )
    variant: str = field(
        default="tams",
        metadata={
            "doc": "Variant of AMS to use (one of [tams, ams])",
        },
    )
    l_j: int = field(
        default=1,
        metadata={
            "doc": "Number of score function levels discarded at each splitting iteration",
        },
    )
    init_ensemble_only: bool = field(
        default=False,
        metadata={
            "doc": "Whether or not to stop after initializing the trajectory ensemble",
        },
    )
    end_time: float = field(
        default=-1.0,
        metadata={
            "doc": "The end time of the trajectory (TAMS)",
        },
    )
    min_score: float | None = field(
        default=None,
        metadata={
            "doc": "The minimum score of the trajectory (AMS)",
        },
    )
    target_score: float = field(
        default=0.95,
        metadata={
            "doc": "The target score defining the rare event: a trajectory reaching it has converged",
        },
    )
    use_custom_termination: bool = field(
        default=False,
        metadata={
            "doc": "Trigger use of the fmodel check_termination during sampling",
        },
    )

    def validate(self) -> None:
        """Validate AMS configuration."""
        if self.ntrajectories <= 0:
            err_msg = " AMSConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)

        if self.nsplititer <= 0:
            err_msg = " AMSConfig.nsplititer must be > 0"
            raise ValueError(err_msg)

        if self.variant not in ["tams", "ams", "hams"]:
            err_msg = " AMSConfig.variant must be one of ['tams', 'ams', 'hams']"
            raise ValueError(err_msg)

        if self.variant == "tams" and self.end_time <= 0.0:
            err_msg = " AMSConfig.end_time must be > 0 for TAMS"
            raise ValueError(err_msg)

        if self.variant == "ams" and self.min_score is None:
            err_msg = " AMSConfig.min_score must be set for AMS"
            raise ValueError(err_msg)

        if self.variant == "hams" and (self.min_score is None or self.end_time <= 0.0):
            err_msg = " AMSConfig.min_score and end_time must be set for HAMS"
            raise ValueError(err_msg)

        if self.min_score is not None and self.target_score <= self.min_score:
            err_msg = " AMSConfig.target_score must be > AMSConfig.min_score"
            raise ValueError(err_msg)
