"""Defines the interface and simple implementations of termination criteria."""

from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar
from pyrevs.trajectory import TrajectoryStateType

if TYPE_CHECKING:
    from pyrevs.core import ForwardModelBaseClass
    from pyrevs.trajectory import Trajectory

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class TerminationCriterion(Protocol):
    """Termination criterion interface."""

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> TrajectoryStateType:
        """Check if the trajectory should terminate.

        Args:
            model: the forward model
            trajectory: the trajectory

        Returns:
            One of the TrajectoryStateType
        """
        raise NotImplementedError


class TimeTerminationCriterion(TerminationCriterion):
    """Termination criterion based on time.

    Will trigger termination if the current time is greater than or equal
    to the end time.
    """

    def __init__(self, end_time: float) -> None:
        """Initialize the termination criterion.

        Args:
            end_time: the end time
        """
        self._end_time = end_time

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> TrajectoryStateType:
        """Check if the trajectory should terminate."""
        _ = model
        if (
            isclose(trajectory.current_time(), self._end_time, abs_tol=1e-9)
            or trajectory.current_time() >= self._end_time
        ):
            return TrajectoryStateType.TERMINATED
        return TrajectoryStateType.ONGOING


class TimeInterruptionCriterion(TerminationCriterion):
    """Interruption criterion based on time.

    Will trigger interruption if the current time is greater than or equal
    to the end time.
    """

    def __init__(self, end_time: float) -> None:
        """Initialize the termination criterion.

        Args:
            end_time: the end time
        """
        self._end_time = end_time

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> TrajectoryStateType:
        """Check if the trajectory should terminate."""
        _ = model
        if (
            isclose(trajectory.current_time(), self._end_time, abs_tol=1e-9)
            or trajectory.current_time() >= self._end_time
        ):
            return TrajectoryStateType.INTERRUPTED
        return TrajectoryStateType.ONGOING


class LowScoreTerminationCriterion(TerminationCriterion):
    """Termination criterion based on score.

    Will trigger termination if the current score is less than or equal
    to a threshold.

    """

    def __init__(self, score_threshold: float) -> None:
        """Initialize the termination criterion.

        Args:
            score_threshold: the score threshold
        """
        self._score_threshold = score_threshold

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> TrajectoryStateType:
        """Check if the trajectory should terminate."""
        _ = trajectory
        if model.score() <= self._score_threshold:
            return TrajectoryStateType.TERMINATED
        return TrajectoryStateType.ONGOING


class ModelTerminationCriterion(TerminationCriterion):
    """Termination criterion based on model.

    Will trigger termination if the forward model has decides to.
    """

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> TrajectoryStateType:
        """Check if the trajectory should terminate."""
        if model.check_termination(trajectory.current_step(), trajectory.current_time()):
            return TrajectoryStateType.TERMINATED
        return TrajectoryStateType.ONGOING
