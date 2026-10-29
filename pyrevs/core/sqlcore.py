"""A class for the core pyREVS data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
from typing import cast
from sqlalchemy import JSON
from sqlalchemy import CursorResult
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from .sqlmanager import BaseSQLManager

_logger = logging.getLogger(__name__)


class CoreBase(DeclarativeBase):
    """A base class for the tables."""


class Trajectory(CoreBase):
    """A table storing the active trajectories."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(default="idle", nullable=False)


class ArchivedTrajectory(CoreBase):
    """A table storing the archived trajectories."""

    __tablename__ = "archived_trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


valid_statuses = ["locked", "idle", "completed"]


class CoreDB(BaseSQLManager):
    """A database holding pyREVS trajectories.

    Allows atomic access to an SQL database from all
    the workers.

    Note: pyREVS works with Python indexing starting at 0,
    while SQL indexing starts at 1. Trajectory ID is
    updated accordingly when accessing/updating the DB.

    Attributes:
        _file_name : The file name
    """

    def __init__(self, file_name: str, in_memory: bool = False, ro_mode: bool = False) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
            in_memory: a bool to trigger in-memory creation
            ro_mode: a bool to trigger read-only access to the database
        """
        super().__init__(file_name, CoreBase.metadata, in_memory, ro_mode)

    def add_trajectory(self, traj_file: str, metadata: dict) -> None:
        """Add a new trajectory to the DB.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a dict with the metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            new_traj = Trajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)

    def update_trajectory(self, traj_id: int, traj_file: str, metadata: dict) -> None:
        """Update a given trajectory data in the DB.

        Args:
            traj_id : The trajectory id
            traj_file : The new trajectory file of that trajectory
            metadata: a dict with the trajectory metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj:
                traj.traj_file = traj_file
                traj.t_metadata = metadata
            else:
                err_msg = f"Trajectory {traj_id} not found !"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def update_trajectory_file(self, traj_id: int, traj_file: str) -> None:
        """Update the trajectory file of a given trajectory data in the DB.

        Args:
            traj_id : The trajectory id
            traj_file : The new trajectory file of that trajectory

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj:
                traj.traj_file = traj_file
            else:
                err_msg = f"Trajectory {traj_id} not found !"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def update_trajectory_weight(self, traj_id: int, weight: float) -> None:
        """Update a given trajectory weight in the DB.

        Args:
            traj_id : The trajectory id
            weight: the new trajectory weight

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj is None:
                err_msg = f"Trajectory {traj_id} not found !"
                _logger.error(err_msg)
                raise ValueError(err_msg)

            metadata_d = dict(traj.t_metadata)
            metadata_d["weight"] = weight
            traj.t_metadata = metadata_d

    def update_trajectory_weights(self, weights: dict[int, float]) -> None:
        """Update the weight of several trajectories in the DB.

        All the weights are updated in a single transaction.

        Args:
            weights: the new weight of each trajectory, keyed by trajectory id

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            for traj_id, weight in weights.items():
                traj = session.get(Trajectory, traj_id + 1)
                if traj is None:
                    err_msg = f"Trajectory {traj_id} not found !"
                    _logger.error(err_msg)
                    raise ValueError(err_msg)

                metadata_d = dict(traj.t_metadata)
                metadata_d["weight"] = weight
                traj.t_metadata = metadata_d

    def lock_trajectory(self, traj_id: int, allow_completed_lock: bool = False) -> bool:
        """Set the status of a trajectory to "locked" if possible.

        Args:
            traj_id : The trajectory id
            allow_completed_lock : Allow to lock a "completed" trajectory

        Return:
            True if the trajectory was successfully locked, False otherwise

        Raises:
            ValueError if the trajectory with the given id does not exist
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            stmt = select(Trajectory).filter(Trajectory.id == traj_id + 1).with_for_update()
            traj = session.execute(stmt).scalar_one_or_none()

            if traj:
                allowed_status = ["idle", "completed"] if allow_completed_lock else ["idle"]
                if traj.status in allowed_status:
                    traj.status = "locked"
                    return True
                return False

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def mark_trajectory_as_completed(self, traj_id: int) -> None:
        """Set the status of a trajectory to "completed" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.execute(select(Trajectory).filter(Trajectory.id == traj_id + 1)).scalar_one_or_none()
            if traj:
                if traj.status == "locked":
                    traj.status = "completed"
                else:
                    warn_msg = f"Attempting to mark completed Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def release_trajectory(self, traj_id: int) -> None:
        """Set the status of a trajectory to "idle" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            traj = session.execute(select(Trajectory).filter(Trajectory.id == traj_id + 1)).scalar_one_or_none()
            if traj:
                if traj.status == "locked":
                    traj.status = "idle"
                else:
                    warn_msg = f"Attempting to release Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def get_trajectory_count(self) -> int:
        """Get the number of trajectories in the DB.

        Returns:
            The number of trajectories
        """
        with self.session_scope() as session:
            return session.scalar(select(func.count(Trajectory.id))) or 0

    def get_terminated_trajectory_count(self) -> int:
        """Return the number of trajectories that have 'terminated' in their metadata."""
        with self.session_scope() as session:
            stmt = select(func.count(Trajectory.id)).where(Trajectory.t_metadata["terminated"].as_boolean())
            return session.scalar(stmt) or 0

    def get_converged_trajectory_count(self) -> int:
        """Return the number of trajectories that have 'converged' in their metadata."""
        with self.session_scope() as session:
            stmt = select(func.count(Trajectory.id)).where(Trajectory.t_metadata["converged"].as_boolean())
            return session.scalar(stmt) or 0

    def get_total_computed_steps(self) -> int:
        """Sum the 'nstep_compute' field across all active and archived trajectories."""
        with self.session_scope() as session:
            # Create a subquery for active trajectories
            active_steps = select(Trajectory.t_metadata["nstep_compute"].as_integer().label("steps"))

            # Create a subquery for archived trajectories
            archived_steps = select(ArchivedTrajectory.t_metadata["nstep_compute"].as_integer().label("steps"))

            # Combine them using union_all
            combined = active_steps.union_all(archived_steps).subquery()

            # Select the sum of the combined column
            total_sum = session.scalar(select(func.sum(combined.c.steps)))

            return int(total_sum) if total_sum else 0

    def fetch_trajectory(self, traj_id: int) -> tuple[str, dict]:
        """Get the trajectory file of a trajectory.

        Args:
            traj_id : The trajectory id

        Return:
            A tuple with trajectory file as a str and the trajectory metadata as dict

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj:
                return traj.traj_file, traj.t_metadata

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def check_trajectory_exist(self, traj_id: int) -> bool:
        """Check if a trajectory exist for a given index.

        Args:
            traj_id : The trajectory id

        Return:
            True if the trajectory exist, False otherwise
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            return bool(traj)

    def release_all_trajectories(self) -> None:
        """Release all trajectories in the DB."""
        with self.session_scope() as session:
            session.execute(update(Trajectory).values(status="idle"))

    def archive_trajectory(self, traj_file: str, metadata: dict) -> None:
        """Add a new trajectory to the archive container.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a dict with the traj metadata
        """
        with self.session_scope() as session:
            new_traj = ArchivedTrajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)

    def fetch_archived_trajectory(self, traj_id: int) -> tuple[str, dict]:
        """Get the trajectory file of a trajectory in the archive.

        Args:
            traj_id : The trajectory id

        Return:
            A tuple with trajectory file as a str and the trajectory metadata as dict

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            db_id = traj_id + 1
            traj = session.get(ArchivedTrajectory, db_id)
            if traj:
                return traj.traj_file, traj.t_metadata

            err_msg = f"Archived Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def get_archived_trajectory_count(self) -> int:
        """Get the number of trajectories in the archive.

        Returns:
            The number of trajectories
        """
        with self.session_scope() as session:
            return session.scalar(select(func.count(ArchivedTrajectory.id))) or 0

    def discard_archived_trajectory(self, traj_id: int) -> None:
        """Remove a trajectory from the archive.

        Args:
            traj_id : The trajectory id
        """
        with self.session_scope() as session:
            db_id = traj_id + 1
            traj = session.get(ArchivedTrajectory, db_id)
            if traj:
                session.delete(traj)

    def clear_archived_trajectories(self) -> int:
        """Delete the content of the archived traj table.

        Returns:
            The number of entries deleted
        """
        with self.session_scope() as session:
            stmt = delete(ArchivedTrajectory)
            result = session.execute(stmt)
            return int(cast("CursorResult", result).rowcount or 0)

    def dump_file_json(self, json_file: str | None = None) -> None:
        """Dump the content of the trajectory table to a json file.

        Args:
            json_file: an optional file name (or path) to dump the data to
        """
        db_data: dict[str, Any] = {}
        with self.session_scope() as session:
            db_data["trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "status": traj.status, "metadata": traj.t_metadata}
                for traj in session.execute(select(Trajectory)).scalars().all()
            }
            db_data["archived_trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "metadata": traj.t_metadata}
                for traj in session.execute(select(ArchivedTrajectory)).scalars().all()
            }

        json_path = Path(json_file) if json_file else Path(f"{Path(self._file_name).stem}.json")
        with json_path.open("w") as f:
            json.dump(db_data, f, indent=2)
