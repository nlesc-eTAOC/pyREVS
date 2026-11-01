.. highlight:: rst

.. _sec:implementation:

Implementation
==============

`pyREVS` implements rare-events sampling algorithms while encapsulating all the model-specific
functionalities into an Abstract Base Class (ABC). By decoupling the physics from the
sampling algorithm, it becomes easier to extend the algorithm to new physics.

In particular, `pyREVS` aims at tackling computationally expensive stochastic models, such as
high-dimensional dynamical systems appearing in climate modeling or fluid dynamics, which requires
High Performance Computing (HPC) platform to be used. As such, `pyREVS` can be less efficient
than more simplistic implementations where pure Python physics model can be efficiently vectorized.
The internals of `pyREVS` rely on a hierarchy of classes to describe data structures, data storage,
workers and eventually the algorithm.

The reader is referred to the API documentation for more details on the classes and functions introduced
hereafter.

Data structures & storage
-------------------------

`pyREVS` uses an Array-Of-Structs (AOS) data structure to represent trajectories.
The low-level data container is a ``Snapshot``, a dataclass gathering the instantaneous state
of the model at a given point in time, along with a time, a noise increment and a value of the score function.
Note that only the time and score are typed (both as ``float``), while the type of the state and noise
are up to the model implementation.

A list of snapshots constitutes a ``trajectory``, along with some metadata such as the start and
end times, the step size or the maximum score. The ``trajectory`` object instantiates the model, and
implements function to advance the model in time or branch a trajectory.

Finally, a list of trajectories is the central container for the `pyREVS`'s ``database``. The algorithm
writes, reads and accesses trajectories through the database.
Algorithm-specific data are stored in an extension of the database: TAMS algorithm's data
such as splitting iterations weights and biases. The ``database`` can be instantiated independently
from a sampling run in order to explore the database contents.

The ``database`` has both an in-memory component (Python's list of trajectories) and, if requested, a
persistence layer for long-running sampling runs and storage. The persistence layer is implemented as
SQL databases, managed with SQLAlchemy.

Workers & parallelism
---------------------

The sampling algorithms can expose parallelism in several places: a plain Monte-Carlo sampling run
is embarassingly parallel since all the trajectories are fully independent, in contrast the TAMS algorithm
iterative process can only parallelize up to the number of trajectories discarded at each iteration.
So the exact number of workers will depends on the sampling strategy used.

Distribution of work is handled by a ``taskrunner`` object, which can have either a ``dask`` or
an ``asyncio`` backend. The runner will spawn several workers, picking up tasks submitted to the
runner. When using the ``dask`` runner with Slurm, the workers are spawned in individual Slurm
jobs.

Algorithm
---------

The user-facing API of `pyREVS` is implemented in a ``Sampler`` object, which orchstrate the initialization
of the ``database`` and the instantiation of a ``SamplingStrategy`` object, effectively implementing the
rare-event sampling algorithm. The code base is organized to make the implementation of new sampling
algorithms easier: ``SamplingStrategy`` are wrapped as self-contained plugins, relying on `pyREVS` core
data structures and storage.
