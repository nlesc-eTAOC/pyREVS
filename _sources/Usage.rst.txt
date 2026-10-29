.. highlight:: rst

.. _sec:usage:

Usage
=====

Running pyREVS
--------------

Using `pyREVS` requires formulating your physics problem within the API defined by the
`pyREVS` Abstract Base Class (ABC) interface.

Once your new problem class is defined, rare-event sampling with `pyREVS` only requires a few
lines of code in say `mysamplingrun.py`:

.. code-block:: python

  from pyrevs.sampler import build_sampler
  from myproblem import my_problem_class
  
  if __name__ == "__main__":
  
      # Define a Sampler object with your problem class
      sampler = build_sampler(fmodel_t = my_problem_class)
  
      # Run TAMS
      probability = sampler.database().get_event_probability()

and then run your code, providing a TOML input file which is described in the next section:

.. code-block:: shell

  python mysamplingrun.py -i input.toml

.. note::
   Most sampling algorithm should run multiple times in order to provide both an estimate of the probability
   and the associated standard error.

Controls
--------

Upon instantiation of a `Sampler` object, the code searches for a TOML file containing the
run parameters. The path to the TOML file can be provided using the `-i` option, and
the default name is `input.toml` in the current working directory.

To get an overview of the available options, run the following command:

.. code-block:: shell

  pyrevs_help

Most input parameters have a default value, but the validity of the input parameters is checked
at runtime.

The TOML input file contains dictionaries associated with the various part of the algorithm and
data structures of the code:

- Sampler and runtime parameters:

  .. code-block:: python
  
    [sampler]
    strategy = "ams"            # [REQ] Sampling strategy
    deterministic = false       # [OPT, DEF = false] Fix the various random seeds for reproducibility

    [runtime]
    loglevel = "WARNING"        # [OPT, DEF = "INFO"] Log level
    logfile = "logTAMS.txt"     # [OPT, DEF = None] A file to redirect the standard logging to
    walltime = 200.0            # [OPT, DEF = 86400] Total walltime in seconds
    plot_diagnostics = false    # [OPT, DEF = false] Trigger ensemble plotting of on-the-fly diagnostics
    diagnostics = ["testdiag"]  # [OPT, DEF = None] A list of on-the-fly diagnostics


  At minima, running `pyREVS` requires specifying a sampling strategy.
  Additionally, runtime parameters allows to cleanly stop after a prescribed wall clock time (defaulted
  to a full day) and control the diagnostics performed on-the-fly during sampling.

- Sampling strategy parameters:
  
  .. code-block:: python
  
    [ams]
    ntrajectories = 20          # [REQ] Number of ensemble members
    nsplititer = 200            # [REQ] Maximum number of splitting iterations
    variant = "tams"            # [OPT, DEF = "tams"] Sampling variant
    end_time = 10.0             # [OPT, DEF = -1] End time, REQ if variant = "tams"
    min_score = 0.01            # [OPT, DEF = None] Minimum score, REQ if variant = "ams"
    l_j = 2                     # [OPT, DEF = 1] Number of score function levels discarded at each iteration


    [montecarlo]
    ntrajectories = 20          # [REQ] Number of ensemble members
    end_time = 10.0             # [OPT, DEF = -1] End time

    [gktl]
    ntrajectories = 20          # [REQ] Number of clones
    end_time = 10.0             # [REQ] End time
    resampling_interval = 0.5   # [REQ] Time interval between resampling steps
    k = 1.0                     # [OPT, DEF = 0.01] Amplitude of the bias
    bias_observable = "integral" # [OPT, DEF = "integral"] "integral" (time integral of the score) or "increment"

  Depending on the strategy prescribed, one of the above blocks is required.
  Whem running AMS, one must specify the number of members in the ensemble :math:`N`
  (``ntrajectories`` in the snippet above) as well as the maximum number of (splitting) iterations :math:`J`
  (``nsplititer`` above). The ``variant`` enable to switch between TAMS and AMS, and a different termination
  must then be provided. By default, a single score function level is discarded at each iteration (``l_j`` above).
  When running a Monte Carlo run, only the number of ensemble members is required (``ntrajectories`` above).
  If no end time is provided when using Monte-Carlo, trajectory will continue until convergence, which might
  take a long time, so it is recommended to provide an end time if only to avoid infinite trajectories.
  When running GKTL, the ensemble of :math:`N` clones (``ntrajectories``) is advanced up to ``end_time`` and
  resampled every ``resampling_interval``: each clone is killed or cloned according to a weight
  :math:`e^{k \Delta A}`, where :math:`\Delta A` is either the time integral or the increment of the score
  over the interval (``bias_observable``). The event probability returned accounts for the bias, and the
  extension also provides ``get_exceedance_probability(threshold)`` for any score threshold. Clones reaching
  the target score before ``end_time`` are frozen (this behavior might change in the future).


- Trajectory parameters:

  .. code-block:: python
    
    [trajectory]
    step_size = 0.01            # [REQ] Step size
    targetscore = 0.95          # [OPT, DEF = 0.95] Target score
    sparse_freq = 1             # [OPT, DEF = 1] Frequency of states sampling
    sparse_start = 0            # [OPT, DEF = 0] Starting index of states sampling
    chkfile_dump_all = false    # [OPT, DEF = false] Update trajectory checkpoint file at each step

  The trajectory object holds the system states in a chronological order, from time :math:`t=0` to
  an optional prescribed end time ``t_end`` provided at runtime.
  The step size must also be prescribed (``step_size``), but note that it needs not be the time step size
  of your dynamical system but rather the relevant step size for the stochastic forcing applied on the system.
  The trajectory object also enables sub-sampling the system state, only storing the state every n steps (``sparse_freq = n``).
  Internally, the trajectory object will keep track of the noise increment to ensure consistency of
  the full history if needed (assuming your model is deterministic under a prescribed noise).

- Runner parameters:

  .. code-block:: python
  
    [runner]
    type = "asyncio"            # [REQ] Runner type
    nworkers = 2                # [OPT, DEF = 1] Number of workers

  The ``runner`` manages scheduling the worker tasks over the course of the algorithm. Currently, two
  runner types are supported: ``asyncio`` is a light runner based on `the asyncio library <https://docs.python.org/3/library/asyncio.html>`_
  more suited when running `pyREVS` locally (or within the scope of a Slurm job), and ``dask``
  leverage `Dask <https://www.dask.org/>`_ and is required when deploying a large `pyREVS` run on a
  cluster.
  The number of independent workers is set by the ``nworkers`` parameter, which defaults to 1. Note that this
  is a maximum numbers of workers, for instance when running AMS iterations with a single discarded level, the actual
  number of workers might be lower (i.e. equal to the ``l_j`` parameter).

- Database parameters:

  .. code-block:: python
  
    [database]
    path = "TamsDB.tdb"         # [OPT, DEF = None] The database path, in-memory database if not specified
    restart = false             # [OPT, DEF = false] If true, move the existing database before starting fresh
    archive_discarded = true    # [OPT, DEF = true] Archive trajectories discarded during splitting iterations

  Running `pyREVS` on models with more than a dozen dimensions can lead to memory limitation
  issues. It is thus advised to enable storing the data to disk by specifying a path to a
  database in the input file.
    
Additionally, when using a ``dask`` runner, one has to provide configuration parameters for the
Dask cluster:

.. code-block:: python

  [dask]
  backend = "slurm"             # [OPT, DEF = "local"] Dask backend
  worker_walltime = "48:00:00"  # [OPT, DEF = "04:00:00"] Slurm job walltime
  queue = "genoa"               # [OPT, DEF = "regular"] Slurm job queue to submit the workers to
  ntasks_per_job = 64           # [OPT, DEF = 1] Number of tasks per Slurm job
  job_prologue = []             # [OPT, DEF = []] List of commands to be executed before the dask worker start

If a ``diagnostics`` list is provided in the `runtime` block, a block must be provided for each
label provided. At the moment only score-based diagnostics are available: they trigger when
the score function crosses levels defined in the block:

.. code-block:: python
  
  [testdiag] 
  type = "FirstCrossing"
  score_min = 0.0 
  score_max = 1.0 
  n_levels = 21 

The data sampled by the diagnostic will be stored in an SQL database located either in the
run folder or within the `pyREVS` database if one is requested.

Finally, note that a full TOML file, i.e. including defaults, is written in the database if requested 
and can be inspected at any time.

To pass model-specific parameters to your `pyREVS` model, the sampler will parse the `[model]` dictionary
of the TOML file and pass it to the model initializer.

See the :ref:`tutorials Section <sec:tutorials>` for a more practical use of the above input parameters.

Accessing the database
----------------------

If requested (see above the `[database]` section), `pyREVS` will write to disk the data
generated while sampling the rare-event. In practice, most large models require to save the data
to disk due to memory limitations or if the model IOs is not controlled by the user.

.. note::
   It is advised to always set `path = "/some/valid/path"` in the ``[database]`` section of
   your input file unless testing some very small models.

It is then possible to access the data (algorithm data, trajectory data, ...) independently
from the sampling runs itself. To do so, in a separate Python script, one can:

.. code-block:: python

  from pathlib import Path
  from pyrevs.utils import setup_logger
  from pyrevs.database import load_database

  if __name__ == "__main__":
      # Ensure we catch loading errors
      setup_logger("INFO")

      # Initiate the Database object, only (light) loading algorithm data from disk
      tdb = load_database(Path("./TestDB.tdb"))

      # Load trajectory data
      tdb.load_data(load_archived_trajectories=True)

The optional argument to `load_data` (defaulting to false) enable loading the discarded
trajectories data (for sampling strategies that archive them).
Upon loading the data, a summary of the database state is logged to screen, e.g.:

.. code-block:: shell

    [INFO] 2025-09-09 11:41:08,481 - 200 trajectories loaded
    [INFO] 2025-09-09 11:41:12,018 -
            ####################################################
            # pyREVS v1.0.0                                    #
            # Date: 2025-09-09 09:30:13.998659+00:00           #
            # Model: DoubleWellModel3D                         #
            # Strategy: ams                                    #
            ####################################################
            # Requested # of traj:                         200 #
            # Requested # of splitting iter:               500 #
            # Number of 'Terminated' trajectories:         200 #
            # Number of 'Converged' trajectories:            7 #
            # Current total number of steps:            463247 #
            ####################################################

One can then access the data in the database using the `database API <./autoapi/pyrevs/database/index.html>`_.
