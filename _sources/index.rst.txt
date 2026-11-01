.. pyREVS documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyREVS's documentation!
==========================================================

`pyREVS` is a modular implementation of rare-event sampling algorithms in stochastic
dynamical systems. Initially focused on Trajectory-adaptive Multilevel Splitting (TAMS), 
`pyREVS` is now amendable to other algorithms of the Importance Splitting (IS) family.
`pyREVS` was developed to specifically handle (computationally) expensive stochastic models,
where integrating the model can take hours to days on supercomputers and using a naive Monte-Carlo
approach is impractical.

.. warning::
   PR#173 introduced significant changes to the source code and API, as well
   as the scope of the package. The documentation has been mostly updated,
   but might still refers solely to TAMS or pyTAMS in some locations. Please
   report any issues to the `GitHub repository <https://github.com/nlesc-eTAOC/pyREVS>`_



Installation:
-------------

To install `pyREVS`, simply use ``pip`` in your favorite environment manager
to get the latest stable version:

.. code-block:: shell

   pip install pyrevs

or if you plan on modifying the code or test the shipped-in examples, install from sources:

.. code-block:: shell

    git clone git@github.com:nlesc-eTAOC/pyREVS.git
    cd pyREVS
    pip install -e .

Quick start:
------------

Only if you have used the second option above you can readily test `pyREVS` on
a simple problem:

.. code-block:: shell

    cd pyREVS/examples/DoubleWell2D
    python sample_dw2dim.py

otherwise, please read through this documentation and in particular follow the
:ref:`tutorials Section <sec:tutorials>` to see how to implement your own model within `pyREVS`.

.. toctree::
   :maxdepth: 2
   :caption: User guide:

   Theory.rst
   Validation.rst
   Usage.rst
   Tutorials.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer guide:

   Implementation.rst 
   Extension.rst
   autoapi/pyrevs/index.rst

Documentation
-------------

The documentation pages are distributed with the code in the ``docs``
folder as "reStructuredText" files. The HTML is built automatically
whenever changes are pushed to the main branch on GitHub.
A local version can also be built as follows:

.. code-block:: shell

    cd <pyREVS_root_folder>/docs
    make html


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
