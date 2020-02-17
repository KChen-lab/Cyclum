Quick Start Guide
=================

Call from bash
--------------

The easiest way to run Cyclum as a stand alone software.
The expression of mESC in TPM is provided in the data directory.
Just make sure you have all the dependencies and :code:`cd` to the folder containing :code:`run_cyclum.py` and run

.. code:: bash

    $ python run_cyclum.py data/mESC/mesc-tpm.csv --cell-is-column --remove

You will find the outputs in the same directory as the input file :code:`data/mESC/`.
:code:`mesc-tpm-cell.csv` contains the pseudotime for all cells.
:code:`mesc-tpm-gene.csv` contains the "magnitude" and "phase" of the cell-cycle components in the genes.
:code:`mesc-tpm-corrected.csv` contains the expression matrix with cell-cycle component removed.

Call from python
----------------
Nonetheless, Cyclum is also a python package.
We provide examples for recovering cell cycle and removing cell cycle by calling it in python.
Please click on the following links to access them.

.. toctree::
    :maxdepth: 1

    example_mESC
    example_mESC_simulated
    example_mel78
    example_mESC_neo