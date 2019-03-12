.. scAE-batch documentation master file, created by
   sphinx-quickstart on Mon Mar 11 17:44:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cyclum's documentation!
==================================

Cyclum is a package to tackle cell cycle.
It provides methods to recover cell cycle information and remove cell cycle factor from the scRNA-seq data.
The methodology is to rely on the circular manifold, instead of the marker genes.
Multiple methods suits this idea.
We provide an Auto-Encoder based realization at this time, and we are adding Gaussian Process Latent Variable Model soon.
Also provided are a set of supplementary tools to visualize and anaylzing the result, in python and in R.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick_start
   examples
   input_formats
   output_formats
   cyclum

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
