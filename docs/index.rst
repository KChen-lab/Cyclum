.. cyclum-doc documentation master file, created by
   sphinx-quickstart on Wed Oct 30 18:49:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cyclum-doc's documentation!
======================================

Cyclum is a package that infer latent periodic process (e.g., cell cycle) from single-cell RNA-seq data.
It provides methods to recover cell cycle information and remove cell cycle effects from the scRNA-seq data.
The methodology projects high dimensional gene expression data to a circular manifold, instead of using marker genes.

We provide an Auto-Encoder based realization at this time, and we will add Gaussian Process Latent Variable Model soon.
Also provided are a set of supplementary tools to visualize and anaylzing the result, in python and in R.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   modules
   faq
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
