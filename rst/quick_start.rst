Quick Start Guide
=================

Dependencies
------------

Python part
^^^^^^^^^^^

The package requires `numpy <http://www.numpy.org//>`_, `scipy <http://www.scipy.org>`_, `pandas <https://pandas.pydata.org/>`_, matplotlib, scikit-learn and `TensorFlow <https://www.tensorflow.org/>`_.

The package versions this package has been tested on is shown in the following table.

+------------------------+------------+------------------------------------------------+
| Package                | Tested on  | Annotation                                     |
|                        | version    |                                                |
+========================+============+================================================+
| numpy                  | 1.15.2     | Provides basic matrix operations.              |
+------------------------+------------+------------------------------------------------+
| scipy                  | 1.1.0      | Provides basic statistic functions.            |
+------------------------+------------+------------------------------------------------+
| pandas                 | 0.23.4     | Provides data frame objects.                   |
+------------------------+------------+------------------------------------------------+
| matplotlib             |            | Plots figures.                                 |
+------------------------+------------+------------------------------------------------+
| sklearn                | 0.23.4     | Provides basic data transformations (e.g. PCA) |
+------------------------+------------+------------------------------------------------+
| TensorFlow             | 1.10.1     | Provides rich optimization tools               |
+------------------------+------------+------------------------------------------------+

For numpy, scipy, pandas and matplotlib, there are rarely big changes. Thus, unless your run into a problem, you do not have to upgrade/downgrade your packages. All the packages are provided in Anaconda.

Although TensorFlow is provided in Anaconda, but the TensorFlow official site recommends the pip version. You can follow the `official tutorial on how to install tensorflow with pip <https://www.tensorflow.org/install/pip>`_. TensorFlow 2.x has just been released, but please stay with 1.x. Also be advised that Tensorflow may not be compatible with python 3.7, so please use python 3.6.x.

To prevent tweaking package versions causing side-effects, a virtual environment manager like Conda is recommended.
The way to installing TensorFlow within a Conda Virtual Environment is provided in the official tutorial.

R part
^^^^^^

Depending on which analysis is preferred in R, packages Mclust, ComplexHeatmap may be required.

Download
--------

The current release can be downloaded from GitHub:

.. code:: bash

    git clone https://github.com/lshh125/cyclum.git

Installation
------------

It is now a small size local package so installation is not required.
Installation will be added in future releases to allow calling from anywhere.

