Output Formats
==============

Several output schemes are available.

Single-cell pseudo-time
-----------------------

This is the embedding of the cells

Transformation matrix
---------------------

This the matrix for transforming.

.. math::

   (a + b)^2 = a^2 + 2ab + b^2

   (a - b)^2 = a^2 - 2ab + b^2

Expression matrix
-----------------

It is an option to output the expression matrix in binary format. Although csv file is human readable, importing csv in R is significantly slow, due to overhead of parsing the file. Binary file is much smaller and faster, but not human readable.
