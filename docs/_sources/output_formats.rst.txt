Output Formats
==============

Several output schemes are available.

Single-cell pseudo-time
-----------------------

This is the embedding of the cells

Rotation matrix
---------------------

This the matrix for transforming the pseudo-times to an "ideal cycling cell". The ideal cell


Corrected expression matrix
---------------------------

To store the corrected expression matrix, there are two ways -- csv and binary file.
Although csv file is human readable, importing csv in R is significantly slow, due to overhead of parsing the file.
Binary file is much smaller and faster, but not human readable.

Csv file
^^^^^^^^

Binary file
^^^^^^^^^^^

For more details, please refer to the module :mod:`cyclum.writer`.