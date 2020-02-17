Output Formats
==============

Several output schemes are available.

Single-cell pseudo-time
-----------------------

This is the inferred pseudo-time for the cells.

Rotation matrix
---------------------

This the matrix for transforming the pseudo-times to an "ideal cycling cell".


Corrected expression matrix
---------------------------

This is provided for convenience.
Because the corrected expression matrix is calculated by the original input matrix, single-cell pseudo-time and rotation matrix, it would be more space-efficient if not storing it.

To store the corrected expression matrix, there are two ways -- csv and binary file.
Although csv file is human readable, importing csv in R is significantly slow, due to overhead of parsing the file.
Binary file is much smaller and faster, but not human readable.

Csv file
^^^^^^^^
Storing matrix or data frame into csv file can be done using the numpy and pandas packages directly.

Binary file
^^^^^^^^^^^

We provide :func:`cyclum.writer.write_matrix_to_binary` and :func:`cyclum.writer.write_df_to_binary` function to store matrix and data frames. For more details about the binary format, please refer to the module :mod:`cyclum.writer`.