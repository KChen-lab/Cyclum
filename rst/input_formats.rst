Input Formats
=============

As a marker-gene free method, cyclum only requires an expression matrix.

Expression matrix
-----------------

Cyclum accept expression matrix as input. Cyclum includes tools to
 + covert count matrix to TPM/PKM value,
 + log transform / Freeman-Tukey transform matrix,
 + center and scale expression data .

For evaluation purpose, cyclum also accept cell labels.

Configuration file (to be supported)
------------------------------------

The parameters for runing cyclum is specified in the configuration file. Fields are listed in the following table.

+------------------------+------------+------------+
| Field                  | Type       | Annotation |
|                        |            |            |
+========================+============+============+
| name                   |            |            |
+------------------------+------------+------------+
| preproc                |            |            |
+------------------------+------------+------------+
| output                 |            |            |
+------------------------+------------+------------+
