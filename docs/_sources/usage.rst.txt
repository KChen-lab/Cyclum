Usage
====================

The fields are shown in the help.

.. code:: bash

   $ python run_cyclum.py --help
   usage: run_cyclum.py [-h] [--cell-is-column] [--no-transform] [--remove]
                        [--binary-output] [--binary-output-only] [--linear q]
                        input [output]

   Recover and remove cell cycle.

   positional arguments:
     input                 input file name
     output                output name mask; the prefix for all output names

   optional arguments:
     -h, --help            show this help message and exit
     --cell-is-column      indicating that each column represents a cells
     --no-transform        indicating that no further transform is needed
     --remove              also output the cell cycle removed expression matrix
     --binary-output       output using binary file; faster to read by python or
                        R
     --binary-output-only  output using binary file only without csv
     --linear q            integer number of linear dimensions



+------------------------+--------------------------------------------------------------------+
| Arguments              | Annotation                                                         |
|                        |                                                                    |
+========================+====================================================================+
| input                  | The full name of the input file (tab delimited with cell names and |
|                        | gene names. The names do not have to have a meaning.)              |
+------------------------+--------------------------------------------------------------------+
| output                 | The prefix of outputs. Default to be the prefix of input           |
+------------------------+--------------------------------------------------------------------+
| \\-\\-cell-is-column   | Tell the program that each column (i.e. not row) stands for a cell |
+------------------------+--------------------------------------------------------------------+
| \\-\\-no-transform     | Tell the program not to log transform the imput data               |
+------------------------+--------------------------------------------------------------------+
| \\-\\-remove           | Besides the cell timings and gene profiles, also output the cell   |
|                        | cycle removed expression matrix                                    |
+------------------------+--------------------------------------------------------------------+
| \\-\\-binary-only      | output using binary file only (without csv)                        |
+------------------------+--------------------------------------------------------------------+
| \\-\\-linear q         | q is an integer, the number of linear dimensions                   |
+------------------------+--------------------------------------------------------------------+

The output file names will be the output prefix attached by "-cell.csv", "-gene.csv" and "-corrected.csv".