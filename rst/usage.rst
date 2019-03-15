Usage
====================

The fields are shown in the help.

.. code:: bash

   $ python run_cyclum.py --help
   usage: run_cyclum.py [-h] [--remove] [--binary] [--binary-only] [--linear q]
                        input [output]

   Recover and remove cell cycle.

   positional arguments:
     input          input file name
     output         output name mask; the prefix for all output names

   optional arguments:
     -h, --help     show this help message and exit
     --remove       also output the cell cycle removed expression matrix
     --binary       output using binary file; faster to read by python or R
     --binary-only  output using binary file only without csv
     --linear q     integer number of linear dimensions


+------------------------+--------------------------------------------------------------------+
| Arguments              | Annotation                                                         |
|                        |                                                                    |
+========================+====================================================================+
| input                  | The full name of the input file                                    |
+------------------------+--------------------------------------------------------------------+
| output                 | The prefix of outputs. Default to be the prefix of input           |
+------------------------+--------------------------------------------------------------------+
| \\-\\-remove           | Besides the cell timings and gene profiles, also output the cell   |
|                        | cycle removed expression matrix                                    |
+------------------------+--------------------------------------------------------------------+
| \\-\\-binary-only      | output using binary file only (without csv)                        |
+------------------------+--------------------------------------------------------------------+
| \\-\\-linear q         | q is an integer, the number of linear dimensions                   |
+------------------------+--------------------------------------------------------------------+
