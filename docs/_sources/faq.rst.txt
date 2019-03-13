FAQ
=================

What preprocess should be done on the scRNA-seq data?
-----------------------------------------------------

Cyclum is noncommittal on what specific transform is used, as long as the circular pattern is preserved. That said, experiments show that scaled (variance = 1), centered (mean = 1) log-transformed (:math:`\log_2(x+1)`) data work well. Using Freeman-Tucky transform (:math:`\sqrt{x} + \sqrt{x+1}`) may also be a good idea.

How do I choose number of linear dimensions?
--------------------------------------------

We find 0 or 1 generally works well when cell cycle is really a problem in the data.
