FAQ
=================

What preprocess should be done on the scRNA-seq data?
-----------------------------------------------------

Cyclum is noncommittal on what specific transform is used, as long as the circular pattern is preserved. That said, experiments show that scaled (variance = 1), centered (mean = 1) log-transformed (:math:`\log_2(x+1)`) data work well. Using Freeman-Tukey transform (:math:`\sqrt{x} + \sqrt{x+1}`) may also be a good idea.

How do I choose number of linear dimensions?
--------------------------------------------

We provide :code:`cyclum.tuning.CyclumAutoTune` to help decide the number of linear dimensions.

What does the pseudo-time mean?
--------------------------------------------

It is the inferred stage of a cell. One pseudo-time for one cell itself does not make sense because it is not a real time. However, pseudo-times for more cells reveals their order and similarities. The pseudo-time for many cells shows the trajectory of cells in a scale.

What does the circular pseudo-time mean?
--------------------------------------------

It means we assume that there is no start or end in the time. Cells are going through a circular process, presumably cell cycle. That said, prior knowledge can be used to determine which pseudo-time is G1 phase and M phase, and they can be considered the start and end.