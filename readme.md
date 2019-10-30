# Cyclum

Cyclum is a package to tackle cell cycle. It provides methods to recover cell cycle information and remove cell cycle factor from the scRNA-seq data. The methodology is to rely on the circular manifold, instead of the marker genes. Multiple methods suits this idea. We provide an Auto-Encoder based realization at this time, and we are adding Gaussian Process Latent Variable Model soon. Also provided are a set of supplementary tools to visualize and anaylzing the result, in python and in R.

For further information, please also refer to the [documentation](https://kchen-lab.github.io/cyclum/) and the [manuscript](https://www.biorxiv.org/content/10.1101/625566v1).

We have developed a more user friendly [version](https://github.com/lshh125/cyclum2), including more readable keras implementation, and various helper functions for transferring data among python, R and GSEA. It also includes a tool to decide how many linear component is needed.
