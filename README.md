# cyclum2

## What's new
We revised almost everything, except for the concept of using sinusoidal function in an autoencoder to find circular biological processes *ab initio*.

The core of Cyclum...
- is now rewritten using [keras](https://keras.io/), in a more readable way. We hope this will help researchers who want to experiment similar network structures;
- is enriched by a new class `cyclum.tuning.CyclumAutoTune`, which automatically select the proper number of linear components to help locate the "most circular" manifold.

Although Python is a good data analysis tool in addition to a general programing language, researchers may want to use R, which is more focused on statistics. Cyclum is implemented in python, but in order to help use both languages, we implemented...
- `mat2hdf` and `hdf2mat` in both Python and R, to help transferring data back and forth rapidly. (Prerequisites: `hdf5r` in R, `h5py` in python.)

[GSEA](http://software.broadinstitute.org/gsea/index.jsp) is a powerful tool to perform downstream gene enrichment analysis. We implemented in R...
- `mat2txt`, which writes a expression matrix to a GSEA compatible `.txt` file (Prerequisite: `data.table`, for much faster writing than `write.table`),
- `vec2cls`, which writes phenotypes (either discrete, e.g., cell type, or continuous, e.g., pseudotime) to a GSEA compatible `.cls` file,
- `mat2cls`, which writes multiple sets of phenotypes (continuous only, e.g., multiple PCs) to a GSEA compatible `.cls` file.

[Documentation](https://lshh125.github.io/cyclum2/index.html) is available at https://lshh125.github.io/cyclum2/index.html.
