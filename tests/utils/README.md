R utilities
- [gsea.R](https://github.com/KChen-lab/Cyclum/blob/master/tests/utils/gsea.R): write gene expression matrix and pseudotime to `.txt` and `.cls` for GSEA to analyse.
- [hdfrw.R](https://github.com/KChen-lab/Cyclum/blob/master/tests/utils/hdfrw.R): Exchange data with [cyclum.hdfrw](https://github.com/KChen-lab/Cyclum/blob/master/cyclum/hdfrw.py) using hdf.
- [mclust_plot.R](https://github.com/KChen-lab/Cyclum/blob/master/tests/utils/mclust_plot.R): Plot mclust results.
- [reader.R](https://github.com/KChen-lab/Cyclum/blob/master/tests/utils/reader.R): Legacy code to read our `.bin` files (and exchange with [cyclum.writer](https://github.com/KChen-lab/Cyclum/blob/master/cyclum/writer.py)). It's basically data dumped as byte. It's more precise and efficient than `.txt` or `.csv`, but still not as good as hdf.
