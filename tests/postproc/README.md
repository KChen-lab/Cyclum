Click for details...

- [Comparison of two hESC samples `hESC-comparison.Rmd`/`hESC-comparison.nb.html`](https://kchen-lab.github.io/Cyclum/tests/postproc/hESC-comparison.nb.html)
  - Compare circular mangitude of genes between nicotine treated and normal cells
  - Yields marker genes including _ENO1_, _LDHA_, etc.
- [Prepare hESC data for GSEA `hESC-gsea.Rmd`/`hESC-gsea.nb.html`](https://kchen-lab.github.io/Cyclum/tests/postproc/hESC-gsea.nb.html)
  - Output expression matrix in a way GSEA recognizes.
  - Breaks the circular pseudotime at some point to acquire a linear pseudotime.
  - Filter marker genes list with regard to available genes. Otherwise GSEA will run into error.
- [EMT and MET markers `emt.Rmd`/`emt.nb.html`](https://kchen-lab.github.io/Cyclum/tests/postproc/emt.nb.html)
  - Find distinct MET markers, such as _Cited1_, _Fzd7_.
