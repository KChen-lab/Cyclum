Jupyter notebooks for running Cyclum.

- EMT Dataset: [E-MTAB-2805](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-2805/)
  - [EMT.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/EMT.ipynb): Calculate circular trajectory for the EMT dataset.
- McDavid Dataset: Already included in [data/McDavid](https://github.com/KChen-lab/Cyclum/tree/master/data/McDavid). Raw data is available as [Data Set S2](https://doi.org/10.1371/journal.pcbi.1003696.s009) in [McDavid et al. Modeling Bi-modality Improves Characterization of Cell Cycle on Gene Expression in Single Cells](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003696).
  - [McDavid-PC3.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/McDavid-PC3.ipynb): Calculate circular trajectory for the McDavid-PC3 dataset
  - [McDavid-MB.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/McDavid-MB.ipynb): Calculate circular trajectory for the McDavid-MB dataset
  - [McDavid-H9.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/McDavid-H9.ipynb): Calculate circular trajectory for the McDavid-H9 dataset
  - [McDavid-PC3-tune.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/McDavid-PC3-tune.ipynb): Manual parameter tuning on McDavid-PC3 dataset
  
- mESC Dataset: A standalone [tutorial](https://github.com/KChen-lab/Cyclum-Demo) is available.
  - [mESC.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC.ipynb): Cyclum on mESC dataset with auto tuning.
  - [mESC-0.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC-0.ipynb): Cyclum on mESC dataset with 0 linear dimensions (i.e., k=1). This is an example when the dimensionality is not optimal.
  - [mESC-1.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC-1.ipynb): Cyclum on mESC dataset with 1 linear dimensions (i.e., k=2). This is an example when the dimensionality is optimal.
  - [mESC-tune.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC-tune.ipynb): Showing the elbow plot and bar plot used for manual tuning (if so desired).
  - [mESC-marker-higher-weight.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC-marker-higher-weight.ipynb): Utilizing known markers by giving them more weights; it results in slightly better performance.
  - [mESC-marker-only.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/mESC-marker-only.ipynb): Utilizing known markers by removing all other genes; it results in **worse** performance.
  
- hESC Dataset: Preprocessing is available in [../preproc](https://github.com/KChen-lab/Cyclum/tree/master/tests/preproc)
  - [hESC-both-regular.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/hESC-both-regular.ipynb): Cyclum on the whole hESC dataset.
  - [hESC-treated-regular.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/hESC-treated-regular.ipynb): Cyclum on the nicotine treated cells.
  - [hESC-control-regular.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/hESC-control-regular.ipynb): Cyclum on the control sample.

- Virtual Tumor Dataset: It can be generated use the code provided in the [old version of Cyclum](https://github.com/KChen-lab/Cyclum/tree/master/old-version)
  
- Miscellaneous:
  - [load-cyclum-model.ipynb](https://github.com/KChen-lab/Cyclum/blob/master/tests/notebooks/load-cyclum-model.ipynb): Load saved Cyclum model
