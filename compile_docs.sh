source activate tensorflow-gpu
cp example_mESC.ipynb rst/
sphinx-build -b html rst docs
rm rst/example_mESC.ipynb
git add docs/