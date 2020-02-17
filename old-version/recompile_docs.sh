#!/usr/bin/env bash
source activate tensorflow-gpu
cp *.ipynb rst/
rm docs/* -r
sphinx-build -b html rst docs
rm rst/*.ipynb
git add docs/