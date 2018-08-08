<a href="https://github.com/sahandha/eif/releases/tag/v1.0.1"> <img src="https://img.shields.io/badge/release-v1.0.1-blue.svg" alt="latest release" /></a><a href="https://pypi.org/project/eif/1.0.1/"><img src="https://img.shields.io/badge/pypi-v1.0.1-orange.svg" alt="pypi version"/></a>
# Extended Isolation Forest

This is a simple package implementation for the Extended Isolation Forest method. It is an improvement on the original algorithm Isolation Forest which is described (among other places) in this [paper](icdm08b.pdf) for detecting anomalies and outliers from a data point distribution. The original code can be found at [https://github.com/mgckind/iso_forest](https://github.com/mgckind/iso_forest)

For an *N* dimensional data set, Extended Isolation Forest has *N* levels of extension, with *0* being identical to the case of standard Isolation Forest, and *N-1* being the fully extended version.

## Installation


    pip install eif


or directly from the repository


    pip install git+https://github.com/sahandha/eif.git


## Requirements

- numpy

No extra requirements are needed.
In addition, it also contains means to draw the trees created using the [igraph](http://igraph.org/) library. See the example for tree visualizations

## Use

See these notebooks for examples on how to use it

- [Basics](Notebooks/IsolationForest.ipynb)
- [3D Example](Notebooks/general_3D_examples.ipynb)
- [Tree visualizations](Notebooks/TreeVisualization.ipynb)

## Release

### v1.0.1
#### 2018-AUG-08
- Bugfix for multidimensional data

### v1.0.0
#### 2018-JUL-15
- Initial Release
