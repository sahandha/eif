# Extended Isolation Forest

This is a simple package implementation for the Extended Isolation Forest method. It is an improvement on the original algorithm Isolation Forest which is described (among other places) in this [paper](icdm08b.pdf) for detecting anomalies and outliers from a data point distribution.

For an *N* dimensional data set, Extended Isolation Forest has *N* levels of extension, with *0* being identical to the case of standard Isolation Forest, and *N-1* being the fully extended version.

## Installation

  pip install git+https://github.com/mgckind/eif.git


## Requirements

No extra requirements are needed, It also contains means to draw the trees created using the [igraph](http://igraph.org/) library. See the example for tree visualizations

## Use

See these notebooks for examples on how to use it

- [Basics](Notebooks/IsolationForest.ipynb)
- [3D Example](Notebooks/general_3D_examples.ipynb)
- [Tree visualizations](Notebooks/TreeVisualization.ipynb)
