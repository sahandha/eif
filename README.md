<a href="https://github.com/sahandha/eif/releases/tag/v1.0.2"> <img src="https://img.shields.io/badge/release-v1.0.2-blue.svg" alt="latest release" /></a><a href="https://pypi.org/project/eif/1.0.2/"><img src="https://img.shields.io/badge/pypi-v1.0.2-orange.svg" alt="pypi version"/></a>
# Extended Isolation Forest

This is a simple package implementation for the Extended Isolation Forest method. It is an improvement on the original algorithm Isolation Forest which is described (among other places) in this [paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) for detecting anomalies and outliers from a data point distribution.

The original algorithm suffers from an inconsistency in producing anomaly scores due to slicing operations. Even though the slicing hyperplanes are selected at random, they are always parallel to the coordinate reference frame. The shortcoming can be seen in score maps as presented in the example notebooks in this repository. In order to improve the situation, we propose an extension which allows the hyperplanes to be taken at random angles. The way in which this is done gives rise to multiple levels of extension depending on the dimensionality of the problem. For an *N* dimensional dataset, Extended Isolation Forest has *N* levels of extension, with *0* being identical to the case of standard Isolation Forest, and *N-1* being the fully extended version.

Here we provide the source code for the algorithm as well as documented example notebooks to help get started. Various visualizations are provided such as score distributions, score maps, aggregate slicing of the domain, and tree and whole forest visualizations. most examples are in 2D. We present one 3D example. However, the algorithm works readily with higher dimensional data. 

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

- [Basics](Notebooks/EIF.ipynb)
- [3D Example](Notebooks/general_3D_examples.ipynb)
- [Tree visualizations](Notebooks/TreeVisualization.ipynb)

## Citation

If you use this code, please considering using the following reference:

```
@ARTICLE{2018arXiv181102141H,
   author = {{Hariri}, S. and {Carrasco Kind}, M. and {Brunner}, R.~J.},
    title = "{Extended Isolation Forest}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1811.02141},
 keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
     year = 2018,
    month = nov,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv181102141H},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
The pre-print article can be found [here](https://arxiv.org/abs/1811.021410)


## Releases

### v1.0.2
#### 2018-OCT-01
- Added documentation, examples and software paper

### v1.0.1
#### 2018-AUG-08
- Bugfix for multidimensional data

### v1.0.0
#### 2018-JUL-15
- Initial Release
