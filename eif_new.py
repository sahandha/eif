""" Extended Isolation forest functions

This is the implementation of the Extended Isolation Forest anomaly detection algorithm. This extension, improves the consistency and reliability of the anomaly score produced by standard Isolation Forest represented by Liu et al.
Our method allows for the slicing of the data to be done using hyperplanes with random slopes which results in improved score maps. The consistency and reliability of the algorithm is much improved using this extension.

This fork of https://github.com/sahandha/eif was rewritten from scratch. More information:
- https://github.com/sahandha/eif/issues/18
- https://github.com/sahandha/eif/pull/24
"""

__author__ = 'Leszek Pryszcz (relying on method developed by Matias Carrasco Kind & Sahand Hariri)'
import numpy as np
import os
#from version import __version__
from numba import jit#, float32, int32

@jit
def c_factor(n):
    """Return average path length of unsuccesful search in a binary search tree
    given n points.
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

@jit
def minmax(x):
    """np.min(x, axis=0), np.max(x, axis=0) for 2D array but faster"""
    m, n = len(x), len(x[0])
    mi, ma = np.empty(n), np.empty(n)
    mi[:] = ma[:] = x[0]
    for i in range(1, m):
        for j in range(n):
            if x[i, j]>ma[j]: ma[j] = x[i, j]
            elif x[i, j]<mi[j]: mi[j] = x[i, j]
    return mi, ma

@jit
def split(x, w):
    """x[w], x[~w] but faster"""
    k = l = 0
    a, b = np.empty_like(x), np.empty_like(x)
    for i in range(len(x)):
        if w[i]: 
            a[k] = x[i]
            k += 1
        else:
            b[l] = x[i]
            l += 1
    return a[:k], b[:l]

@jit
def scale_minmax(a, mi, ma):
    """Return a scaled by mi-ma"""
    return a * (ma-mi) + mi

@jit
def dot(p, n):
    """Return p.dot(n)"""
    return p.dot(n)

@jit
def update_nodes(ni, w, limit, e):
    """Update node index base on w, limit and e"""
    r = 2**(limit-e)
    for i in range(len(ni)):
        if w[i]: ni[i] += 1
        else: ni[i] += r
    return ni

@jit
def score_false(e, sel):
    """Return scores for internal-terminal nodes"""
    return e*(~sel).sum()

@jit
def score_terminal(limit, ni, size):
    """Return limit*len(ni) + c_factor(size[size>1]).sum()"""
    return limit*len(ni) + c_factor(size[size>1]).sum()

class iForest(object):
    """
    Create an iForest object holding trees (iTree_array objects) trained on provided data (X).

    Parameters
    ----------
    ntrees : int, default=200
        Number of trees in the forest. 
    sample : int, default=min(256, X.shape[1])
        Size of the sample to be used for tree creation.
    limit : int, default=int(np.ceil(np.log2(sample)))
        Maximum depth a tree can have. 
    exlevel: int, default=X.shape[1]-1
        Extension level to be used in the creating splitting critera.    
    random_state: int, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
    
    Attributes
    ----------
    trees : numpy array storing tree information
   
    Methods
    -------
    fit(X)
        Trains ensemble of trees on data X.
    
    score_samples(X)
        Computes the anomaly scores for data X. 
    """
    def __init__(self, ntrees=200, sample=256, limit=None, exlevel=None, random_state=None):

        # define random seed
        if random_state is not None:
            np.random.seed(random_state)
        self.exlevel = exlevel
        self.sample = sample
        self.limit = limit
        self.ntrees = ntrees
        self.compute_paths = self.score_samples#_using_childs
        self.cut_off_ = 0.5
        
    def fit_predict(self, X):
        """Return array with outlier predictions: normal (False) and outlier (True)."""
        self.fit(X)
        return self.score_samples(X)>self.cut_off_
    
    def predict(self, X):
        """Return normal (0) and outlier (1) predictions"""
        return self.score_samples(X)>self.cut_off_

    def fit(self, X):
        """Return iForest trained on data from X. 

        Parameters
        ----------
        X: 2D array (samples, features)
            Data to be trained on.
        """
        #X = X.astype(dtype="f8")
        # initialise variables based on X
        self.sample = min(self.sample, X.shape[0])
        self.dim = X.shape[1]
        if self.exlevel is None:
            self.exlevel = self.dim-1
        # 0 < exlevel < X.shape[1]
        if self.exlevel < 0 or self.exlevel >= self.dim:
            raise Exception("Extension level has to be an integer between 0 and %s."%(self.dim-1,))        
        # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        if not self.limit:
            self.limit = int(np.ceil(np.log2(self.sample)))
        # sample from normal distribution in order to save time later
        maxnodes = 2**(self.limit+1)-1
        self.rng = np.random.default_rng()
        self.normal = np.random.normal(0, 1, size=(self.ntrees, maxnodes, self.dim))
        self.uniform = np.random.uniform(size=(self.ntrees, maxnodes, self.dim))
        if self.dim-self.exlevel-1: # shit, this may have replacements :/
            self.choice = np.random.choice(self.dim, size=(self.ntrees, maxnodes, self.dim-self.exlevel-1))#.astype(dtype="f4")
        # populate trees
        dtype = [("n", "%sf2"%self.dim), ("pdotn", "f2"), ("size", "u2")]#, ("left", "u2"), ("right", "u2")]
        self.trees = np.zeros((self.ntrees, maxnodes), dtype=dtype)
        for treei in range(self.ntrees): 
            idx = np.random.choice(X.shape[0], self.sample, replace=False)
            self.populate_nodes(X[idx], treei)
        # clean-up
        del self.normal, self.uniform
        if self.dim-self.exlevel-1: del self.choice
        return self

    def populate_nodes(self, X, treei, nodei=0, e=0):
        """Builds the tree recursively from a given node (e). 
        By default starts from root note (e=0) and make all trees symmetrical. 
        """
        # for terminal nodes store only the size of dataset at final split
        if e==self.limit or len(X)<2:
            self.trees["size"][treei, nodei] = len(X)
        # for internal nodes store everything
        else:
            # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            n = self.normal[treei, nodei]
            # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            if self.dim-self.exlevel-1:
                n[self.choice[treei, nodei]] = 0
            # Picking a random intercept point for the hyperplane splitting data.
            mi, ma = minmax(X) # much faster than X.min(axis=0), X.max(axis=0)
            # calculating pdotn here will make classification faster and take less space to store
            pdotn = dot(scale_minmax(self.uniform[treei, nodei], mi, ma), n)
            # Criteria that determines if a data point should go to the left or right child node.
            w = X.dot(n) < pdotn # here X.dot(n) uses BLAS so no need to optimise ;)
            # store current node
            self.trees[treei, nodei] = n, pdotn, len(X)
            # split data from X in order to populate left & right nodes
            left, right = split(X, w) # faster than X[~w], X[w]
            # and add left & right node
            self.populate_nodes(left, treei, nodei+1, e+1)
            self.populate_nodes(right, treei, nodei+2**(self.limit-e), e+1) #2**(self.limit-e)

    def score_samples(self, X):
        """
        Compute anomaly scores for all data points in a dataset X. 

        ----------
        X: 2D array (samples, features)
            Data to be scored on. 

        Returns
        -------
        S: 1D array (X.shape[0])
            Anomaly scores calculated for all samples from all trees. 
        """
        # this will store scores
        S = np.zeros(X.shape[0])
        trees = self.trees
        n, pdotn, sizes = trees["n"], trees["pdotn"], trees["size"]
        # iterate over samples
        for xi in range(X.shape[0]):
            ni = np.zeros(len(trees), dtype='int')
            w = X[xi].dot(n[:, 0].T) < pdotn[:, 0]
            ni = update_nodes(ni, w, self.limit, 0)
            tidx = np.arange(trees.shape[0])
            for e in range(1, self.limit):
                w = X[xi].dot(n[tidx, ni].T) < pdotn[tidx, ni]
                ni = update_nodes(ni, w, self.limit, e)
                sel = sizes[tidx, ni]>0
                S[xi] += score_false(e, sel) #e*(~sel).sum()
                tidx, ni = tidx[sel], ni[sel]
            # the size matters only at terminal nodes
            size = sizes[tidx, ni]
            S[xi] += score_terminal(self.limit, ni, size) #self.limit*len(ni) + c_factor(size[size>1]).sum()
        # calculate anomaly scores
        S = np.power(2, -S / len(trees) / c_factor(self.sample))
        return S
