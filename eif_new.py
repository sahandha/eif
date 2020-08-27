""" Extended Isolation forest functions

This is the implementation of the Extended Isolation Forest anomaly detection algorithm. This extension, improves the consistency and reliability of the anomaly score produced by standard Isolation Forest represented by Liu et al.
Our method allows for the slicing of the data to be done using hyperplanes with random slopes which results in improved score maps. The consistency and reliability of the algorithm is much improved using this extension.

"""

__author__ = 'Matias Carrasco Kind & Sahand Hariri (rewritten by Leszek Pryszcz)'
import numpy as np
import os
from version import __version__

def c_factor(n):
    """
    Average path length of unsuccesful search in a binary search tree given n points
    
    Parameters
    ----------
    n : int
        Number of data points for the BST.

    Returns
    -------
    float
        Average path length of unsuccesful search in a BST
    """
    #if n<2: return 0
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

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
        self.compute_paths = self.score_samples
        
    def fit(self, X):
        """ Fit the ensemble of trees on data from X. 

        Parameters
        ----------
        X: 2D array (samples, features)
            Data to be trained on.
        
        Returns
        -------
        iForest trained on dataset X. 
        """
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
        maxtreei = 2**(self.limit+1)-1
        self.normal = np.random.normal(0, 1, size=(self.ntrees, maxtreei, self.dim))
        self.uniform = np.random.uniform(size=(self.ntrees, maxtreei, self.dim))
        if self.dim-self.exlevel-1: # shit, this may have replacements :/
            self.choice = np.random.choice(self.dim, size=(self.ntrees, maxtreei, self.dim-self.exlevel-1))
        # populate trees
        dtype = [("n", "%sf4"%self.dim), ("pdotn", "f4"), ("left", "u2"), ("right", "u2"), ("size", "u2")] 
        self.trees = np.zeros((self.ntrees, maxtreei), dtype=dtype)
        #idx = np.random.choice(X.shape[0], (self.ntrees, self.sample))
        for i in range(self.ntrees):
            self.treei = i
            self.nodei = -1
            self.populate_nodes(X[np.random.choice(X.shape[0], self.sample, replace=False)])
        # clean-up
        del self.treei, self.nodei, self.uniform, self.normal
        if self.dim-self.exlevel-1: del self.choice
        return self

    def populate_nodes(self, X, e=0):
        """Builds the tree recursively from a given node (e).
        By default starts from root note (e=0)
        """
        self.nodei += 1
        # for terminal nodes store only the size of dataset at final split
        if e==self.limit or len(X)<2:
            self.trees["size"][self.treei, self.nodei] = len(X)
            # and make sure all trees have nodes in identical positions/order in the array
            if e<self.limit: self.nodei += 2**(1+self.limit-e)-2
        # for internal nodes store everything
        else:
            # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            n = self.normal[self.treei, self.nodei]
            # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            if self.dim-self.exlevel-1:
                n[self.choice[self.treei, self.nodei]] = 0
            # Picking a random intercept point for the hyperplane splitting data.
            p = self.uniform[self.treei, self.nodei]*(X.max(axis=0)-X.min(axis=0)) + X.min(axis=0)
            pdotn = p.dot(n) # calculating pdotn here will make classification faster and take less space to store
            # Criteria that determines if a data point should go to the left or right child node.
            w = X.dot(n) < pdotn
            # add left nodes
            idx = self.nodei
            nodeL = self.nodei+1
            self.populate_nodes(X[w], e+1)
            # add right nodes
            nodeR = self.nodei+1
            self.populate_nodes(X[~w], e+1)
            # finally store current node
            self.trees[self.treei, idx] = n, pdotn, nodeL, nodeR, len(X)

    def score_samples(self, X):
        """
        Compute anomaly scores for all data points in a dataset X. 

        Parameters
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
        n, pdotn, left, right, sizes = trees["n"], trees["pdotn"], trees["left"], trees["right"], trees["size"]
        # iterate over samples
        for xi in range(X.shape[0]):
            tidx = np.arange(trees.shape[0])
            # get childs based on dot product from all trees
            ni = np.where(X[xi].dot(n[:, 0].T) < pdotn[:, 0], left[:, 0].T, right[:, 0].T)
            # iterate through all levels of all trees at once
            for e in range(1, self.limit):
                # get for e nodes from all trees
                w = X[xi].dot(n[tidx, ni].T) < pdotn[tidx, ni]
                # get childs from all trees
                ni = np.where(w, left[tidx, ni].T, right[tidx, ni].T)
                # capture terminal nodes
                S[xi] += e*(ni==0).sum()
                # trim the trees for which terminal node was reached
                tidx, ni = tidx[ni>0], ni[ni>0]
                #if not len(ni): break
            # store number trees for which the deepest node was reached
            # and their sizes (size matters only at terminal nodes)
            size = sizes[tidx, ni]
            S[xi] += self.limit*len(ni) + c_factor(size[size>1]).sum()
        # divide by total number of trees in the fores
        S *= 1. / len(trees)
        # and calculate anomaly scores
        S = 2.0**(-S / c_factor(self.sample))
        return S

    def score_samples_without_using_childidx(self, X):
        """This implementation doesn't rely on child info from trees
        but its slightly slower than the default (~7%).
        The results aren't exactly the same though. 
        """
        # this will store scores
        S = np.zeros(X.shape[0])
        trees = self.trees
        n, pdotn, sizes = trees["n"], trees["pdotn"], trees["size"]
        #powers = np.power(2, np.arange(self.limit+1))
        # iterate over samples
        for xi in range(X.shape[0]):
            ni = np.zeros(len(trees), dtype='int')
            w = X[xi].dot(n[:, 0].T) < pdotn[:, 0]
            ni[w] += 1
            ni[~w] += 2**self.limit
            tidx = np.arange(trees.shape[0])
            for e in range(1, self.limit):
                w = X[xi].dot(n[tidx, ni].T) < pdotn[tidx, ni]
                ni[w] += 1
                ni[~w] += 2**(self.limit-e)
                sel = sizes[tidx, ni]>1
                S[xi] += e*(~sel).sum()
                tidx, ni = tidx[sel], ni[sel]
            # the size matters only at terminal nodes
            size = sizes[tidx, ni]
            S[xi] += self.limit*len(ni) + c_factor(size[size>1]).sum()
        # divide by total number of trees in the fores
        S *= 1. / len(trees)
        # and calculate anomaly scores
        S = 2.0**(-S / c_factor(self.sample))
        return S
