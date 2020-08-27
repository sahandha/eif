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
    X : 2D array (samples, features)
        Data to be trained on.
    ntrees : int, default=200
        Number of trees in the forest. 
    sample : int, default=min(256, X.shape[1])
        Size of the sample to be used for tree creation.
    limit : int, default=int(np.ceil(np.log2(sample)))
        Maximum depth a tree can have. 
    exlevel: int, default=0
        Extension level to be used in the creating splitting critera.    
    random_state: int, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
    
    Attributes
    ----------
    Trees : list
        A list of fitted tree objects.
   
    Methods
    -------
    score_samples(X)
        Computes the anomaly scores for data X. 
    """
    def __init__(self, X, ntrees=200, sample=256, limit=None, exlevel=0, random_state=None):
        self.exlevel = exlevel
        if self.exlevel < 0 or self.exlevel >= X.shape[1]:
            raise Exception("Extension level has to be an integer between 0 and %s."%(X.shape[1]-1,))

        # define random seed
        if random_state is not None:
            np.random.seed(random_state)
            
        self.ntrees = ntrees
        self.sample = min(sample, X.shape[0])
        self.compute_paths = self.score_samples1
        # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        self.limit = limit if limit else int(np.ceil(np.log2(self.sample)))
        # This loop builds an ensemble of iTrees (the forest).
        idx = np.random.choice(X.shape[0], (self.ntrees, self.sample))
        self.Trees = [iTree(X[idx[i]], self.limit, self.exlevel) for i in range(self.ntrees)]
        #self.Trees = [iTree(X[np.random.choice(X.shape[0], self.sample, replace=False)], self.limit, self.exlevel) for i in range(self.ntrees)]

    def score_samples1(self, X):
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
        S = np.zeros(X.shape[0])
        trees = np.array([t.nodes for t in self.Trees])
        n, pdotn, left, right, sizes = trees["n"], trees["pdotn"], trees["left"], trees["right"], trees["size"]
        for xi in range(X.shape[0]):
            ni = np.where(X[xi].dot(n[:, 0].T) < pdotn[:, 0], left[:, 0].T, right[:, 0].T)
            tidx = np.arange(trees.shape[0])
            for e in range(1, self.limit):
                w = X[xi].dot(n[tidx, ni].T) < pdotn[tidx, ni]
                ni = np.where(w, left[tidx, ni].T, right[tidx, ni].T)
                S[xi] += e*(ni==0).sum()
                tidx, ni = tidx[ni>0], ni[ni>0]
            # the size matters only at terminal nodes
            size = sizes[tidx, ni]
            S[xi] += self.limit*len(ni) + c_factor(size[size>1]).sum()
        S *= 1. / len(trees)
        S = 2.0**(-S / c_factor(self.sample))
        return S
        
    def score_samples0(self, X):
        Eh = np.zeros(X.shape[0])
        for t in self.Trees:
            t.get_paths(X=X)
            Eh += t.scores
            del t.scores
        Eh *= 1.0 / self.ntrees
        S = 2.0**(-Eh / c_factor(self.sample)) 
        return S

class iTree(object):

    """
    A single tree in the forest that is build using a unique subsample
    and stored in numpy array.

    Attributes
    ----------
    nodes: 2D array of shape (nodes, (n, pdotn, (left_child, right_child), size))
        An array storing tree structure and all information for splits. 
    dim: int
        number of features
    limit: int
        max tree depth
    exlevel: int
        Exention level to be used in the creating splitting critera.
    
    Methods
    -------
    get_paths(X)
        Get tree depth reach for every sample from X and save it into self.scores. 
    """

    def __init__(self, X, limit, exlevel):
        self.limit = limit
        self.dim = X.shape[1]
        self.exlevel = exlevel
        # for each split there can be n^2 new nodes,
        # so in total sum(2^0, 2^1, ... 2^n) where n=limit+2
        maxtreei = 2**(self.limit+1)-1 
        # sample from normal distribution in order to save time later
        self.normal = np.random.normal(0, 1, size=(maxtreei, self.dim))
        self.uniform = np.random.uniform(size=(maxtreei, self.dim))
        if self.dim-self.exlevel-1: # shit, this may have replacements :/
            self.choice = np.random.choice(self.dim, size=(maxtreei, self.dim-self.exlevel-1))
        # store all nodes in single array - here probably f2 would be more than enough
        dtype = [("n", "%sf4"%self.dim), ("pdotn", "f4"), ("left", "u2"), ("right", "u2"), ("size", "u2")] 
        self.nodes = np.zeros(maxtreei, dtype=dtype)
        # track array population
        self.treei = -1
        self._populate_nodes(X)
        # trim unused nodes
        #self.nodes = self.nodes[:self.treei+1]
        # clean-up
        del self.normal, self.uniform, self.treei
        if self.dim-self.exlevel-1: del self.choice

    def get_paths(self, X=[], nodei=0, e=0, idx=None):
        """Stores the paths as self.scores for data
        based on the splitting criteria stored at each node.
        """
        # initialise
        if not nodei:
            idx = np.arange(X.shape[0])
            self.scores = np.zeros(X.shape[0])
        # unload data
        n, pdotn, left, right, size = self.nodes[nodei]
        # for internal nodes
        if left:
            # split data accordingly to each node criteria
            w = X.dot(n) < pdotn
            # and process two partition in child nodes - can this be multi threaded?
            self.get_paths(X[w], left, e+1, idx[w])
            self.get_paths(X[~w], right, e+1, idx[~w])
        # store information from terminal nodes
        elif size>1:
            self.scores[idx] = e + c_factor(size) if size>1 else 0
        else:
            self.scores[idx] = e #+ c_factor(size) if size>1 else 0

    def _populate_nodes(self, X, e=0):
        """Builds the tree recursively from a given node (e).
        By default starts from root note (e=0)
        """
        self.treei += 1
        # for terminal nodes store only the size of dataset at final split
        if e==self.limit or len(X)<2:
            self.nodes["size"][self.treei] = len(X)
            # and make sure all trees have nodes in identical positions/order in the array
            if e<self.limit: self.treei += 2**(1+self.limit-e)-2
        # for internal nodes store everything
        else:
            # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            n = self.normal[self.treei]
            # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            if self.dim-self.exlevel-1:
                n[self.choice[self.treei]] = 0
            # Picking a random intercept point for the hyperplane splitting data.
            p = self.uniform[self.treei]*(X.max(axis=0)-X.min(axis=0)) + X.min(axis=0)
            pdotn = p.dot(n) # calculating pdotn here will make classification faster and take less space to store
            # Criteria that determines if a data point should go to the left or right child node.
            w = X.dot(n) < pdotn
            # add left nodes
            idx = self.treei
            nodeL = self.treei+1
            self._populate_nodes(X[w], e+1)
            # add right nodes
            nodeR = self.treei+1
            self._populate_nodes(X[~w], e+1)
            # finally store current node
            self.nodes[idx] = n, pdotn, nodeL, nodeR, len(X)#, e
