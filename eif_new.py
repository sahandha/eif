""" Extended Isolation forest functions

This is the implementation of the Extended Isolation Forest anomaly detection algorithm. This extension, improves the consistency and reliability of the anomaly score produced by standard Isolation Forest represented by Liu et al.
Our method allows for the slicing of the data to be done using hyperplanes with random slopes which results in improved score maps. The consistency and reliability of the algorithm is much improved using this extension.

"""

__author__ = 'Matias Carrasco Kind & Sahand Hariri (rewritten by Leszek Pryszcz)'
import numpy as np
import os
from version import __version__

def c_factor(n) :
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
        self.compute_paths = self.score_samples
        # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        self.limit = limit if limit else int(np.ceil(np.log2(self.sample)))            
        # This loop builds an ensemble of iTrees (the forest).        
        self.Trees = [iTree(X[np.random.choice(X.shape[0], self.sample, replace=False)], 
                            self.limit, self.exlevel) for i in range(self.ntrees)]
                
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
        Eh = np.zeros(X.shape[0])
        for t in self.Trees:
            t.get_paths(X)
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
        # so in total sum(2^0, 2^1, ... 2^n) where n=limit
        maxtreei = sum(2**i for i in range(0, self.limit+1))
        # sample from normal distribution in order to save time later
        self.normal = np.random.normal(0, 1, size=(maxtreei, self.dim))
        self.uniform = np.random.uniform(size=(maxtreei, self.dim))
        if self.dim-self.exlevel-1: # shit, this may have replacements :/
            self.choice = np.random.choice(self.dim*maxtreei, size=(maxtreei, self.dim-self.exlevel-1), replace=False)%self.dim
        # store all nodes in single array - here probably f2 would be more than enough
        self.nodes = np.zeros(maxtreei, dtype="(%s,)f2, f2, 2u2, u2"%self.dim)
        # track array population
        self.treei = -1
        self._populate_nodes(X)
        # trim unused nodes
        self.nodes = self.nodes[:self.treei+1]
        # clean-up
        del self.normal, self.uniform, self.treei
        if self.dim-self.exlevel-1: del self.choice

    def get_paths(self, X, nodei=0, e=0, idx=None):
        """Stores the paths as self.scores for data
        based on the splitting criteria stored at each node.
        """
        # initialize tree
        if not nodei:
            idx = np.arange(X.shape[0])
            self.scores = np.zeros(X.shape[0])
        # unload node info
        n, pdotn, (left, right), size = self.nodes[nodei]
        # for internal nodes
        if left:
            # split data accordingly to each node criteria
            w = X.dot(n) < pdotn
            # and process two partition in child nodes - can this be multi threaded?
            self.get_paths(X[w], left, e+1, idx[np.argwhere(w).flatten()])
            self.get_paths(X[~w], right, e+1, idx[np.argwhere(~w).flatten()])
        # store information from terminal nodes
        else:
            self.scores[idx] = e + c_factor(size) if size>1 else e

    def _populate_nodes(self, X, e=0):
        """Builds the tree recursively from a given node (e).
        By default starts from root note (e=0)
        """
        self.treei += 1
        # for terminal nodes store only the size of dataset at final split
        if e >= self.limit or len(X)<2:
            self.nodes[self.treei][-1] = len(X) 
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
            self.nodes[idx] = n, pdotn, (nodeL, nodeR), len(X)
