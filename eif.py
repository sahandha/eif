    "Extended Isolation forest functions"

__author__ = 'Matias Carrasco Kind & Sahand Hariri'
import numpy as np
import random as rn
import os
import warnings
from version import __version__


def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

class iForest(object):
    '''
    Creates an iForest object. This object holds the data as well as the trained trees (iTree objects).
    '''
    def __init__(self, X, ntrees,  sample_size, limit=None, ExtensionLevel=0):
        '''
        iForest(X, ntrees,  sample_size, limit=None, ExtensionLevel=0)
        Initialize a forest by passing in training data, number of trees to be used and the subsample size.
            * X - Training data. List of [x1,x2,...,xn] coordinate points.
            * ntrees - Number of trees to be used. Integer value.
            * sample_size - The size of the subsample to be used in creation of each tree. Integer. Must be smaller than |X|
            * limit - The maximum allowed tree depth. This is by default set to average length of unsucessful search in a binary tree. Integer.
            * ExtensionLevel - Specifies degree of freedom in choosing the hyperplanes for dividing up data. Integer. Must be smaller than the dimension n of the dataset.
        '''
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.exlevel = ExtensionLevel
        self.CheckExtensionLevel()                                              # Extension Level check. See def for explanation.
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))                     # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        self.c = c_factor(self.sample)
        for i in range(self.ntrees):                                            # This look builds an ensemble of iTrees (the forest).
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(iTree(X_p, 0, self.limit, exlevel=self.exlevel))

    def CheckExtensionLevel(self):
        '''
        This function makes sure the extension level provided by the user does not exceed the dimension of the data. An exception will be raised in the case of a violation.
        '''
        dim = self.X.shape[1]
        if self.exlevel < 0:
            raise Exception("Extension level has to be an integer between 0 and "+ str(dim-1)+".")
        if self.exlevel > dim-1:
            raise Exception("Your data has "+ str(dim) + " dimensions. Extension level can't be higher than " + str(dim-1) + ".")

    def compute_paths(self, X_in = None):
        '''
        compute_paths(X_in = None)
        Compute anomaly scores for all data points in a dataset X_in
            * X_in - Data to be scored. iForest.Trees are used for computing the depth reached in each tree by each data point.
        '''
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i],self.Trees[j]).path*1.0            # Compute path length for each point
            Eh = h_temp/self.ntrees                                             # Average of path length travelled by the point in all trees.
            S[i] = 2.0**(-Eh/self.c)                                            # Anomaly Score
        return S

class Node(object):
    '''
    A single node from each tree (each iTree object). Nodes containe information on hyperplanes used for data division, date to be passed to left and right nodes, whether they are external or internal nodes.
    '''
    def __init__(self, X, n, p, e, left, right, node_type = '' ):
        '''
        Node(X, n, p, e, left, right, node_type = '' )
        Create a node in a given tree (iTree objectg)
            * X - Training data available to each node. List of [x1,x2,...,xn] coordinate points.
            * n - Normal vector for the hyperplane used for splitting data. List of floats.
            * p - Intercept point for the hyperplane used for splitting data. List of floats.
            * left - Left child node. A Node object.
            * right - Right child node. A Node object.
            * node_type - Specifies if the node is external or internal. String. Takes two values: 'exNode', 'inNode'.
        '''
        self.e = e
        self.size = len(X)
        self.X = X # to be removed
        self.n = n
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type

class iTree(object):

    """
    A single tree in the forest that is build using a unique subsample.
    """

    def __init__(self,X,e,l, exlevel=0):
        '''
        iTree(X, e, l, exlevel=0)
        Create a tree
            * X - Subsample of training data. |X| = iForest.sample_size. List of [x1,x2,...,xn] coordinate points
            * e - Depth of the tree as it is being traversed down. Integer. e <= l.
            * l - The maximum depth the tree can reach before its creation is terminated. Integer.
            * exlevel - Specifies degree of freedom in choosing the hyperplanes for dividing up data. Integer. Must be smaller than the dimension n of the dataset.
        '''
        self.exlevel = exlevel
        self.e = e
        self.X = X                                                              #save data for now. Not really necessary.
        self.size = len(X)
        self.dim = self.X.shape[1]
        self.Q = np.arange(np.shape(X)[1], dtype='int')                         # n dimensions
        self.l = l
        self.p = None                                                           # Intercept for the hyperplane for splitting data at a given node.
        self.n = None                                                           # Normal vector for the hyperplane for splitting data at a given node.
        self.exnodes = 0
        self.root = self.make_tree(X,e,l)                                       # At each node create a new tree, starting with root node.

    def make_tree(self,X,e,l):
        '''
        make_tree(X,e,l)
        Builds the tree recursively from a given node. Returns a Node object.
            * X Subsample of training data. |X| = iForest.sample_size. List of [x1,x2,...,xn] coordinate point.
            * e - Depth of the tree as it is being traversed down. Integer. e <= l.
            * l - The maximum depth the tree can reach before its creation is terminated. Integer.
        '''
        self.e = e
        if e >= l or len(X) <= 1:                                               # A point is isolated in traning data, or the depth limit has been reached.
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.n, self.p, e, left, right, node_type = 'exNode')
        else:                                                                   # Building the tree continues. All these nodes are internal.
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            idxs = np.random.choice(range(self.dim), self.dim-self.exlevel-1, replace=False)  # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            self.n = np.random.normal(0,1,self.dim)                             # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            self.n[idxs] = 0
            self.p = np.random.uniform(mins,maxs)                               # Picking a random intercept point for the hyperplane splitting data.
            w = (X-self.p).dot(self.n) < 0                                      # Criteria that determines if a data point should go to the left or right child node. 
            return Node(X, self.n, self.p, e,\
            left=self.make_tree(X[w],e+1,l),\
            right=self.make_tree(X[~w],e+1,l),\
            node_type = 'inNode' )

class PathFactor(object):
    '''
    Given a single tree (iTree objext) and a data point x = [x1,x2,...,xn], compute the legth of the path traversed by the point on the tree when it reaches an external node.
    '''
    def __init__(self,x,itree):
        '''
        PathFactor(x, itree)
        Given a single tree (iTree objext) and a data point x = [x1,x2,...,xn], compute the legth of the path traversed by the point on the tree when it reaches an external node.
            * x - A data point x = [x1, x2, ..., xn].
            * itree - An iTree object.
        '''
        self.path_list=[]
        self.x = x
        self.e = 0
        self.path  = self.find_path(itree.root)

    def find_path(self,T):
        '''
        find_path(T)
        Given a tree, find the path for a single data point based on the splitting criteria stored at each node.
            * T - An iTree object.
        '''
        if T.ntype == 'exNode':
            if T.size <= 1: return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            p = T.p                                                             # Intercept for the hyperplane for splitting data at a given node.
            n = T.n                                                             # Normal vector for the hyperplane for splitting data at a given node.

            self.e += 1

            if (self.x-p).dot(n) < 0:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)

def all_branches(node, current=[], branches = None):
    '''
    Utility function used in generating a graph visualization. It returns all the branches of a given tree so they can be visualized.
    '''
    current = current[:node.e]
    if branches is None: branches = []
    if node.ntype == 'inNode':
        current.append('L')
        all_branches(node.left, current=current, branches=branches)
        current = current[:-1]
        current.append('R')
        all_branches(node.right, current=current, branches=branches)
    else:
        branches.append(current)
    return branches
