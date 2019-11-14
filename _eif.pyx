# Cython wrapper for Extended Isolation Forest

# distutils: language = C++
# distutils: sources  = eif.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np
from version import __version__

cimport __eif

np.import_array()

cdef class iForest:
    cdef int size_X
    cdef int dim
    cdef int _ntrees
    cdef int _limit
    cdef int sample
    cdef int tree_index
    cdef int exlevel
    cdef __eif.iForest* thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__ (self, np.ndarray[double, ndim=2] X not None, int ntrees, int sample_size, int limit=0, int ExtensionLevel=0, int seed=-1):
        if ExtensionLevel < 0:
            raise Exception("Wrong Extension")
        self.thisptr = new __eif.iForest (ntrees, sample_size, limit, ExtensionLevel, seed)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        self.size_X = X.shape[0]
        self.dim = X.shape[1]
        self.sample = sample_size
        self._ntrees = ntrees
        self._limit = self.thisptr.limit
        self.exlevel = ExtensionLevel
        self.thisptr.fit (<double*> np.PyArray_DATA(X), self.size_X, self.dim)

    @property
    def ntrees(self):
        return self._ntrees

    @property
    def limit(self):
        return self._limit

    def __dealloc__ (self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_paths (self, np.ndarray[double, ndim=2] X_in=None):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if X_in is None:
            S = np.empty(self.size_X, dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), NULL, 0)
        else:
            if not X_in.flags['C_CONTIGUOUS']:
                X_in = X_in.copy(order='C')
            S = np.empty(X_in.shape[0], dtype=np.float64, order='C')
            self.thisptr.predict (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X_in), X_in.shape[0])
        return S

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_paths_single_tree (self, np.ndarray[double, ndim=2] X_in=None, tree_index=0):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        if X_in is None:
            S = np.empty(self.size_X, dtype=np.float64, order='C')
            self.thisptr.predictSingleTree (<double*> np.PyArray_DATA(S), NULL, 0, tree_index)
        else:
            if not X_in.flags['C_CONTIGUOUS']:
                X_in = X_in.copy(order='C')
            S = np.empty(X_in.shape[0], dtype=np.float64, order='C')
            self.thisptr.predictSingleTree (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X_in), X_in.shape[0], tree_index)
        return S

    def output_tree_nodes (self, int tree_index):
        self.thisptr.OutputTreeNodes (tree_index)
