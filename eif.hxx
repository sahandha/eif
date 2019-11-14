#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

#define EULER_CONSTANT 0.5772156649

#define RANDOM_ENGINE std::mt19937_64
#define RANDOM_SEED_GENERATOR std::random_device


/****************************
        Class Node
 ****************************/
class Node
{

    private:

    protected:

    public:
	int e;
        int size;
//      double* X;      // unused in original code
        std::vector<double> normal_vector;
        std::vector<double> point;
        Node* left;
        Node* right;
        std::string node_type;

        Node (int, int, double*, double*, int, Node*, Node*, std::string);
        ~Node ();

};


/****************************
        Class iTree
 ****************************/
class iTree
{

    private:
        int exlevel;
        int e;
        int size;
//      int* Q;         // unused in original code
        int dim;
        int limit;
        int exnodes;
//	double* point;		// in original code, but not necessary
//	double* normal_vector;	// in original code, but not necessary
//      double* X;      // unused in original code
    protected:

    public:
        Node* root;

        iTree ();
        ~iTree ();
        void build_tree (double*, int, int, int, int, RANDOM_ENGINE&, int);
        Node* add_node (double*, int, int, RANDOM_ENGINE&);

};


/*************************
        Class Path
 *************************/
class Path
{

    private:
        int dim;
        double* x;
        double e;
    protected:

    public:
        std::vector<char> path_list;
        double pathlength;

        Path (int, double*, iTree);
        ~Path ();
        double find_path (Node*);

};


/****************************
        Class iForest
 ****************************/
class iForest
{

    private:
        int nobjs;
        int dim;
        int sample;
        int ntrees;
        int exlevel;
        double* X;
        double c;
        iTree* Trees;
        unsigned random_seed;

	bool CheckExtensionLevel ();
	bool CheckSampleSize ();
    protected:

    public:
        int limit;
        iForest (int, int, int, int, int);
        ~iForest ();
        void fit (double*, int, int);
        void predict (double*, double*, int);
        void predictSingleTree (double*, double*, int, int);
	    void OutputTreeNodes (int);

};


/********************************
        Utility functions
 ********************************/
inline double inner_product (double*, double*, int);
inline double c_factor (int);
inline std::vector<int> sample_without_replacement (int, int, RANDOM_ENGINE&);
void output_tree_node (Node*, std::string);
void delete_tree_node (Node*);
