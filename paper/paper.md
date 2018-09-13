---
title: 'eif: Extended Isolation Forest'
tags:
  - Python
  - Machine Learning
  - Anomaly Detection
authors:
  - name: Sahand Hariri
    orcid:
    affiliation: 1
  - name: Matias Carrasco Kind
    orcid: 0000-0002-4802-3194
    affiliation: 2
affiliations:
  - name: Mechanical Science and Engineering, University of Illinois at Urbana-Champaign, USA
    index: 1
  - name: National Center for Supercomputing Applications, University of Illinois at Urbana-Champaign. 1205 W Clark St, Urbana, IL USA 61801
    index: 2
date: 11 Sep 2018
bibliography: paper.bib
---


# Summary

The problem of anomaly detection has wide range of applications in various fields including scientific applications. anomalous data can have as much scientific value as normal data or in some cases even more. In this paper, we present an extension to the model-free anomaly detection algorithm, Isolation Forest. This extension, named Extended Isolation Forest (EIF), improves the consistency and reliability of the anomaly score produced by standard methods for a given data point. We show that the standard Isolation Forest produces inconsistent scores using score maps, and that these score maps suffer from an artifact produced as a result of how the criteria for branching operation of the binary tree is selected. We propose two different approaches for improving the reliability of anomaly detection. First we propose methods for transforming the data before the creation of each tree in the forest. Second, which is the preferred method of this paper, is to allow the slicing of the data to use hyperplanes with random slopes. This approach results in improved score maps. We show that the consistency and reliability of the algorithm is much improved using this extension by looking at the variance of scores of data points distributed along constant score lines. We find no appreciable difference in the rate of convergence nor in computational time between the standard Isolation Forest and EIF which highlights its potential as anomaly detection algorithm

# Motivation

![a) Shows an example tree formed from the example data while b) shows the forest generated where each tree is represented by a radial line from the center to  the  outer  circle.  Anomalous  points  (shown  in  red)  are  isolated  very  quickly,which means they reach shallower depths than nominal points (shown in blue).](forest.png)

While various techniques exist for approaching anomaly detection, Isolation Forest [@Liu2012] is one with unique capabilities. This algorithm can readily work on high dimensional data, it is model free, and it is computationally scalable. In the algorithm, data is sub-sampled, and processed in a tree structure based on random cuts in the values of randomly selected features in the data set. Those samples that travel deeper into the tree branches are less likely to be anomalous, while shorter branches are indicative of anomaly. As such, the aggregated lengths of the tree branches provide for a measure of anomaly or an “anomaly score” for every given point.

![ Comparison of the standard Isolation Forest with rotated Isolation Forest, and Extended Isolation Forest for the case of two blobs.](score_maps.png)


# The `eif` algorithm


| **Algorithm 1** $iForest(X, t, \psi)$ |
| ---------------------------------- |
| **Require:** $X$ - input data, $t$ - number of trees, $h$ - sub-sampling size |
| **Ensure:** a set of $t$ $iTrees$ |
|   $\quad$ 1. **Initialize** $Forest$ |
|   $\quad$ 2. set height limit $l = ceiling(\log_2 \psi)$ |
|   $\quad$ 3. **for** $i = 1$ to $t$ **do** |
|   $\quad$ 4. $X' \gets sample(X, \psi)$ |
|   $\quad$ 5. $Forest \gets Forest \cup iTree(X', 0, l)$ |
|   $\quad$ 6. **end for** |

| **Algorithm 2** $iTree(X, e, l)$ |
| -------------------------------- |
| **Require:** $X$ - input data, $e$ - current tree height, $l$ - height limit |
| **Ensure:** an $iTree$ |
| $\quad$ 1. **if** $e \geq l$ or $|X| \leq 1$ **then** |
| $\quad$ 2. $\quad$ **return** $exNode\{Size \gets |X|\}$ |
| $\quad$ 3. **else** |
| $\quad$ 4. get a random normal vector $\vec{n} \in {\rm I\!R}^{|X|}$ where each coordinate is $\sim \mathcal{N}(0,\,1)$|
| $\quad$ 5. randomly select an intercept point $\vec{p} \in  {\rm I\!R}^{|X|}$ in the range of $X$ |
| $\quad$ 6. set coordinates of $\vec{n}$ to zero according to extension level |
| $\quad$ 7. $X_l \gets filter(X,(X-\vec{p})\cdot \vec{n} \leq 0)$ |
| $\quad$ 8. $X_r \gets filter(X,(X-\vec{p})\cdot \vec{n} > 0)$  |
| $\quad$ 9. **return** $inNode\{$ |
| $\quad$$\quad$$\quad$  $Left \gets iTree(X_l,e+1, l),$ |
| $\quad$$\quad$$\quad$  $Right \gets iTree(X_r,e+1,l),$ |
| $\quad$$\quad$$\quad$  $Normal \gets \vec{n},$ |
| $\quad$$\quad$$\quad$  $Intercept \gets \vec{p} \}$  |
| $\quad$ 10. **end if** |


| **Algorithm 3**  $PathLength(x,T,e)$|
| -------------------------------- |
| **Require:** $\vec{x}$ - an instance, $T$ - an iTree, $e$ - current path length; initialized to 0|
| **Ensure:**  path length of $\vec{x}$ |
| $\quad$ 1. **if** $T$ is an external node **then** |
| $\quad$ 2. $\quad$ **return** $e + c(T.size)\{c(.) \text{ is defined in Equation 1}\}$ |
| $\quad$ 3. **end if** |
| $\quad$ 4. $\vec{n} \gets T.Normal$ |
| $\quad$ 5. $\vec{p} \gets T.Intercept$ |
| $\quad$ 6. **if** {$(\vec{x}-\vec{p})\cdot \vec{n} \leq 0$} **then** |
| $\quad$ 7. $\quad$ return $PathLength(\vec{x},T.left, e+1)$ |
| $\quad$ 8. **else if**  {$(\vec{x}-\vec{p})\cdot \vec{n} > 0$} **then** |
| $\quad$ 9. **return** $PathLength(\vec{x},T.rigth, e+1)$ |
| $\quad$ 10. **end if** |


![a) Shows the dataset used, some sample anomalous data points discovered using the algorithm are highlighted in black. We also highlight some nominal points in red. In b), we have the distribution of anomaly scores obtained by the algorithm.](example.png)



# Acknowledgements
MCK is supported by the National Science Foundation under Grant NSF AST 07-15036 and NSF AST 08-13543

# References
