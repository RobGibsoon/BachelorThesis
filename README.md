# Bachelor Thesis
This is the Repository for my Bachleorthesis of the FS2023. 
Supervised by Mathias Fuchs

## Graph Representation Through Topological Descriptors

The objective of the present project is similar to that of graph kernels, i.e. we want to benefit from both the 
universality of graphs for pattern representation and the computational convenience of vectors for pattern recognition. 
In contrast to graph kernels, however, the proposed thesis results in an explicit embedding of graphs from some graph 
domain in a real vector space.

So called constitutional and topological descriptors are numerical quantifiers of the graph topology obtained by the 
application of algebraic operators to matrices representing graphs and whose values are independent of node numbering 
or labelling. Constitutional and topological descriptors are often used, especially in chemoinformatics, 
to numerically describe molecules represented as graphs (e.g., the Zagreb Index or the Winer Index). The aim of this 
project is to implement several descriptors and evaluate their expressiveness in a molecule classification experiment.

## Process
The Bachelorthesis will consist of two parts
### Part 1
Pick indexes and implement methods to build feature vectors.
### Part 2
Classify embeddings with 3 classfiers (SVM, KNN, ANN) and report what accuracy is achieved when classifying the graphs based on the feature
vectors that we get by running the methods from part 1. Classification is binary. For example if a molecul is carcinogenic or not.
