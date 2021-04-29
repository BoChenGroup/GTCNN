# GTCNN

=====================
Python code for the paper "Generative Text Convolutional Neural Network for Hierarchial Document Representation Learning"

=====================
CPFA_Mnist_Demo folder contains 4 different training algorithms for CPFA:

CPFA_Mnist_Batch_Full_Matrix (Deal with the whole input matrix on CPU)

CPFA_Mnist_Batch_Sparse (Deal with the input matrix under sparse representation the on CPU)

CPFA_Mnist_Batch_Sparse_Parallel (Deal with the input matrix under sparse representation the on GPU)

CPFA_Mnist_Online_Sparse_Parallel (SGMCMC to handle with large-scale datasets on GPU)

We provide this demo on Mnist dataset to illustrate the benefits of CPFA, which can directly handle with the matrix under sparse representation and parallelize the Gibbs sampler for training on GPU.

=====================
GTCNN_Text_Demo folder contains GTCNN with/without document-level pooling

GTCNN_no_pooling_Batch_Likelihood (GTCNN without document-level pooling for plotting point-likelyhood)

GTCNN_no_pooling_Batch_Classication (GTCNN without document-level pooling for document classification)

GTCNN_with_pooling_Batch_Likelihood (GTCNN with document-level pooling for plotting point-likelyhood)

GTCNN_with_pooling_Batch_Classication (GTCNN with document-level pooling for document classification)


=====================
The code for "Convolutional Poisson Gamma Belief Network" has been released in https://github.com/BoChenGroup/CPGBN




