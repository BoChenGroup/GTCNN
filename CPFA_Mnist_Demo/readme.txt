=============================================================================================================
Python code for the paper "Convolutional Poisson Gamma Belief Network" published in ICML2019

=============================================================================================================
CPFA_Mnist_Demo folder contains 4 different training algorithms for CPFA:

CPFA_Mnist_Batch_Full_Matrix (Deal with the whole input matrix on CPU)

CPFA_Mnist_Batch_Sparse (Deal with the input matrix under sparse representation the on CPU)

CPFA_Mnist_Batch_Sparse_Parallel (Deal with the input matrix under sparse representation the on GPU)

CPFA_Mnist_Online_Sparse_Parallel (SGMCMC to handle with large-scale datasets on GPU)

=============================================================================================================

We provide this demo on Mnist dataset to illustrate the benefits of CPFA, which can directly handle with the matrix under sparse representation and parallelize the Gibbs sampler for training on GPU.


