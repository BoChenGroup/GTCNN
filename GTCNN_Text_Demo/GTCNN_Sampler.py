# 注意， cuda初始化中间不要加入cuda的操作,例如tensorflow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# cuda par
import pycuda.curandom as curandom
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
import torch
import numpy as np
import time
from Util import *

x = torch.cuda.FloatTensor(8)
cuda_generator = XORWOWRandomNumberGenerator()

mod = SourceModule("""

#include <stdio.h>

__device__ int cudarand(long long seed)
{
    if (seed == 0)
    {
        seed = 1;
    }
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

#include <stdio.h>

__global__ void Conv_Multi_Sampler(float* randomseed, int* para, int* row_index, int* col_index, int* n_index, float* value_index, float* W1_nk1, float* D1_k1, float* W1_nk1_Aug, float* D1_k1_Aug)
{   
    int K0         = para[0];
    int K1         = para[1];
    int K1_S1      = para[2];
    int K1_S2      = para[3];
    int K1_S3      = para[4];
    int K1_S4      = para[5];
    int word_total = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {
        int v1 = row_index[idx];                 // row_index
        int v2 = col_index[idx];                 // col_index
        int n  = n_index[idx];                   // file_index
        float value = value_index[idx];          
        int seed = randomseed[idx] * 2147483647;

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((v1 - K1_S3 + 1) > 0)
            word_k1_min = v1 - K1_S3 + 1;
        else
            word_k1_min = 0;

        if (v1 > K1_S1 -1)
            word_k1_max = K1_S1 -1;
        else
            word_k1_max = v1;

        int l_word_k1 = word_k1_max - word_k1_min + 1;

        // word_k2
        if ((v2 - K1_S4 + 1) > 0)
            word_k2_min = v2 - K1_S4 + 1;
        else
            word_k2_min = 0;

        if (v2 > K1_S2 -1)
            word_k2_max = K1_S2 -1;
        else
            word_k2_max = v2;

        int l_word_k2 = word_k2_max - word_k2_min + 1;

        // N*K0*K1_V1*K1_V2 => N*K1*K1_S1*K1_S2, K0*K1*K1_S3*K1_S4
        
        float MultRate_sum = 0;

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int word_k1 = word_k1_min + k;
                    int word_k2 = word_k2_min + j;
                    int temp_a = (n) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + word_k1 * K1_S2 + (word_k2);
                    int temp_b = (i) * K1_S3 * K1_S4 + (v1 - word_k1) * K1_S4 + (v2 - word_k2);

                    MultRate_sum = MultRate_sum + W1_nk1[temp_a] * D1_k1[temp_b];
                }
            }
        }

        if (MultRate_sum == 0) 
        {
            return;
        }

        for (int token = 0; token<value; token++)
        {
            float cumsum=0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * MultRate_sum;
            int flag=0;

            for (int i = 0; i < K1; i++)
            {
                for (int k = 0; k < (l_word_k1); k++)
                {
                    for (int j = 0; j < (l_word_k2); j++)
                    {
                        int word_k1 = word_k1_min + k;
                        int word_k2 = word_k2_min + j;
                        int temp_a = (n) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + word_k1 * K1_S2 + (word_k2);
                        int temp_b = (i) * K1_S3 * K1_S4 + (v1 - word_k1) * K1_S4 + (v2 - word_k2);

                        float prob = W1_nk1[temp_a] * D1_k1[temp_b];
                        cumsum += prob;
                        if (cumsum>=probrnd)
                        {
                             atomicAdd(&W1_nk1_Aug[temp_a], 1.0);
                             atomicAdd(&D1_k1_Aug[temp_b], 1.0);
                             flag = 1;
                             break;        
                        }
                    }

                    if (flag==1) break;
                }

                if (flag==1) break;
            }
        }

    }
}

__global__ void Crt_Conv_Multi_Sampler(float* randomseed, int* para, int* n_index, int* k1_index, int* row_index, int* col_index, float* value_index, float* D2_k2, float* W2_nk2, float* Phi_Theta, float* D2_k2_Aug, float* W2_nk2_Aug)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int K2 = para[4];
    int K2_S1 = para[5];
    int K2_S2 = para[6];
    int K2_S3 = para[7];
    int K2_S4 = para[8];
    int word_total = para[9];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {
        int n = n_index[idx];
        int k1 = k1_index[idx];
        int row = row_index[idx];
        int col = col_index[idx];
        float value = value_index[idx];
        
        // Crt: N*K1*K1_S1*K1_S2

        int seed = randomseed[idx] * 2147483647;
        int temp_a = n*K1*K1_S1*K1_S2 + k1*K1_S1*K1_S2 + row*K1_S1 + col;
        float sum = Phi_Theta[temp_a]; // Phi_Theta: N*K1*K1_S1*K1_S2
        int table = 0;
        int token = 0;

        if (value<0.5)
        {
            table = 0;
            return;
        }
        else
        {
            for ( token = 1, table = 1; token<value; token++)
            {
                seed = cudarand(seed);
                float probrnd = ((float)(seed) / 2147483647.0);
                if (probrnd <= sum / (sum + token))
                    table++;
            }
        }

        //W1_nk1: N*K1*K1_S1*K1_S2 => W2_nk2: N*K2*K2_S1*K2_S2  D2_k2: K1*K2*K2_S3*K2_S4

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((row - K2_S3 + 1) > 0)
            word_k1_min = row - K2_S3 + 1;
        else
            word_k1_min = 0;

        if (row > K2_S1 -1)
            word_k1_max = K2_S1 -1;
        else
            word_k1_max = row;

        int l_word_k1 = word_k1_max - word_k1_min + 1;

        // word_k2
        if ((col - K2_S4 + 1) > 0)
            word_k2_min = col - K2_S4 + 1;
        else
            word_k2_min = 0;

        if (col > K2_S2 -1)
            word_k2_max = K2_S2 -1;
        else
            word_k2_max = col;

        int l_word_k2 = word_k2_max - word_k2_min + 1;

        float MultRate_sum = 0;

        for (int i = 0; i < K2; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int word_k1 = word_k1_min + k;
                    int word_k2 = word_k2_min + j;
                    int temp_a = (n) * K2 * K2_S1 * K2_S2 + (i) * K2_S1 * K2_S2 + word_k1 * K2_S2 + (word_k2);
                    int temp_b = k1 * K2 * K2_S3 * K2_S4 +  (i) * K2_S3 * K2_S4 + (row - word_k1) * K2_S4 + (col - word_k2);
                    MultRate_sum = MultRate_sum + W2_nk2[temp_a] * D2_k2[temp_b];
                }
            }
        }

        if (MultRate_sum == 0) 
        {
            return;
        }


        for (int token = 0; token<table; token++)
        {
            float cumsum=0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * MultRate_sum;
            int flag=0;

            for (int i = 0; i < K2; i++)
            {
                for (int k = 0; k < (l_word_k1); k++)
                {
                    for (int j = 0; j < (l_word_k2); j++)
                    {
                        int word_k1 = word_k1_min + k;
                        int word_k2 = word_k2_min + j;
                        int temp_a = (n) * K2 * K2_S1 * K2_S2 + (i) * K2_S1 * K2_S2 + word_k1 * K2_S2 + (word_k2);
                        int temp_b = k1 * K2 * K2_S3 * K2_S4 +  (i) * K2_S3 * K2_S4 + (row - word_k1) * K2_S4 + (col - word_k2);

                        float prob = W2_nk2[temp_a] * D2_k2[temp_b];
                        cumsum += prob;
                        if (cumsum>=probrnd)
                        {
                             atomicAdd(&W2_nk2_Aug[temp_a], 1.0);
                             atomicAdd(&D2_k2_Aug[temp_b], 1.0);
                             flag = 1;
                             break;        
                        }
                    }

                    if (flag==1) break;
                }

                if (flag==1) break;
            }
        }

    }
}

__global__ void Crt_Multi_Sampler(float *randomseed, int* Para, int* Xt_to_t1, float* Phi_t1, float* Theta_t1, float* Xt1_VK, float* Xt1_KJ)
{

    const int V = Para[0];
    const int K = Para[1];
    const int J = Para[2];
    const int N = Para[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx] * 2147483647;
    
    if (idx < N && Xt_to_t1[idx] >= 0.5 )
    {
        float sum = 0.0;
        float cumsum = 0.0;
        int token, table;
        int v = idx / J;   // row first
        int j = idx % J;   // row first
        
        for (int k = 0; k<K; k++)
        {
            sum += Phi_t1[v*K + k] * Theta_t1[k*J + j]; // C index is different of Matlab
        }
        
        for (token = 1, table = 1; token<Xt_to_t1[idx]; token++)
        {
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0);
            if (probrnd <= sum / (sum + token))
                table++;
        }
        
        for (token = 0; token<table; token++)
        {
            int Embedding_K = K - 1;
            float sumprob = 0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * sum;

            for (int k = 0; k < K; k++)
            {
                cumsum = Phi_t1[v*K + k] * Theta_t1[k*J + j];
                if (sumprob + cumsum >= probrnd)
                {
                    Embedding_K = k;
                    break;
                }
                sumprob += cumsum;
            }
            
            atomicAdd(&Xt1_VK[v*K + Embedding_K], 1);
            atomicAdd(&Xt1_KJ[Embedding_K*J + j], 1);
        }
    }
}

__global__ void Sum_Pooling(int* para, float* M_nk, float* M_nk_pool, float* M_nk_w)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int word_total = para[4];
    int stride = para[5];
    int K1_S2_pool = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {   
        int row_index = idx / K1_S2_pool;
        int remain = idx - row_index*K1_S2_pool;
        
        int idx_stride = stride;
        if (remain == K1_S2_pool-1)
        {
            idx_stride = K1_S2 - (K1_S2_pool-1)*stride;
        }
        
        //printf("%d, %d, %d", row_index, remain, idx_stride);
        
        for (int i=0; i<idx_stride; i++)
        {
            int temp_a = row_index*K1_S2 + remain*stride + i;
            atomicAdd(&M_nk_pool[idx], M_nk[temp_a] + 0.0000001); 
        }
        
        for (int i=0; i<idx_stride; i++)
        {            
            int temp_a = row_index*K1_S2 + remain*stride + i;
            float rate = (M_nk[temp_a] + 0.0000001) / (M_nk_pool[idx]);
            atomicAdd(&M_nk_w[temp_a], rate); 
        }
    }
}

__global__ void Up_Pooling(int* para, float* M_nk, float* M_nk_pool, float* M_nk_w)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int word_total = para[4];
    int stride = para[5];
    int K1_S2_pool = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {   
        int row_index = idx / K1_S2_pool;
        int remain = idx - row_index*K1_S2_pool;
        
        int idx_stride = stride;
        if (remain == K1_S2_pool-1)
        {
            idx_stride = K1_S2 - (K1_S2_pool-1)*stride;
        }
        
        for (int i=0; i<idx_stride; i++)
        {
            int temp_a = row_index*K1_S2 + remain*stride + i;
            float rate = M_nk_pool[idx] * M_nk_w[temp_a];
            if (rate <= 0.0000001)
            {
                rate = 0.0000001;
            }
            
/*            if (M_nk_pool[idx] == 0)
            {
                printf("value error");
            }*/
            atomicAdd(&M_nk[temp_a], rate); 
        }
    }
}
 """)
print("kernel intial finish")

def Crt_Multirnd_Matrix(Xt_to_t1_t, Phi_t1, Theta_t1, dtype='dense'):

    if dtype=='dense':

        [K_t, J] = Xt_to_t1_t.shape
        K_t1 = Theta_t1.shape[0]
        N = K_t*J
        Para = np.array([K_t, K_t1, J, N], dtype=np.int32)


    Xt_to_t1_t = np.array(Xt_to_t1_t, dtype=np.int32, order='C')
    Xt_to_t1_t1 = np.zeros([K_t1, J], dtype=np.float32, order='C')
    WSZS_t1 = np.zeros([K_t, K_t1], dtype=np.float32, order='C')
    Phi_t1 = np.array(Phi_t1, dtype=np.float32, order='C')
    Theta_t1 = np.array(Theta_t1, dtype=np.float32, order='C')

    if N!=0:

        block_x = int(400)
        grid_x = int(np.floor(N / block_x) + 1)

        randomseed = np.random.rand(N)
        randomseed = np.array(randomseed, dtype=np.float32, order='C')

        func = mod.get_function('Crt_Multi_Sampler')
        func(drv.In(randomseed), drv.In(Para), drv.In(Xt_to_t1_t), drv.In(Phi_t1),
             drv.In(Theta_t1), drv.InOut(WSZS_t1), drv.InOut(Xt_to_t1_t1),
             grid=(grid_x, 1), block=(block_x, 1, 1))

    return Xt_to_t1_t1, WSZS_t1

def Conv_Multi_Sampler(row_index, col_index, n_index, value_index, D1_k1, W1_nk1):

    # check by chaojies
    [K1, K0, K1_S3, K1_S4] = D1_k1.shape  # K1*K0*K1_S3*K1_S4
    [N, K1, K1_S1, K1_S2] = W1_nk1.shape  # N*K1*K1_S1*K1_S2

    # 增广矩阵用于更新s12维度上增广
    X_rows = gpuarray.to_gpu(np.array(row_index, dtype=np.int32, order='C'))
    X_cols = gpuarray.to_gpu(np.array(col_index + 1, dtype=np.int32, order='C'))  # padding!!!
    X_file_index = gpuarray.to_gpu(np.array(n_index, dtype=np.int32, order='C'))
    X_value = gpuarray.to_gpu(np.array(value_index, dtype=np.float32, order='C'))
    word_total = int(X_rows.size)

    if word_total == 0:
        return np.zeros([N, K1, K1_S1, K1_S2]), np.zeros([K1, K0, K1_S3, K1_S4])
    else:
        W1_nk1 = gpuarray.to_gpu(np.array(W1_nk1, dtype=np.float32, order='C'))
        D1_k1 = gpuarray.to_gpu(np.array(np.swapaxes(D1_k1, 0, 1), dtype=np.float32, order='C'))  # K1*K0*K1_S3*K1_S4

        W1_nk1_Aug = gpuarray.zeros(W1_nk1.shape, dtype=np.float32, order='C')
        D1_k1_Aug = gpuarray.zeros(D1_k1.shape, dtype=np.float32, order='C')

        randomseed = cuda_generator.gen_uniform([word_total], dtype=np.float32)

        # 转化为GPU的输入形式
        fuc = mod.get_function("Conv_Multi_Sampler")
        Batch_Para = gpuarray.to_gpu(np.array([K0, K1, K1_S1, K1_S2, K1_S3, K1_S4, word_total], dtype=np.int32, order='C'))

        block_x = int(500)
        grid_x = int(np.floor(word_total / block_x) + 1)

        fuc(randomseed, Batch_Para, X_rows, X_cols, X_file_index, X_value, W1_nk1, D1_k1,
            W1_nk1_Aug, D1_k1_Aug, grid=(grid_x, 1, 1), block=(block_x, 1, 1))  # 一般最多512个并行线程

        return W1_nk1_Aug.get(), np.swapaxes(D1_k1_Aug.get(), 0, 1)


def Crt_Conv_Multi_Sampler(M2_nk1, D2_k2, W2_nk2, Phi_Theta_2):

    # check by chaojie
    [N, K1, K1_S1, K1_S2] = M2_nk1.shape
    K1_S2 += 2  # padding!!!
    [K2, K1, K2_S3, K2_S4] = D2_k2.shape
    [N, K2, K2_S1, K2_S2] = W2_nk2.shape

    [n_index, k1_index, row_index, col_index] = np.where(M2_nk1 > 0.5)
    X_n = gpuarray.to_gpu(np.array(n_index, dtype=np.int32, order='C'))
    X_k1 = gpuarray.to_gpu(np.array(k1_index, dtype=np.int32, order='C'))
    X_row = gpuarray.to_gpu(np.array(row_index, dtype=np.int32, order='C'))
    X_col = gpuarray.to_gpu(np.array(col_index + 1, dtype=np.int32, order='C'))  # padding!!!
    X_value = gpuarray.to_gpu(np.array(M2_nk1[(n_index, k1_index, row_index, col_index)], dtype=np.float32, order='C'))
    word_total = int(X_row.size)

    # print(word_total)
    if word_total == 0:
        return np.zeros([N, K2, K2_S1, K2_S2]), np.zeros([K2, K1, K2_S3, K2_S4])
    else:
        Phi_Theta_2 = gpuarray.to_gpu(np.array(Phi_Theta_2, dtype=np.float32, order='C'))  # N*K1*K1_S1*K1_S2
        W2_nk2 = gpuarray.to_gpu(np.array(W2_nk2, dtype=np.float32, order='C'))  # N*K2*K2_S1*K2_S2
        D2_k2 = gpuarray.to_gpu(np.array(np.swapaxes(D2_k2, 0, 1), dtype=np.float32, order='C'))  # K1*K2*K2_S3*K2_S4

        W2_nk2_Aug = gpuarray.zeros(W2_nk2.shape, dtype=np.float32, order='C')  # N*K2*K2_S1*K2_S2
        D2_k2_Aug = gpuarray.zeros(D2_k2.shape, dtype=np.float32, order='C')  # K1*K2*K2_S3*K2_S4

        fuc = mod.get_function("Crt_Conv_Multi_Sampler")
        Batch_Para = gpuarray.to_gpu(np.array([N, K1, K1_S1, K1_S2, K2, K2_S1, K2_S2, K2_S3, K2_S4, word_total], dtype=np.int32, order='C'))

        randomseed = cuda_generator.gen_uniform([word_total], dtype=np.float32)

        block_x = int(500)
        grid_x = int(np.floor(word_total / block_x) + 1)

        fuc(randomseed, Batch_Para, X_n, X_k1, X_row, X_col, X_value, D2_k2, W2_nk2, Phi_Theta_2,
            D2_k2_Aug, W2_nk2_Aug, grid=(grid_x, 1, 1), block=(block_x, 1, 1))  # 一般最多512个并行线程


        return W2_nk2_Aug.get(), np.swapaxes(D2_k2_Aug.get(), 0, 1)


def Sum_Pooling(M2_nk1, stride):

    [N, K1, K1_S1, K1_S2] = M2_nk1.shape
    K1_S2_pool = np.int32(np.ceil(K1_S2 / stride))

    M2_nk1 = gpuarray.to_gpu(np.array(M2_nk1, dtype=np.float32, order='C'))
    M2_nk1_w = gpuarray.zeros([N, K1, K1_S1, K1_S2], dtype=np.float32, order='C')
    M2_nk1_p = gpuarray.zeros([N, K1, K1_S1, K1_S2_pool], dtype=np.float32, order='C')

    word_total = N * K1 * K1_S1 * K1_S2_pool
    Batch_Para = gpuarray.to_gpu(np.array([N, K1, K1_S1, K1_S2, word_total, stride, K1_S2_pool], dtype=np.int32))

    block_x = int(500)
    grid_x = int(np.floor(word_total / block_x) + 1)

    # start_time = time.time()
    fuc = mod.get_function("Sum_Pooling")
    fuc(Batch_Para, M2_nk1, M2_nk1_p, M2_nk1_w, grid=(grid_x, 1, 1), block=(block_x, 1, 1))  # 一般最多512个并行线程
    # end_time = time.time()
    # print(end_time - start_time)

    return M2_nk1_p.get(), M2_nk1_w.get()


def Up_Pooling(M2_nk1_p, M2_nk1_w, stride):

    [N, K1, K1_S1, K1_S2] = M2_nk1_w.shape
    [N, K1, K1_S1, K1_S2_pool] = M2_nk1_p.shape

    M2_nk1_p = gpuarray.to_gpu(np.array(M2_nk1_p, dtype=np.float32, order='C'))
    M2_nk1_w = gpuarray.to_gpu(np.array(M2_nk1_w, dtype=np.float32, order='C'))
    M2_nk1 = gpuarray.zeros([N, K1, K1_S1, K1_S2], dtype=np.float32, order='C')


    word_total = N * K1 * K1_S1 * K1_S2_pool
    Batch_Para = gpuarray.to_gpu(np.array([N, K1, K1_S1, K1_S2, word_total, stride, K1_S2_pool], dtype=np.int32))

    block_x = int(500)
    grid_x = int(np.floor(word_total / block_x) + 1)

    fuc = mod.get_function("Up_Pooling")
    fuc(Batch_Para, M2_nk1, M2_nk1_p, M2_nk1_w, grid=(grid_x, 1, 1), block=(block_x, 1, 1))  # 一般最多512个并行线程

    return M2_nk1.get()


def ProjSimplexSpecial(Phi_tmp, Phi_old, epsilon):
    Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
    if np.where(Phinew[:, :] <= 0)[0].size > 0:
        Phinew = np.maximum(epsilon, Phinew)
        Phinew = Phinew / np.maximum(real_min, Phinew.sum(0))
    return Phinew

