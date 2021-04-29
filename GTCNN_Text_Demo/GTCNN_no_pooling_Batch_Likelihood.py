import numpy as np
from Util import *
import scipy.io as sio
import time
from scipy.special import gamma
import matplotlib.pyplot as plt
from GTCNN_Sampler import Crt_Conv_Multi_Sampler, Conv_Multi_Sampler
import torch.nn.functional as F
import torch
import _pickle as cPickle
import cupy

np.random.RandomState(1)
realmin = 2.2e-10

def log_max(x):
    return np.log(np.maximum(x, realmin))


# =============Load Data============== #
DATA = cPickle.load(open("./MR.pkl", "rb"), encoding='iso-8859-1')

data_vab_list = DATA['Vocabulary']
data_vab_count_list = DATA['Vab_count']
data_vab_length = DATA['Vab_Size']
data_label = DATA['Label']

data_train_list = DATA['Train_Origin']
data_train_label = np.array(DATA['Train_Label'])
data_train_split = DATA['Train_Word_Split']
data_train_list_index = DATA['Train_Word2Index']
print('load data')


# =============Preprocess============== #
Batch_Sparse = Empty()
Batch_Sparse.doc = data_train_list_index[:200]
Batch_Sparse.doc_label = data_train_label[:200]

count_value = 25
for doc_index, doc in enumerate(Batch_Sparse.doc):  # 所有样本遍历

    doc_len = len(doc)
    doc_word_index = np.reshape(doc, [doc_len]).astype(np.int32)

    if doc_index == 0:
        Batch_Sparse.doc_len = np.array([doc_len])
        Batch_Sparse.rows = doc_word_index
        Batch_Sparse.cols = np.arange(doc_len)
        Batch_Sparse.file_index = np.ones_like(doc_word_index) * doc_index
        Batch_Sparse.values = np.ones_like(doc_word_index) * count_value
        Batch_Sparse.labels = np.array([Batch_Sparse.doc_label[doc_index]])

    else:
        Batch_Sparse.doc_len = np.concatenate((Batch_Sparse.doc_len, np.array([doc_len])), axis=0)
        Batch_Sparse.rows = np.concatenate((Batch_Sparse.rows, doc_word_index), axis=0)
        Batch_Sparse.cols = np.concatenate((Batch_Sparse.cols, np.arange(doc_len)), axis=0)
        Batch_Sparse.file_index = np.concatenate((Batch_Sparse.file_index, np.ones_like(doc_word_index) * doc_index), axis=0)
        Batch_Sparse.values = np.concatenate((Batch_Sparse.values, np.ones_like(doc_word_index) * count_value), axis=0)
        Batch_Sparse.labels = np.concatenate((Batch_Sparse.labels, np.array([Batch_Sparse.doc_label[doc_index]])), axis=0)

print('preprocess finished')

# =============Settings============== #

Setting = Empty()
Setting.N_train = len(Batch_Sparse.doc)

Setting.K = [200]  # convolutional layer settings
K_Aug = [1] + Setting.K
Setting.T = len(Setting.K)

Setting.K_V1 = []
Setting.K_V2 = []
Setting.K_S1 = []
Setting.K_S2 = []
Setting.K_S3 = []
Setting.K_S4 = []

Likelihood_list = []
Timecost_list = []

for t in range(Setting.T):
    if t == 0:
        Setting.K_V1.append(data_vab_length)
        Setting.K_V2.append(np.max(Batch_Sparse.doc_len) + 2)  # padding
        Setting.K_S3.append(data_vab_length)
        Setting.K_S4.append(3)
        Setting.K_S1.append(Setting.K_V1[t] + 1 - Setting.K_S3[t])
        Setting.K_S2.append(Setting.K_V2[t] + 1 - Setting.K_S4[t])
    else:
        Setting.K_V1.append(Setting.K_S1[t-1])
        Setting.K_V2.append(Setting.K_S2[t-1] + 2)  # padding
        Setting.K_S3.append(Setting.K_S1[t-1])
        Setting.K_S4.append(3)
        Setting.K_S1.append(Setting.K_V1[t] + 1 - Setting.K_S3[t])
        Setting.K_S2.append(Setting.K_V2[t] + 1 - Setting.K_S4[t])

Setting.Iter = 1000
Setting.Burnin = 0.75 * Setting.Iter
Setting.Collection = Setting.Iter - Setting.Burnin

# =============SuperParamsSetting============== #
SuperParams = Empty()
SuperParams.gamma0 = 0.1
SuperParams.c0 = 0.1
SuperParams.a0 = 0.1  # p
SuperParams.b0 = 0.1
SuperParams.e0 = 0.1  # c
SuperParams.f0 = 0.1
SuperParams.eta = 0.05  # Phi

# =============Gibbs Sampler============== #
Params = Empty()
Params.D_k = [0] * Setting.T
Params.W_nk = [0] * Setting.T
Params.W_nk_collect = [0] * Setting.T
Params.c_n = [0] * (Setting.T + 1)
Params.p_n = [0] * (Setting.T + 1)

for t in range(Setting.T):

    D_k = np.random.rand(Setting.K[t], K_Aug[t], Setting.K_S3[t], Setting.K_S4[t])
    Params.D_k[t] = D_k / np.sum(np.sum(np.sum(D_k, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True)
    Params.W_nk[t] = np.random.rand(Setting.N_train, Setting.K[t], Setting.K_S1[t], Setting.K_S2[t])
    Params.W_nk_collect[t] = np.zeros([Setting.N_train, Setting.K[t]])

    if t == 0:
        Params.c_n[t] = 1 * np.ones([Setting.N_train])
        Params.p_n[t] = (1 - 1 / np.exp(1)) * np.ones([Setting.N_train])

    else:
        Params.c_n[t] = 1 * np.ones([Setting.N_train])
        tmp = - log_max(1 - Params.p_n[t])
        Params.p_n[t] = (tmp / (tmp + Params.c_n[t]))

Params.c_n[Setting.T] = 1 * np.ones([Setting.N_train])
tmp = -log_max(1 - Params.p_n[Setting.T - 1])
Params.p_n[Setting.T] = tmp / (tmp + Params.c_n[Setting.T])

Params.Gamma = 0.1 * np.ones([Setting.K[Setting.T - 1], Setting.K_S1[Setting.T - 1], Setting.K_S2[Setting.T - 1]])

for it in range(Setting.Iter):

    D_k_Aug = [0] * Setting.T
    W_nk_Aug = [0] * Setting.T
    Phi_Theta = [0] * (Setting.T - 1)

    start_time = time.time()

    for t in range(Setting.T):
        if t == 0:
            # N*K1*K1_S1*K1_S4, K1*K0*K1_S3*K1_S4
            W_nk_Aug[t], D_k_Aug[t] = Conv_Multi_Sampler(Batch_Sparse.rows, Batch_Sparse.cols, Batch_Sparse.file_index, Batch_Sparse.values, Params.D_k[t], Params.W_nk[t])

        else:
            device = "cuda:0"
            Phi_tensor = torch.from_numpy(Params.D_k[t]).to(torch.float64).to(device)  # K2*K1*K2_S3*K2_S4
            Theta_tensor = torch.from_numpy(Params.W_nk[t]).to(torch.float64).to(device)  # N*K2*K2_S1*K2_S2
            Phi_Theta_tensor = F.conv_transpose2d(Theta_tensor, Phi_tensor, bias=None, stride=1, padding=0,
                                                  output_padding=0, groups=1, dilation=1)  # N*K1*K1_S1*K1_S2
            Phi_Theta[t-1] = Phi_Theta_tensor.to(torch.float32).cpu().numpy()  # N*K1*K2_V1*K2_V2

            W_nk_Aug[t], D_k_Aug[t] = Crt_Conv_Multi_Sampler(W_nk_Aug[t-1], Params.D_k[t], Params.W_nk[t], Phi_Theta[t-1])

    # ==========================更新参数========================== #
    # 更新Phi
    for t in range(Setting.T):
        Params.D_k[t] = (D_k_Aug[t] + SuperParams.eta) / (realmin + np.sum(np.sum(np.sum(D_k_Aug[t] + SuperParams.eta, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True))

    # update c_j,p_j
    # if Setting.T > 1:
    #     p2_n_aug = np.sum(np.sum(np.sum(W_nk_Aug[0], axis=3), axis=2), axis=1)
    #     Phi_Theta_flat = np.sum(np.sum(np.sum(Phi_Theta[0], axis=3), axis=2), axis=1)  # conv(Phi_2, Theta_2)
    #     Params.p_n[1] = np.random.beta(p2_n_aug + SuperParams.a0, Phi_Theta_flat + SuperParams.b0)
    #
    # else:
    #     p2_n_aug = np.sum(np.sum(np.sum(W_nk_Aug[0], axis=3), axis=2), axis=1)
    #     Params.p_n[1] = np.random.beta(p2_n_aug + SuperParams.a0, np.sum(Params.Gamma) + SuperParams.b0)
    # Params.c_n[1] = (1 - Params.p_n[1]) / Params.p_n[1]

    for t in [i for i in range(Setting.T + 1) if i > 0]:  # only T>=2 works  ==> for t = 3:T+1
        if t == Setting.T:
            Params.c_n[t] = cupy.random.gamma(shape=np.sum(Params.Gamma) + SuperParams.e0, scale=1).get()
            Params.c_n[t] = Params.c_n[t]/(SuperParams.f0 + np.sum(np.sum(np.sum(Params.W_nk[t-1], axis=3), axis=2), axis=1))

        else:
            Phi_Theta_flat = np.sum(np.sum(np.sum(Phi_Theta[t-1], axis=3), axis=2), axis=1)
            Params.c_n[t] = cupy.random.gamma(shape=Phi_Theta_flat + SuperParams.e0, scale=1).get()
            Params.c_n[t] = Params.c_n[t] / (np.sum(np.sum(np.sum(Params.W_nk[t-1], axis=3), axis=2), axis=1) + SuperParams.f0)

    for t in [i for i in range(Setting.T) if i > 0]:  # only T>=2 works  ==> for t = 2:T
        tmp = -log_max(1 - Params.p_n[t-1])
        Params.p_n[t] = (tmp / (tmp + Params.c_n[t]))

    # 更新Theta
    for t in range(Setting.T - 1, -1, -1):  ## for t = T:-1 :1
        if t == Setting.T - 1:
            # 更新最高层的Theta
            # N*K_t-1*K_T*K_S1*K_S2
            Params.W_nk[t] = cupy.random.gamma(W_nk_Aug[t] + Params.Gamma[np.newaxis, :], scale=1).get()  # N*K_T*K_S1*K_S2
            Params.W_nk[t] = Params.W_nk[t] / (- log_max(1 - Params.p_n[t]) + Params.c_n[t+1])[:, np.newaxis, np.newaxis, np.newaxis]

        else:
            # 更新底层的Theta
            Params.W_nk[t] = cupy.random.gamma(W_nk_Aug[t] + Phi_Theta[t][:, :, :, 1:-1], scale=1).get()  # N*K_T*K_S1*K_S2
            Params.W_nk[t] = Params.W_nk[t] / (- log_max(1 - Params.p_n[t]) + Params.c_n[t+1])[:, np.newaxis, np.newaxis, np.newaxis]

        if it >= Setting.Burnin:
            Params.W_nk_collect[t] += (np.sum(np.sum(Params.W_nk[t]/Setting.Collection, axis=3), axis=2) / np.reshape(Batch_Sparse.doc_len, [Batch_Sparse.doc_len.shape[0], 1]))

    end_time = time.time()

    # # Likelyhood
    if (it + 1) % 1 == 0:
        device = "cuda:0"
        Phi_tensor = torch.from_numpy(Params.D_k[0]).to(torch.float64).to(device)  # K2*K1*K2_S3*K2_S4
        Theta_tensor = torch.from_numpy(Params.W_nk[0]).to(torch.float64).to(device)  # N*K2*K2_S1*K2_S2
        Phi_Theta_tensor = F.conv_transpose2d(Theta_tensor, Phi_tensor, bias=None, stride=1, padding=0,
                                              output_padding=0, groups=1, dilation=1)  # N*K1*K1_S1*K1_S2
        Phi_Theta = Phi_Theta_tensor.to(torch.float32).cpu().numpy()  # N*K1*K2_V1*K2_V2

        X_orgin = np.zeros([Setting.N_train, 1, Setting.K_V1[0], Setting.K_V2[0]], dtype=np.float32)
        X_orgin[(Batch_Sparse.file_index, 0, Batch_Sparse.rows, Batch_Sparse.cols + 1)] = count_value
        likelihood = np.sum(X_orgin/25 * log_max(Phi_Theta/25) - Phi_Theta/25 - log_max(gamma(X_orgin/25 + 1))) / Setting.N_train
        likelihood_2 = np.sum(X_orgin * log_max(Phi_Theta) - Phi_Theta - log_max(gamma(X_orgin + 1))) / Setting.N_train

        X_orgin_mask = np.zeros([Setting.N_train, 1, Setting.K_V1[0], Setting.K_V2[0]], dtype=np.float32)
        X_orgin_mask[(Batch_Sparse.file_index, 0, Batch_Sparse.rows, Batch_Sparse.cols + 1)] = 1
        X_orgin_sum = np.sum(np.sum(np.sum(X_orgin/count_value, axis=3, keepdims=True), axis=2, keepdims=True), axis=1,
                             keepdims=True)
        ppl = np.exp(-np.sum(X_orgin_mask * log_max(Phi_Theta/count_value / X_orgin_sum)) / np.sum(X_orgin_sum))

        print("epoch {:}: ppl:{:8f} likelihood: {:8f} takes {:8f} seconds".format(it, ppl, likelihood, end_time-start_time))

        Likelihood_list.append(likelihood)
        Timecost_list.append(end_time-start_time)

    print("epoch " + str(it) + " takes " + str(end_time - start_time) + " seconds")

print("train phase finished")

file_name = ''.join([str(k) for k in Setting.K]) + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

sio.savemat(file_name + '_PGCN_MR.mat', {'Likelihood': Likelihood_list,
                                 'Timecost': Timecost_list})


# # =============Load Data============== #
# data_test_list = DATA['Test_Origin']
# data_test_label = np.array(DATA['Test_Label'])
# data_test_split = DATA['Test_Word_Split']
# data_test_list_index = DATA['Test_Word2Index']
#
#
# # =============Preprocess============== #
# Batch_Sparse_te = Empty()
# Batch_Sparse_te.doc = data_test_list_index
# Batch_Sparse_te.doc_label = data_test_label
#
# count_value = 25
# for doc_index, doc in enumerate(Batch_Sparse_te.doc):  # 所有样本遍历
#
#     doc_len = len(doc)
#     doc_word_index = np.reshape(doc, [doc_len]).astype(np.int32)
#
#     if doc_index == 0:
#
#         Batch_Sparse_te.doc_len = np.array([doc_len])
#         Batch_Sparse_te.rows = doc_word_index
#         Batch_Sparse_te.cols = np.arange(doc_len)
#         Batch_Sparse_te.file_index = np.ones_like(doc_word_index) * doc_index
#         Batch_Sparse_te.values = np.ones_like(doc_word_index) * count_value
#         Batch_Sparse_te.labels = np.array([Batch_Sparse_te.doc_label[doc_index]])
#     else:
#
#         Batch_Sparse_te.doc_len = np.concatenate((Batch_Sparse_te.doc_len, np.array([doc_len])), axis=0)
#         Batch_Sparse_te.rows = np.concatenate((Batch_Sparse_te.rows, doc_word_index), axis=0)
#         Batch_Sparse_te.cols = np.concatenate((Batch_Sparse_te.cols, np.arange(doc_len)), axis=0)
#         Batch_Sparse_te.file_index = np.concatenate((Batch_Sparse_te.file_index, np.ones_like(doc_word_index) * doc_index), axis=0)
#         Batch_Sparse_te.values = np.concatenate((Batch_Sparse_te.values, np.ones_like(doc_word_index) * count_value), axis=0)
#         Batch_Sparse_te.labels = np.concatenate((Batch_Sparse_te.labels, np.array([Batch_Sparse_te.doc_label[doc_index]])), axis=0)
#
# print('preprocess finished')
#
# # =============Setting============== #
# Setting.N_test = len(Batch_Sparse_te.doc)
# Setting.K_V2 = []
# Setting.K_S1 = []
# Setting.K_S2 = []
#
# for t in range(Setting.T):
#     if t == 0:
#         Setting.K_V2.append(np.max(Batch_Sparse_te.doc_len) + 2)  # padding
#         Setting.K_S1.append(Setting.K_V1[t] + 1 - Setting.K_S3[t])
#         Setting.K_S2.append(Setting.K_V2[t] + 1 - Setting.K_S4[t])
#     else:
#         Setting.K_V2.append(Setting.K_S2[t-1] + 2)  # padding
#         Setting.K_S1.append(Setting.K_V1[t] + 1 - Setting.K_S3[t])
#         Setting.K_S2.append(Setting.K_V2[t] + 1 - Setting.K_S4[t])
#
# # =============Gibbs Sampler============== #
# Params.W_nk_te = [0] * Setting.T
# Params.W_nk_te_collect = [0] * Setting.T
# Params.c_n_te = [0] * (Setting.T + 1)
# Params.p_n_te = [0] * (Setting.T + 1)
#
# Params.Gamma = 0.1 * np.ones([Setting.K[Setting.T - 1], Setting.K_S1[Setting.T - 1], Setting.K_S2[Setting.T - 1]])
#
# for t in range(Setting.T):
#
#     Params.W_nk_te[t] = np.random.rand(Setting.N_test, Setting.K[t], Setting.K_S1[t], Setting.K_S2[t])
#     Params.W_nk_te_collect[t] = np.zeros([Setting.N_test, Setting.K[t]])
#
#     if t == 0:
#         Params.c_n_te[t] = 1 * np.ones([Setting.N_test])
#         Params.p_n_te[t] = (1 - 1 / np.exp(1)) * np.ones([Setting.N_test])
#
#     else:
#         Params.c_n_te[t] = 1 * np.ones([Setting.N_test])
#         tmp = - log_max(1 - Params.p_n_te[t])
#         Params.p_n_te[t] = (tmp / (tmp + Params.c_n_te[t]))
#
# Params.c_n_te[Setting.T] = 1 * np.ones([Setting.N_test])
# tmp = -log_max(1 - Params.p_n_te[Setting.T - 1])
# Params.p_n_te[Setting.T] = tmp / (tmp + Params.c_n_te[Setting.T])
#
# for it in range(Setting.Iter):
#
#     D_k_Aug = [0] * Setting.T
#     W_nk_Aug = [0] * Setting.T
#     Phi_Theta = [0] * (Setting.T - 1)
#
#     start_time = time.time()
#
#     for t in range(Setting.T):
#         if t == 0:
#             W_nk_Aug[t], D_k_Aug[t] = Conv_Multi_Sampler(Batch_Sparse_te.rows, Batch_Sparse_te.cols, Batch_Sparse_te.file_index, Batch_Sparse_te.values, Params.D_k[t], Params.W_nk_te[t])
#
#         else:
#             device = "cuda:0"
#             Phi_tensor = torch.from_numpy(Params.D_k[t]).to(torch.float64).to(device)  # K2*K1*K2_S3*K2_S4
#             Theta_tensor = torch.from_numpy(Params.W_nk_te[t]).to(torch.float64).to(device)  # N*K2*K2_S1*K2_S2
#             Phi_Theta_tensor = F.conv_transpose2d(Theta_tensor, Phi_tensor, bias=None, stride=1, padding=0,
#                                                   output_padding=0, groups=1, dilation=1)  # N*K1*K1_S1*K1_S2
#             Phi_Theta[t-1] = Phi_Theta_tensor.to(torch.float32).cpu().numpy()  # N*K1*K2_V1*K2_V2
#
#             W_nk_Aug[t], D_k_Aug[t] = Crt_Conv_Multi_Sampler(W_nk_Aug[t-1], Params.D_k[t], Params.W_nk_te[t], Phi_Theta[t-1])
#
#
#     # ==========================更新参数========================== #
#     # update c_j,p_j
#     if Setting.T > 1:
#         p2_n_aug = np.sum(np.sum(np.sum(W_nk_Aug[0], axis=3), axis=2), axis=1)
#         Phi_Theta_flat = np.sum(np.sum(np.sum(Phi_Theta[0], axis=3), axis=2), axis=1)  # conv(Phi_2, Theta_2)
#         Params.p_n_te[1] = np.random.beta(p2_n_aug + SuperParams.a0, Phi_Theta_flat + SuperParams.b0)
#
#     else:
#         p2_n_aug = np.sum(np.sum(np.sum(W_nk_Aug[0], axis=3), axis=2), axis=1)
#         Params.p_n_te[1] = np.random.beta(p2_n_aug + SuperParams.a0, np.sum(Params.Gamma) + SuperParams.b0)
#     Params.c_n_te[1] = (1 - Params.p_n_te[1]) / Params.p_n_te[1]
#
#     for t in [i for i in range(Setting.T + 1) if i > 1]:  # only T>=2 works  ==> for t = 3:T+1
#         if t == Setting.T:
#             Params.c_n_te[t] = dsg.gamma(shape=np.sum(Params.Gamma) + SuperParams.e0, scale=1)
#             Params.c_n_te[t] = Params.c_n_te[t]/(SuperParams.f0 + np.sum(np.sum(np.sum(Params.W_nk_te[t-1], axis=3), axis=2), axis=1))
#
#         else:
#             Phi_Theta_flat = np.sum(np.sum(np.sum(Phi_Theta[t-1], axis=3), axis=2), axis=1)
#             Params.c_n_te[t] = dsg.gamma(shape=Phi_Theta_flat + SuperParams.e0, scale=1)
#             Params.c_n_te[t] = Params.c_n_te[t] / (np.sum(np.sum(np.sum(Params.W_nk_te[t-1], axis=3), axis=2), axis=1) + SuperParams.f0)
#
#     for t in [i for i in range(Setting.T) if i > 1]:  # only T>=2 works  ==> for t = 2:T
#         tmp = -log_max(1 - Params.p_n_te[t-1])
#         Params.p_n_te[t] = (tmp / (tmp + Params.c_n_te[t]))
#
#     # 更新Theta
#     for t in range(Setting.T - 1, -1, -1):  ## for t = T:-1 :1
#         if t == Setting.T - 1:
#             # 更新最高层的Theta
#             # N*K_t-1*K_T*K_S1*K_S2
#             Params.W_nk_te[t] = dsg.gamma(W_nk_Aug[t] + Params.Gamma[np.newaxis, :], scale=1)  # N*K_T*K_S1*K_S2
#             Params.W_nk_te[t] = Params.W_nk_te[t] / (- log_max(1 - Params.p_n_te[t]) + Params.c_n_te[t+1])[:, np.newaxis, np.newaxis, np.newaxis]
#
#         else:
#             # 更新底层的Theta
#             Params.W_nk_te[t] = dsg.gamma(W_nk_Aug[t] + Phi_Theta[t][:, :, :, 1:-1], scale=1)  # N*K_T*K_S1*K_S2
#             Params.W_nk_te[t] = Params.W_nk_te[t] / (- log_max(1 - Params.p_n_te[t]) + Params.c_n_te[t+1])[:, np.newaxis, np.newaxis, np.newaxis]
#
#         if it >= Setting.Burnin:
#             Params.W_nk_te_collect[t] += (np.sum(np.sum(Params.W_nk_te[t]/Setting.Collection, axis=3), axis=2) / np.reshape(Batch_Sparse_te.doc_len, [Batch_Sparse_te.doc_len.shape[0], 1]))
#
#     end_time = time.time()
#
#     # Likelyhood
#     print("epoch " + str(it) + " takes " + str(end_time - start_time) + " seconds")
#
# print("test phase finished")
#
# from sklearn import svm
#
# clf = svm.SVC(kernel='rbf')                             # class
# W_train = Params.W_nk_collect[0] #np.sum(np.sum(Params['W_nk_collect'][0]/Setting['Collection'], axis=3), axis=2)
# W_test = Params.W_nk_te_collect[0]  #np.sum(np.sum(Params['W_nk_te_collect'][0]/Setting['Collection'], axis=3), axis=2)
# clf.fit(W_train, Batch_Sparse.labels)            # training the svc model
#
# print(clf.score(W_train, Batch_Sparse.labels))   # training the svc model
# print(clf.score(W_test,  Batch_Sparse_te.labels))   # training the svc model
