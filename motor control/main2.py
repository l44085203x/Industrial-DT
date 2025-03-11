# -*- coding: utf-8 -*-
import argparse
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import time
import random
import matplotlib.pyplot as plt
from reservoir_computing import *
from reservoir_module import *
from reservior_computing_based_link_prediction import generate_A, A
import xlwt
import xlrd
from Results import save_result, Loss_cal, save_results, save_complexity


def load_dataset(path, nWindow, nPredict, nInterval):
    workbook = xlrd.open_workbook(path)
    sheet1 = workbook.sheet_by_name('U')
    sheet2 = workbook.sheet_by_name('Delta_w')
    sheet3 = workbook.sheet_by_name('I')
    sheet4 = workbook.sheet_by_name('T_e')
    sheet5 = workbook.sheet_by_name('w')
    # sheet6 = workbook.sheet_by_name('W')

    nTime = sheet1.nrows - 1
    nEpoch = sheet1.ncols

    data = []
    nSmooth = 4
    for i in range(nEpoch):
        U = np.array(sheet1.col_values(i, 1, nTime + 1)) / 10 
        Delta_w = np.array(sheet2.col_values(i, 1, nTime + 1)) / 10000
        I = np.array(sheet3.col_values(i, 1, nTime + 1)) / 1000
        T_e = np.array(sheet4.col_values(i, 1, nTime + 1)) / 100
        w = np.array(sheet5.col_values(i, 1, nTime + 1))
        # W = np.array(sheet6.col_values(i, 1, nTime+1))/1000

        U = np.convolve(U, np.ones(nSmooth)) / nSmooth
        Delta_w = np.convolve(Delta_w, np.ones(nSmooth)) / nSmooth
        I = np.convolve(I, np.ones(nSmooth)) / nSmooth
        T_e = np.convolve(T_e, np.ones(nSmooth)) / nSmooth
        w = np.convolve(w, np.ones(nSmooth)) / nSmooth
        # W = np.convolve(W, np.ones(nSmooth)) / nSmooth

        data.append([U, Delta_w, I, T_e, w])
    data = np.array(data)

    train_Partial = 0.8
    data = data.transpose((0, 2, 1))
    nFeature = data.shape[2]
    data_train = data[0:int(nEpoch * train_Partial), :, :]
    data_test = data[int(nEpoch * train_Partial):, :, :]

    data_train = data_train.reshape(-1, nFeature)

    x = []
    y = []
    for epoch in range(data_test.shape[0]):
        for i in range(0, nTime - nWindow - nPredict + 1, nInterval):
            head = i
            tail = i + nWindow
            x.append(data_test[epoch, head:tail])
            y.append(data_test[epoch, tail:tail + nPredict])
    x = np.array(x)
    y = np.array(y)

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test_x = torch.tensor(x, dtype=torch.float32)
    data_test_y = torch.tensor(y, dtype=torch.float32)
    return data_train, data_test_x, data_test_y


def load_dataset2(path, nWindow, nPredict, nInterval, nc, train_Partial):
    workbook = xlrd.open_workbook(path)
    sheet1 = workbook.sheet_by_name('U')
    sheet2 = workbook.sheet_by_name('Delta_w')
    sheet3 = workbook.sheet_by_name('I')
    sheet4 = workbook.sheet_by_name('T_e')
    sheet5 = workbook.sheet_by_name('w')
    # sheet6 = workbook.sheet_by_name('W')

    nTime = sheet1.nrows - 1 - nc
    nEpoch = sheet1.ncols

    data = []
    nSmooth = 4
    for i in range(nEpoch):
        U = np.array(sheet1.col_values(i, 1, nTime + 1))
        Delta_w = np.array(sheet2.col_values(i, 1, nTime + 1))
        I = np.array(sheet3.col_values(i, 1, nTime + 1))
        T_e = np.array(sheet4.col_values(i, 1, nTime + 1))
        w = np.array(sheet5.col_values(i, 1, nTime + 1))
        # W = np.array(sheet6.col_values(i, 1, nTime+1))/1000

        # U = np.convolve(U, np.ones(nSmooth)) / nSmooth
        # Delta_w = np.convolve(Delta_w, np.ones(nSmooth)) / nSmooth
        # I = np.convolve(I, np.ones(nSmooth)) / nSmooth
        # T_e = np.convolve(T_e, np.ones(nSmooth)) / nSmooth
        # w = np.convolve(w, np.ones(nSmooth)) / nSmooth
        # W = np.convolve(W, np.ones(nSmooth)) / nSmooth

        data.append([U, Delta_w, I, T_e, w])
    data = np.array(data)

    data = data.transpose((0, 2, 1))
    nFeature = data.shape[2]
    data_train = data[0:train_Partial, :, :]
    data_test = data[train_Partial:, :, :]

    data_train = data_train.reshape(-1, nFeature)

    sheet7 = workbook.sheet_by_name('Labels')
    data_labels = []
    for i in range(train_Partial, nEpoch):
        data_label = np.array(sheet7.col_values(i, 1, nTime + 1))
        data_labels.append(data_label)
    data_labels = np.array(data_labels)

    x = []
    y = []
    label = []
    for epoch in range(data_test.shape[0]):
        for i in range(0, nTime - nWindow - nPredict + 1, nInterval):
            head = i
            tail = i + nWindow
            x.append(data_test[epoch, head:tail])
            y.append(data_test[epoch, tail:tail + nPredict])
            label.append(data_labels[epoch, tail])
    x = np.array(x)
    y = np.array(y)
    label = np.array(label)

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test_x = torch.tensor(x, dtype=torch.float32)
    data_test_y = torch.tensor(y, dtype=torch.float32)
    data_test_label = torch.tensor(label, dtype=torch.float32)

    return data_train, data_test_x, data_test_y, data_test_label



if __name__ == '__main__':
    # No prior knowledge utilization
    # filepath = 'output2.xls'
    # train_Partial = 10
    # nTest = 5
    # nTest_see = 5
    # nWindow = 15
    # nc = 335
    # nPredict = 15
    # nInterval = 1
    # data_train, data_test_x, data_test_y, data_test_label = load_dataset2(filepath, nWindow, nPredict, nInterval, nc, train_Partial)
    #
    #
    # Input_feature = 1
    # Input_node_num = data_train.shape[1]
    # L_train = data_train.shape[0]
    #
    # Method_index = 0
    #
    # t1 = time.time()
    # for e in range(20):
    #     torch.manual_seed(e)
    #     A = generate_A(e)
    #     torch.manual_seed(15)
    #     Reservoir_node_num = 250
    #
    #     rho_R = 0.8
    #     rho_I = 0.843
    #
    #     # (torch.max(data_train)+torch.min(data_train))/2
    #     # 1.5/(torch.max(data_train)-torch.min(data_train))
    #     b = 0.4
    #     delta = 0.70
    #     density_input = 0.8
    #     transient = round(L_train * 0.1)
    #     M_K = 0.7
    #
    #     Att_N = 3
    #
    #     DT_Mean = torch.mean(data_train, dim=0)
    #     DT_Std = torch.std(data_train, dim=0)
    #
    #     DT_MAX = torch.max(data_train)
    #     DT_MIN = torch.min(data_train)
    #
    #     train_data = (data_train - DT_Mean) / DT_Std
    #     test_data_x = (data_test_x - DT_Mean) / DT_Std
    #
    #     # train_data = F.normalize(data_train, dim=0)
    #     # test_data_x = F.normalize(data_test_x, dim=0)
    #     # test_data_y = F.normalize(data_test_y, dim=0)
    #
    #     L_pre = np.array([1, 2, 3, 6, 8, 10, 12])
    #     W_in = (torch.rand(Input_feature, Reservoir_node_num) * 2 - 1)
    #
    #
    #     R_network_weight = torch.rand(Reservoir_node_num, Reservoir_node_num)
    #     R_network_0 = np.eye(Reservoir_node_num, k=1)
    #     R_network = torch.from_numpy(R_network_0) * (R_network_weight + R_network_weight.T) / 2
    #     R_network = R_network.type(torch.float32)
    #
    #     # Type_input_network = 'rand'
    #     # if Type_input_network == 'rand':
    #     #     Input_network_weight = torch.rand(Input_node_num, Input_node_num) * rho_I
    #     #     Input_network_0 = Network_initial('WS', network_size=Input_node_num, density=density_input)
    #     #     Input_network = torch.from_numpy(Input_network_0) * (Input_network_weight + Input_network_weight.T) / 2
    #     # elif Type_input_network == 'no_connection':
    #     #     Input_network = torch.eye(Input_node_num)
    #     # elif Type_input_network == 'fully_connected':
    #     #     Input_network = torch.ones(Input_node_num, Input_node_num)
    #     # Input_network = Input_network.type(torch.float32)
    #
    #     Input_network = A
    #
    #     # model = Reservoir(Dr=Reservoir_node_num, N_V=Input_node_num)
    #     model = Reservoir2(Dr=Reservoir_node_num, N_V=Input_node_num, ltrain=L_train - 1, W_in=W_in, R_network=R_network, rho=rho_R, delta=delta, b=b)
    #
    #     RC_learner = reservoir_computing(model=model, Dr=Reservoir_node_num, N_V=Input_node_num, N_F=Input_feature, rho=rho_R, delta=delta,
    #                                      b=b, transient=transient, Att_N=Att_N)
    #     W_out, Pre_train_output, R_state = RC_learner.Train_offline(train_data[:(L_train - 1), :],
    #                                                                 train_data[1:L_train, :],
    #                                                                 R_network, Input_network,
    #                                                                 W_in,
    #                                                                 index_method=4,
    #                                                                 K=M_K)
    #
    #
    #     R_state_test = RC_learner.Forward_B(test_data_x[:, :-1, :],
    #                                         R_network, Input_network, W_in, None, Method_index, W_out)
    #     Preds, _ = RC_learner.Predicting_phase_B(test_data_x[:, -1, :], R_state_test[:, -1, :, :],
    #                                              R_network, Input_network, W_in, W_out, Pre_L=12)
    #
    #     Real1 = data_test_y[:, :].numpy()
    #     Preds1 = (Preds * DT_Std + DT_Mean).detach().numpy()
    #
    #
    #     len_test = int(Preds1.shape[0] / nTest)
    #     x = range(len_test)
    #     # for i in range(nTest):
    #     #     plt.subplot(6, nTest, 1 + i)
    #     #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 0], label="U_real")
    #     #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 0], label="U")
    #     #     # plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 0] - Real1[i*len_test:(i+1)*len_test, 0, 0], label="error")
    #     #     plt.legend()
    #     #
    #     #     plt.subplot(6, nTest, 1 + i + nTest * 1)
    #     #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 1], label="Delta_w_real")
    #     #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 1], label='Delta_w')
    #     #     plt.legend()
    #     #
    #     #     plt.subplot(6, nTest, 1 + i + nTest * 2)
    #     #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 2], label="I_real")
    #     #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 2], label="I")
    #     #     plt.legend()
    #     #
    #     #     plt.subplot(6, nTest, 1 + i + nTest * 3)
    #     #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 3], label="T_e_real")
    #     #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 3], label="T_e")
    #     #     plt.legend()
    #     #
    #     #     plt.subplot(6, nTest, 1 + i + nTest * 4)
    #     #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 4], label="w_real")
    #     #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 4], label="w")
    #     #     plt.legend()
    #     #
    #     #     plt.subplot(6, nTest, 1 + i + nTest * 5)
    #     #     plt.plot(x, data_test_label[i * len_test:(i + 1) * len_test], label="Label")
    #     #     plt.legend()
    #     #
    #     #     # plt.subplot(6, 1, 6)
    #     #     # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")
    #     #     # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")
    #     #     # plt.legend()
    #     # plt.show()
    #
    #
    #     # x = range(Preds1.shape[0])
    #     # plt.subplot(6, 1, 1)
    #     # plt.plot(x, Real1[:, 0, 0] * 1000, label="U_real")
    #     # plt.plot(x, Preds1[:, 0, 0] * 1000, label="U")
    #     # plt.legend()
    #     #
    #     # plt.subplot(6, 1, 2)
    #     # plt.plot(x, Real1[:, 0, 1] * 1000, label="Delta_w_real")
    #     # plt.plot(x, Preds1[:, 0, 1] * 1000, label='Delta_w')
    #     # plt.legend()
    #     #
    #     # plt.subplot(6, 1, 3)
    #     # plt.plot(x, Real1[:, 0, 2] * 100, label="I_real")
    #     # plt.plot(x, Preds1[:, 0, 2] * 100, label="I")
    #     # plt.legend()
    #     #
    #     # plt.subplot(6, 1, 4)
    #     # plt.plot(x, Real1[:, 0, 3] * 10, label="T_e_real")
    #     # plt.plot(x, Preds1[:, 0, 3] * 10, label="T_e")
    #     # plt.legend()
    #     #
    #     # plt.subplot(6, 1, 5)
    #     # plt.plot(x, Real1[:, 0, 4] * 1000, label="w_real")
    #     # plt.plot(x, Preds1[:, 0, 4] * 1000, label="w")
    #     # plt.legend()
    #     #
    #     # plt.subplot(6, 1, 6)
    #     # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")
    #     # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")
    #     # plt.legend()
    #     #
    #     # plt.show()
    #
    #     # for zz in range(L_pre.shape[0]):
    #     #     for i in range(3):
    #     #         Loss_test[zz, i] = Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i)
    #     # np.set_printoptions(suppress=True)
    #     # print(Loss_test)
    #     Loss_test = []
    #     for i in range(3):
    #         for zz in range(L_pre.shape[0]):
    #             Loss_test.append(Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i, data_test_label=np.expand_dims(data_test_label.numpy(), axis=1)))
    #     Loss_test = np.array(Loss_test).reshape((3, L_pre.shape[0], -1))
    #     # print(Loss_test)
    #     anomaly_scores = np.zeros_like(Preds1[0:len_test * nTest_see])
    #     for i in range(Preds1.shape[1]):
    #         a_score = np.sqrt((Preds1[0:len_test * nTest_see, i] - Real1[0:len_test * nTest_see, i]) ** 2)
    #         anomaly_scores[:, i] = a_score
    #     scaler = StandardScaler()
    #     scaled_anomaly_scores = scaler.fit_transform(anomaly_scores[:, 0, :])
    #     # x = scaler.mean_
    #     # y = scaler.scale_
    #     # z = (anomaly_scores[:,0,:] - scaler.mean_)/scaler.scale_
    #     scaled_anomaly_scores_max = np.max(scaled_anomaly_scores, 1)
    #     precisions, recalls, thresholds = precision_recall_curve(data_test_label[0:len_test * nTest_see], scaled_anomaly_scores_max)
    #     f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    #     best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    #     best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    #     print(best_f1_score, thresholds[best_f1_score_index])
    #
    #     save_filename = 'motor_control_random.xls'
    #     save_results(save_filename, Preds1, Real1, data_test_label.numpy(), thresholds, precisions, recalls, best_f1_score, best_f1_score_index, Loss_test)

    # Parametric study
    # filepath = 'output2.xls'
    # train_Partial = 10
    # nTest = 5  
    # nTest_see = 5  
    # nWindow = 15
    # nc = 335
    # nPredict = 15
    # nInterval = 1
    # data_train, data_test_x, data_test_y, data_test_label = load_dataset2(filepath, nWindow, nPredict, nInterval, nc, train_Partial)
    #
    # 
    # Input_feature = 1  
    # Input_node_num = data_train.shape[1]  
    # L_train = data_train.shape[0]  
    #
    # Method_index = 0
    #
    # time_lists = []
    # for node in range(50, 650, 50):
    #     time_list = []
    #     for e in range(10):
    #         t1 = time.time()
    #         torch.manual_seed(e)  
    #         Reservoir_node_num = node  
    #
    #         rho_R = 0.8  
    #         rho_I = 0.843  
    #
    #         # (torch.max(data_train)+torch.min(data_train))/2
    #         # 1.5/(torch.max(data_train)-torch.min(data_train))
    #         b = 0.4  
    #         delta = 0.70  
    #         density_input = 0.8  
    #         transient = round(L_train * 0.1)  
    #         M_K = 0.7  
    #
    #         Att_N = 3  
    #
    #         DT_Mean = torch.mean(data_train, dim=0)
    #         DT_Std = torch.std(data_train, dim=0)
    #
    #         DT_MAX = torch.max(data_train)
    #         DT_MIN = torch.min(data_train)
    #
    #         train_data = (data_train - DT_Mean) / DT_Std
    #         test_data_x = (data_test_x - DT_Mean) / DT_Std
    #
    #         # train_data = F.normalize(data_train, dim=0)
    #         # test_data_x = F.normalize(data_test_x, dim=0)
    #         # test_data_y = F.normalize(data_test_y, dim=0)
    #
    #         
    #         L_pre = np.array([1, 2, 3, 6, 8, 10, 12])
    #
    #         
    #         W_in = (torch.rand(Input_feature, Reservoir_node_num) * 2 - 1)  
    #
    #         
    #         R_network_weight = torch.rand(Reservoir_node_num, Reservoir_node_num)
    #         R_network_0 = np.eye(Reservoir_node_num, k=1)
    #         R_network = torch.from_numpy(R_network_0) * (R_network_weight + R_network_weight.T) / 2
    #         R_network = R_network.type(torch.float32)
    #
    #         
    #         # Type_input_network = 'rand'
    #         # if Type_input_network == 'rand':
    #         #     Input_network_weight = torch.rand(Input_node_num, Input_node_num) * rho_I
    #         #     Input_network_0 = Network_initial('WS', network_size=Input_node_num, density=density_input)
    #         #     Input_network = torch.from_numpy(Input_network_0) * (Input_network_weight + Input_network_weight.T) / 2
    #         # elif Type_input_network == 'no_connection':
    #         #     Input_network = torch.eye(Input_node_num)
    #         # elif Type_input_network == 'fully_connected':
    #         #     Input_network = torch.ones(Input_node_num, Input_node_num)
    #         # Input_network = Input_network.type(torch.float32)
    #
    #         Input_network = A
    #
    #         # model = Reservoir(Dr=Reservoir_node_num, N_V=Input_node_num)
    #         model = Reservoir2(Dr=Reservoir_node_num, N_V=Input_node_num, ltrain=L_train - 1, W_in=W_in, R_network=R_network, rho=rho_R, delta=delta, b=b)
    #
    #         RC_learner = reservoir_computing(model=model, Dr=Reservoir_node_num, N_V=Input_node_num, N_F=Input_feature, rho=rho_R, delta=delta,
    #                                          b=b, transient=transient, Att_N=Att_N)
    #         W_out, Pre_train_output, R_state = RC_learner.Train_offline(train_data[:(L_train - 1), :],
    #                                                                     train_data[1:L_train, :],
    #                                                                     R_network, Input_network,
    #                                                                     W_in,
    #                                                                     index_method=4,
    #                                                                     K=M_K)
    #         t2 = time.time()
    #         # print(node, t2-t1)
    #         time_list.append(t2 - t1)
    #        
    #         R_state_test = RC_learner.Forward_B(test_data_x[:, :-1, :],
    #                                             R_network, Input_network, W_in, None, Method_index, W_out)  
    #         Preds, _ = RC_learner.Predicting_phase_B(test_data_x[:, -1, :], R_state_test[:, -1, :, :],
    #                                                  R_network, Input_network, W_in, W_out, Pre_L=12)
    #
    #         Real1 = data_test_y[:, :].numpy()
    #         Preds1 = (Preds * DT_Std + DT_Mean).detach().numpy()
    #
    #         # 画图，看差多少
    #         # 画每个组
    #         len_test = int(Preds1.shape[0] / nTest)
    #         x = range(len_test)
    #         # for i in range(nTest):
    #         #     plt.subplot(6, nTest, 1 + i)
    #         #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 0], label="U_real")  
    #         #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 0], label="U")  
    #         #     # plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 0] - Real1[i*len_test:(i+1)*len_test, 0, 0], label="error")
    #         #     plt.legend()
    #         #
    #         #     plt.subplot(6, nTest, 1 + i + nTest * 1)
    #         #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 1], label="Delta_w_real")
    #         #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 1], label='Delta_w')
    #         #     plt.legend()
    #         #
    #         #     plt.subplot(6, nTest, 1 + i + nTest * 2)
    #         #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 2], label="I_real")
    #         #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 2], label="I")
    #         #     plt.legend()
    #         #
    #         #     plt.subplot(6, nTest, 1 + i + nTest * 3)
    #         #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 3], label="T_e_real")
    #         #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 3], label="T_e")
    #         #     plt.legend()
    #         #
    #         #     plt.subplot(6, nTest, 1 + i + nTest * 4)
    #         #     plt.plot(x, Real1[i * len_test:(i + 1) * len_test, 0, 4], label="w_real")
    #         #     plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 4], label="w")
    #         #     plt.legend()
    #         #
    #         #     plt.subplot(6, nTest, 1 + i + nTest * 5)
    #         #     plt.plot(x, data_test_label[i * len_test:(i + 1) * len_test], label="Label")  
    #         #     plt.legend()
    #         #
    #         #     # plt.subplot(6, 1, 6)
    #         #     # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")  
    #         #     # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")  
    #         #     # plt.legend()
    #         # plt.show()
    #
    #         
    #         # x = range(Preds1.shape[0])
    #         # plt.subplot(6, 1, 1)
    #         # plt.plot(x, Real1[:, 0, 0] * 1000, label="U_real")  
    #         # plt.plot(x, Preds1[:, 0, 0] * 1000, label="U")  
    #         # plt.legend()
    #         #
    #         # plt.subplot(6, 1, 2)
    #         # plt.plot(x, Real1[:, 0, 1] * 1000, label="Delta_w_real")
    #         # plt.plot(x, Preds1[:, 0, 1] * 1000, label='Delta_w')
    #         # plt.legend()
    #         #
    #         # plt.subplot(6, 1, 3)
    #         # plt.plot(x, Real1[:, 0, 2] * 100, label="I_real")
    #         # plt.plot(x, Preds1[:, 0, 2] * 100, label="I")
    #         # plt.legend()
    #         #
    #         # plt.subplot(6, 1, 4)
    #         # plt.plot(x, Real1[:, 0, 3] * 10, label="T_e_real")
    #         # plt.plot(x, Preds1[:, 0, 3] * 10, label="T_e")
    #         # plt.legend()
    #         #
    #         # plt.subplot(6, 1, 5)
    #         # plt.plot(x, Real1[:, 0, 4] * 1000, label="w_real")
    #         # plt.plot(x, Preds1[:, 0, 4] * 1000, label="w")
    #         # plt.legend()
    #         #
    #         # plt.subplot(6, 1, 6)
    #         # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")  
    #         # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")  
    #         # plt.legend()
    #         #
    #         # plt.show()
    #
    #         
    #         # for zz in range(L_pre.shape[0]):
    #         #     for i in range(3):
    #         #         Loss_test[zz, i] = Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i)
    #         # np.set_printoptions(suppress=True)
    #         # print(Loss_test)  
    #         Loss_test = []
    #         for i in range(3):
    #             for zz in range(L_pre.shape[0]):
    #                 Loss_test.append(Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i, data_test_label=np.expand_dims(data_test_label.numpy(), axis=1)))
    #         Loss_test = np.array(Loss_test).reshape((3, L_pre.shape[0], -1))
    #         # print(Loss_test)  
    #
    #         
    #         anomaly_scores = np.zeros_like(Preds1[0:len_test * nTest_see])
    #         for i in range(Preds1.shape[1]):
    #             a_score = np.sqrt((Preds1[0:len_test * nTest_see, i] - Real1[0:len_test * nTest_see, i]) ** 2)
    #             anomaly_scores[:, i] = a_score
    #     
    #         scaler = StandardScaler()
    #         scaled_anomaly_scores = scaler.fit_transform(anomaly_scores[:, 0, :])
    #         # x = scaler.mean_
    #         # y = scaler.scale_
    #         # z = (anomaly_scores[:,0,:] - scaler.mean_)/scaler.scale_ 
    #         scaled_anomaly_scores_max = np.max(scaled_anomaly_scores, 1)
    #         precisions, recalls, thresholds = precision_recall_curve(data_test_label[0:len_test * nTest_see], scaled_anomaly_scores_max)
    #         
    #         f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    #         best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    #         best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    #         
    #         print(best_f1_score, thresholds[best_f1_score_index])
    #
    #         save_filename = 'motor_control' + str(Reservoir_node_num) + '.xls'
    #         save_results(save_filename, Preds1, Real1, data_test_label.numpy(), thresholds, precisions, recalls, best_f1_score, best_f1_score_index, Loss_test)
    #     time_list += [np.mean(time_list), np.max(time_list), np.min(time_list)]
    #     time_lists.append(time_list)
    # save_complexity(time_lists)


    filepath = 'output2.xls'
    train_Partial = 10
    nTest = 5
    nTest_see = 5
    nWindow = 15
    nc = 335
    nPredict = 15
    nInterval = 1
    data_train, data_test_x, data_test_y, data_test_label = load_dataset2(filepath, nWindow, nPredict, nInterval, nc, train_Partial)

    t1 = time.time()

    Input_feature = 1
    Input_node_num = data_train.shape[1]
    L_train = data_train.shape[0]

    Method_index = 0
    torch.manual_seed(15)
    t1 = time.time()

    Reservoir_node_num = 250

    rho_R = 0.8
    rho_I = 0.843

    # (torch.max(data_train)+torch.min(data_train))/2
    # 1.5/(torch.max(data_train)-torch.min(data_train))
    b = 0.4
    delta = 0.70
    density_input = 0.8
    transient = round(L_train * 0.1)
    M_K = 0.7

    Att_N = 3

    DT_Mean = torch.mean(data_train, dim=0)
    DT_Std = torch.std(data_train, dim=0)

    DT_MAX = torch.max(data_train)
    DT_MIN = torch.min(data_train)

    train_data = (data_train - DT_Mean) / DT_Std
    test_data_x = (data_test_x - DT_Mean) / DT_Std

    # train_data = F.normalize(data_train, dim=0)
    # test_data_x = F.normalize(data_test_x, dim=0)
    # test_data_y = F.normalize(data_test_y, dim=0)


    L_pre = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    W_in = (torch.rand(Input_feature, Reservoir_node_num) * 2 - 1)


    R_network_weight = torch.rand(Reservoir_node_num, Reservoir_node_num)
    R_network_0 = np.eye(Reservoir_node_num, k=1)
    R_network = torch.from_numpy(R_network_0) * (R_network_weight + R_network_weight.T) / 2
    R_network = R_network.type(torch.float32)


    # Type_input_network = 'rand'
    # if Type_input_network == 'rand':
    #     Input_network_weight = torch.rand(Input_node_num, Input_node_num) * rho_I
    #     Input_network_0 = Network_initial('WS', network_size=Input_node_num, density=density_input)
    #     Input_network = torch.from_numpy(Input_network_0) * (Input_network_weight + Input_network_weight.T) / 2
    # elif Type_input_network == 'no_connection':
    #     Input_network = torch.eye(Input_node_num)
    # elif Type_input_network == 'fully_connected':
    #     Input_network = torch.ones(Input_node_num, Input_node_num)
    # Input_network = Input_network.type(torch.float32)

    Input_network = A

    # model = Reservoir(Dr=Reservoir_node_num, N_V=Input_node_num)
    model = Reservoir2(Dr=Reservoir_node_num, N_V=Input_node_num, ltrain=L_train - 1, W_in=W_in, R_network=R_network, rho=rho_R, delta=delta, b=b)

    RC_learner = reservoir_computing(model=model, Dr=Reservoir_node_num, N_V=Input_node_num, N_F=Input_feature, rho=rho_R, delta=delta,
                                     b=b, transient=transient, Att_N=Att_N)
    W_out, Pre_train_output, R_state = RC_learner.Train_offline(train_data[:(L_train - 1), :],
                                                                train_data[1:L_train, :],
                                                                R_network, Input_network,
                                                                W_in,
                                                                index_method=4,
                                                                K=M_K)
    t2 = time.time()
    print(t2 - t1)

    R_state_test = RC_learner.Forward_B(test_data_x[:, :-1, :],
                                        R_network, Input_network, W_in, None, Method_index, W_out)
    Preds, _ = RC_learner.Predicting_phase_B(test_data_x[:, -1, :], R_state_test[:, -1, :, :],
                                             R_network, Input_network, W_in, W_out, Pre_L=12)

    Real1 = data_test_y[:, :].numpy()
    Preds1 = (Preds * DT_Std + DT_Mean).detach().numpy()

    len_test = int(Preds1.shape[0] / nTest)
    x = range(len_test)
    for i in range(nTest):
        plt.subplot(6, nTest, 1+i)
        plt.plot(x, Real1[i*len_test:(i+1)*len_test, 0, 0], label="U_real")
        plt.plot(x, Preds1[i*len_test:(i+1)*len_test, 0, 0], label="U")
        # plt.plot(x, Preds1[i * len_test:(i + 1) * len_test, 0, 0] - Real1[i*len_test:(i+1)*len_test, 0, 0], label="error")
        plt.legend()

        plt.subplot(6, nTest, 1 + i + nTest * 1)
        plt.plot(x, Real1[i*len_test:(i+1)*len_test, 0, 1], label="Delta_w_real")
        plt.plot(x, Preds1[i*len_test:(i+1)*len_test, 0, 1], label='Delta_w')
        plt.legend()

        plt.subplot(6, nTest, 1 + i + nTest * 2)
        plt.plot(x, Real1[i*len_test:(i+1)*len_test, 0, 2], label="I_real")
        plt.plot(x, Preds1[i*len_test:(i+1)*len_test, 0, 2], label="I")
        plt.legend()

        plt.subplot(6, nTest, 1 + i + nTest * 3)
        plt.plot(x, Real1[i*len_test:(i+1)*len_test, 0, 3], label="T_e_real")
        plt.plot(x, Preds1[i*len_test:(i+1)*len_test, 0, 3], label="T_e")
        plt.legend()

        plt.subplot(6, nTest, 1 + i + nTest * 4)
        plt.plot(x, Real1[i*len_test:(i+1)*len_test, 0, 4], label="w_real")
        plt.plot(x, Preds1[i*len_test:(i+1)*len_test, 0, 4], label="w")
        plt.legend()

        plt.subplot(6, nTest, 1 + i + nTest * 5)
        plt.plot(x, data_test_label[i*len_test:(i+1)*len_test], label="Label")
        plt.legend()

        # plt.subplot(6, 1, 6)
        # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")
        # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")
        # plt.legend()
    plt.show()


    # x = range(Preds1.shape[0])
    # plt.subplot(6, 1, 1)
    # plt.plot(x, Real1[:, 0, 0] * 1000, label="U_real")
    # plt.plot(x, Preds1[:, 0, 0] * 1000, label="U")
    # plt.legend()
    #
    # plt.subplot(6, 1, 2)
    # plt.plot(x, Real1[:, 0, 1] * 1000, label="Delta_w_real")
    # plt.plot(x, Preds1[:, 0, 1] * 1000, label='Delta_w')
    # plt.legend()
    #
    # plt.subplot(6, 1, 3)
    # plt.plot(x, Real1[:, 0, 2] * 100, label="I_real")
    # plt.plot(x, Preds1[:, 0, 2] * 100, label="I")
    # plt.legend()
    #
    # plt.subplot(6, 1, 4)
    # plt.plot(x, Real1[:, 0, 3] * 10, label="T_e_real")
    # plt.plot(x, Preds1[:, 0, 3] * 10, label="T_e")
    # plt.legend()
    #
    # plt.subplot(6, 1, 5)
    # plt.plot(x, Real1[:, 0, 4] * 1000, label="w_real")
    # plt.plot(x, Preds1[:, 0, 4] * 1000, label="w")
    # plt.legend()
    #
    # plt.subplot(6, 1, 6)
    # plt.plot(x, Real1[:, 1, 0] * 1000, label="U_real")
    # plt.plot(x, Preds1[:, 1, 0] * 1000, label="U_2")
    # plt.legend()
    #
    # plt.show()


    # for zz in range(L_pre.shape[0]):
    #     for i in range(3):
    #         Loss_test[zz, i] = Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i)
    # np.set_printoptions(suppress=True)
    # print(Loss_test)
    Loss_test = []
    for i in range(3):
        for zz in range(L_pre.shape[0]):
            Loss_test.append(Loss_cal(Preds1[:, L_pre[zz] - 1, :], Real1[:, L_pre[zz] - 1, :], Method=i, data_test_label = np.expand_dims(data_test_label.numpy(), axis=1)))
    Loss_test = np.array(Loss_test).reshape((3,L_pre.shape[0],-1))
    # print(Loss_test)


    anomaly_scores = np.zeros_like(Preds1[0:len_test*nTest_see])
    for i in range(Preds1.shape[1]):
        a_score = np.sqrt((Preds1[0:len_test*nTest_see, i] - Real1[0:len_test*nTest_see, i]) ** 2)
        anomaly_scores[:, i] = a_score

    scaler = StandardScaler()
    scaled_anomaly_scores = scaler.fit_transform(anomaly_scores[:, 0, :])
    # x = scaler.mean_
    # y = scaler.scale_
    # z = (anomaly_scores[:,0,:] - scaler.mean_)/scaler.scale_
    scaled_anomaly_scores_max = np.max(scaled_anomaly_scores, 1)
    precisions, recalls, thresholds = precision_recall_curve(data_test_label[0:len_test*nTest_see], scaled_anomaly_scores_max)

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    

    # compare = np.zeros((2, scaled_anomaly_scores_max.shape[0]))
    # compare[0,:] = data_test_label[0:len_test*nTest_see].detach().numpy()
    # for x in range(scaled_anomaly_scores_max.shape[0]):
    #     if scaled_anomaly_scores_max[x]>=thresholds[best_f1_score_index]:
    #         compare[1, x] = 1

    save_filename = 'motor_control.xls'
    save_result(save_filename, Preds1, Real1, data_test_label.numpy(), thresholds, precisions, recalls, best_f1_score, best_f1_score_index, Loss_test)
    print("Results save to motor_control.xls")