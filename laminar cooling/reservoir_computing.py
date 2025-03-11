import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LassoCV, RidgeCV, Ridge, ElasticNetCV, orthogonal_mp, OrthogonalMatchingPursuit
from sklearn import linear_model
import networkx as nx


np.random.seed(1)
torch.manual_seed(1)


def R_shuffle(node_number=0, path_length=0):
    x = [np.random.random() for i in range(path_length)]
    #    x = [path_length/node_number for i in range(path_length)]
    e = [int(i / sum(x) * (node_number - path_length)) + 1 for i in x]
    re = node_number - sum(e)
    u = [np.random.randint(0, path_length - 1) for i in range(re)]

    for i in range(re):
        e[u[i]] += 1
    return e


def Network_initial(network_name=None, network_size=300, density=0.2, Depth=10, MC_configure=None):
    if network_name is "ER":
        rg = nx.erdos_renyi_graph(network_size, density, seed=1, directed=False)
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name is "regular":
        K_number = int(network_size * density)
        rg = nx.random_graphs.random_regular_graph(K_number, network_size)
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name is "WS":
        K_number = int(network_size * density)
        rg = nx.random_graphs.watts_strogatz_graph(network_size, K_number, 0.3)
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name is "DCG":
        rg = nx.erdos_renyi_graph(network_size, density, directed=True)
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name is "DAG":
        if MC_configure is not None:
            xx = np.append(0, np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0] - 1):
                Reject_index = 1
                for j in range(0, xx.shape[0] - 1):
                    if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1], MC_configure[j + 1] + 1)):
                        Reject_index = 0
                if Reject_index == 1 and (MC_configure[i + 1] != 1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")
                    Reject_index = 2
            if Reject_index != 2:
                R_initial_0 = np.zeros((network_size, network_size))
                for i in range(xx.shape[0] - 1):
                    for j in range(xx.shape[0] - 1):
                        if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1] + 1, MC_configure[j + 1])):
                            R_initial_0[xx[i]:xx[i + 1], xx[j]:xx[j + 1]] = 1
                R_initial = np.triu(R_initial_0, 1)
            else:
                R_initial = None

        else:
            xx = R_shuffle(network_size, Depth)
            # xx=np.array([3,4,3])
            # xx=np.array([60,60,60,60,60])
            # xx=np.array([30,30,30,30,30,30,30,30,30,30])*3
            rg = nx.complete_multipartite_graph(*tuple(xx))
            x = nx.adjacency_matrix(rg).toarray()
            R_initial = np.triu(x, 1)
            # R_initial= np.tril(x,1)
        Real_density = np.sum(R_initial > 0) * 1.0 / (network_size ** 2)
        if Real_density > 0 and density < Real_density:
            R_initial[np.random.rand(*R_initial.shape) <= (1.0 - density / Real_density)] = 0

        R_initial = np.triu(R_initial, 1)
    return R_initial


def Loss_cal(Prediction=None, Real_data=None, Method=0):
    Loss = 0
    Loss_Final = 0
    mask = np.not_equal(Real_data, 0)
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    if Method == 1:
        Loss = np.square(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_int = np.sqrt(np.mean(Loss, 1))
        Loss_Final = np.median(Loss_int)
    elif Method == 0:
        Loss = np.abs(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_int = np.mean(Loss, 1)
        Loss_Final = np.median(Loss_int)
    else:
        Loss = np.abs(np.divide(np.subtract(Prediction, Real_data).astype('float32')
                                , Real_data))
        Loss = np.nan_to_num(mask * Loss)
        Loss_int = np.mean(Loss, 1)
        Loss_Final = np.median(Loss_int) * 100
    return Loss_Final


def traing_Wout(train_data, R_state, index=0, k=0.8):
    W_out = torch.zeros(R_state.shape[1], train_data.shape[1])
    if index == 0:
        W_out = torch.mm(torch.pinverse(R_state), train_data)
    else:
        alphas = 10 ** np.linspace(-10, 10, 100)
        if index == 1:
            alphas = 10 ** np.linspace(-4, 3, 15)
            base_cv = LassoCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        if index == 2:
            alphas = 10 ** np.linspace(-4, 3, 15)
            base_cv = RidgeCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        if index == 3:
            alphas = 10 ** np.linspace(-4, 3, 15)
            base_cv = ElasticNetCV(alphas=alphas, cv=20, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        if index == 4:
            if int(R_state.shape[1] * k) > 1:
                anova_filter = SelectKBest(f_regression, k=int(R_state.shape[1] * k))
            else:
                anova_filter = SelectKBest(f_regression, k=int(1))
        if index == 5:
            base = linear_model.LinearRegression(fit_intercept=True)
            anova_filter = RFECV(base)
        if index == 6:
            base_cv = OrthogonalMatchingPursuit(fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)

        clf = Pipeline([
            ('feature_selection', anova_filter),
            ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
        ])
        for X_i in range(train_data.shape[1]):
            clf.fit(R_state, train_data[:, X_i])
            W_out.data[clf.named_steps['feature_selection'].get_support(), X_i] = torch.from_numpy(clf.named_steps['Linearregression'].coef_)
    return W_out



class reservoir_computing(nn.Module):
    def __init__(self, model=None, Dr=300, N_V=40, N_F=1, rho=1, delta=0.1, b=0, transient=1000, Att_N=2, input_window=12, output_window=12):
        super().__init__()
        self.model = model

        self.N_V = N_V
        self.Dr = Dr
        self.transient = transient
        self.N_F = N_F
        self.Att_N = Att_N
        self.Input_window = input_window
        self.Output_window = output_window
        self.x_offsets = np.sort(np.concatenate((np.arange(-self.Input_window + 1, 1, 1),)))
        self.y_offsets = np.sort(np.arange(1, self.Output_window + 1, 1))


        self.delta = torch.rand(1).fill_(delta)
        self.rho = torch.rand(1).fill_(rho)
        self.b = torch.rand(1).fill_(b)

    def Train_offline(self, train_data, ground_truth, R_network, Input_network, W_in, index_method=0, K=0):
        L_T = train_data.shape[0]
        R_state = torch.zeros(L_T, self.N_V, self.Dr)
        Pre_train_output = torch.zeros(L_T, self.N_V)
        W_out = torch.zeros(self.N_V, self.Dr)

        for i in range(1, (L_T)):
            R_state[i, :] = (torch.tanh(torch.mm(Input_network.T, self.rho * torch.mm(R_state[i - 1, :], R_network))
                                        + self.delta * torch.mm(train_data[i, :].view(-1, 1), W_in) + self.b))

        for i in range(self.N_V):
            W_out[i, :] = traing_Wout(ground_truth[:, i].view(-1, 1),
                                      R_state[:, i, :], index=index_method, k=K).view(self.Dr)

        for i in range(self.N_V):
            Pre_train_output[:, i] = torch.mm(R_state[:, i, :], W_out[i, :].view(-1, 1)).view(L_T)
        return W_out, Pre_train_output, R_state

    def Forward_B(self, test_data, R_network, Input_network, W_in, R_state_initial=None, Method=0, W_out=None):
        B_T = test_data.shape[0]
        L_T = test_data.shape[1]
        R_state = torch.zeros(B_T, L_T, self.N_V, self.Dr)
        if R_state_initial is not None:
            R_state[:, 0, :] = R_state_initial

        R_state[:, 0, :] = (torch.tanh(self.delta * torch.matmul(test_data[:, 0, :].unsqueeze(-1), W_in)
                           + torch.matmul(Input_network.T, self.rho * torch.matmul(R_state[:, 0, :], R_network)) + self.b))  # R_state[0,:] 相当于-1
        x = R_state[:, 0, :]
        y = test_data[:, 0, :].unsqueeze(-1)
        for i in range(1, (L_T)):
            R_state[:, i, :] = (torch.tanh(self.delta * torch.matmul(test_data[:, i, :].unsqueeze(-1), W_in)
                               + torch.matmul(Input_network.T, self.rho * torch.matmul(R_state[:, i - 1, :], R_network)) + self.b))
        return R_state

    def Predicting_phase_B(self, test_data, R_state_initial, R_network, Input_network, W_in,
                           W_out, Pre_L=100):

        test_data_x = test_data.clone().detach()
        R_state = []
        test_output = []

        for i in range(Pre_L):
            R_state_initial = (torch.tanh(self.delta * torch.matmul(test_data_x.unsqueeze(-1), W_in)
                                          + torch.matmul(Input_network.T, self.rho * torch.matmul(R_state_initial, R_network))
                                          + self.b))
            for ii in range(self.N_V):
                # x = test_data_x[:, ii]
                # y = torch.matmul(R_state_initial[:, ii, :], W_out[ii, :])
                test_data_x[:, ii] = torch.matmul(R_state_initial[:, ii, :], W_out[ii, :])
            R_state.append(R_state_initial)
            input_initial_final = test_data_x.clone()
            test_output.append(input_initial_final)

        R_state_final = torch.stack(R_state, dim=1)
        test_output_final = torch.stack(test_output, dim=1)
        return test_output_final, R_state_final

    def Predict_online_B(self, test_data, R_network, Input_network, W_in, W_out, Method_index, Pre_L=12):
        L_T = test_data.shape[1] - 1
        R_state = torch.zeros(1, self.N_V, self.Dr)

        for i in range(L_T):
            R_state = (torch.tanh(self.delta * torch.matmul(test_data[:, i, :].unsqueeze(-1), W_in)
                               + torch.matmul(Input_network.T, self.rho * torch.matmul(R_state, R_network)) + self.b))
        test_output = []
        data = test_data[:, -1, :].clone()
        print(data)
        for i in range(Pre_L):
            R_state = (torch.tanh(self.delta * torch.matmul(data.unsqueeze(-1), W_in)
                                + torch.matmul(Input_network.T, self.rho * torch.matmul(R_state, R_network)) + self.b))
            for ii in range(self.N_V):
                data[:, ii] = torch.matmul(R_state[:, ii, :], W_out[ii, :])
            test_output.append(data.clone())
        test_output_final = torch.stack(test_output, dim=1)
        return test_output_final