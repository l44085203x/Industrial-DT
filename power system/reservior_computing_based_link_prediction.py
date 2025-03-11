import torch
import numpy as np

# No prior knowledge utilization
def generate_A(seed):
    torch.manual_seed(seed)
    n = 8
    A1 = torch.randint(0, 2, (n, n), dtype=torch.float32)
    A2 = torch.randint(0, 2, (n, n), dtype=torch.float32)
    for i in range(n):
        A1[i, i] = 0
        A2[i, i] = 0

    Input_node_num = A1.shape[0]
    Reservoir_node_num = 10
    rho = 0.8
    delta = 0.8
    b = 0
    beta = 5

    W_in = (torch.rand(Input_node_num, Reservoir_node_num) * 2 - 1)

    R_network_weight = torch.rand(Reservoir_node_num, Reservoir_node_num)
    R_network_0 = np.eye(Reservoir_node_num, k=1)
    R_network = torch.from_numpy(R_network_0) * (R_network_weight + R_network_weight.T) / 2
    R_network = R_network.type(torch.float32)

    R_state1 = torch.zeros(Input_node_num, Reservoir_node_num)
    R_state2 = torch.zeros(Input_node_num, Reservoir_node_num)
    for i in range(A1.shape[1]):
        R_state1 = (torch.tanh(rho * torch.mm(R_state1, R_network)
                        + delta * torch.mm(A1, W_in) + b))
        R_state2 = (torch.tanh(rho * torch.mm(R_state2, R_network)
                        + delta * torch.mm(A2, W_in) + b))

    W_out = torch.mm(torch.inverse(torch.mm(R_state1.T, R_state1)
                                 + torch.mm(R_state2.T, R_state2)
                                 + beta*torch.eye(Reservoir_node_num)),
                     torch.mm(R_state1.T, A1)+torch.mm(R_state2.T, A2))

    train_output1 = torch.mm(R_state1, W_out)
    train_output2 = torch.mm(R_state2, W_out)
    A = train_output1 + train_output2

    X_min = A.min()
    X_max = A.max()

    A = (A - X_min) / (X_max - X_min)
    print("Weighted_topology:", A)
    return A


A1 = torch.tensor([[0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)

A2 = torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.float32)

torch.manual_seed(1)
Input_node_num = A1.shape[0]
Reservoir_node_num = 10
rho = 0.8
delta = 0.8
b = 0
beta = 5

W_in = (torch.rand(Input_node_num, Reservoir_node_num) * 2 - 1)

R_network_weight = torch.rand(Reservoir_node_num, Reservoir_node_num)
R_network_0 = np.eye(Reservoir_node_num, k=1)
R_network = torch.from_numpy(R_network_0) * (R_network_weight + R_network_weight.T) / 2
R_network = R_network.type(torch.float32)

R_state1 = torch.zeros(Input_node_num, Reservoir_node_num)
R_state2 = torch.zeros(Input_node_num, Reservoir_node_num)
for i in range(A1.shape[1]):
    R_state1 = (torch.tanh(rho * torch.mm(R_state1, R_network)
                    + delta * torch.mm(A1, W_in) + b))
    R_state2 = (torch.tanh(rho * torch.mm(R_state2, R_network)
                    + delta * torch.mm(A2, W_in) + b))

W_out = torch.mm(torch.inverse(torch.mm(R_state1.T, R_state1)
                             + torch.mm(R_state2.T, R_state2)
                             + beta*torch.eye(Reservoir_node_num)),
                 torch.mm(R_state1.T, A1)+torch.mm(R_state2.T, A2))

train_output1 = torch.mm(R_state1, W_out)
train_output2 = torch.mm(R_state2, W_out)
A = train_output1 + train_output2

X_min = A.min()
X_max = A.max()

A = (A - X_min) / (X_max - X_min)
np.save('A_PC.npy', A1)
np.save('A_CE.npy', A2)
np.save('A.npy', A)
print("Weighted_topology:", A)