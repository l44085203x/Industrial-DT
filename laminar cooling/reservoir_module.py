import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

class Reservoir(nn.Module):
    def __init__(self, Dr=300, N_V=40):
        super(Reservoir, self).__init__()
        self.N_V = N_V
        self.Dr = Dr

        self.W_out2 = torch.zeros(self.N_V, self.Dr, dtype=torch.float)
        self.W_out2 = torch.nn.Parameter(self.W_out2, requires_grad=True)

    def forward(self, R_state2):
        output = torch.Tensor()
        for i in range(self.N_V):
            out = F.linear(R_state2[:, i, :].squeeze(1), self.W_out2[i, :]).view(-1, 1)
            output = torch.cat((output, out), dim=1)
        return output


class Reservoir2(nn.Module):
    def __init__(self, Dr, N_V, ltrain, W_in, R_network, rho, delta, b):
        super(Reservoir2, self).__init__()
        self.N_V = N_V
        self.Dr = Dr
        self.ltrain = ltrain

        self.W_in = W_in
        self.R_network = R_network
        self.rho = rho
        self.delta = delta
        self.b = b

        self.W_out1 = torch.zeros(self.N_V, self.Dr, dtype=torch.float)
        self.W_out1 = torch.nn.Parameter(self.W_out1, requires_grad=True)
        self.W_out2 = torch.zeros(self.N_V, self.Dr, dtype=torch.float)
        self.W_out2 = torch.nn.Parameter(self.W_out2, requires_grad=True)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        off_diag = np.ones([self.N_V, self.N_V])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

        self.embedding_dim = 100
        self.fc = nn.Linear(self.ltrain, self.embedding_dim)
        self.bn = nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)


    def forward(self, R_state1, train_data):
        output1 = torch.Tensor()
        for i in range(self.N_V):
            out1 = F.linear(R_state1[:, i, :].squeeze(1), self.W_out1[i, :]).view(-1, 1)
            output1 = torch.cat((output1, out1), dim=1)
        output1 = output1.permute(1, 0)
        x = self.fc(output1)
        x = F.relu(x)
        x = self.bn(x)
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        Input_network = gumbel_softmax(x, temperature=0.5, hard=True)
        Input_network = Input_network[:, 0].clone().reshape(self.N_V, -1)
        mask = torch.eye(self.N_V, self.N_V).bool()
        Input_network.masked_fill_(mask, 0)

        R_state2 = torch.zeros(self.ltrain, self.N_V, self.Dr)
        for i in range(1, (self.ltrain)):
            R_state2[i, :] = (torch.tanh(torch.mm(Input_network.T, self.rho * torch.mm(R_state2[i - 1, :], self.R_network))
                                        + self.delta * torch.mm(train_data[i, :].view(-1, 1), self.W_in) + self.b))

        output = torch.Tensor()
        for i in range(self.N_V):
            out = F.linear(R_state2[:, i, :].squeeze(1), self.W_out2[i, :]).view(-1, 1)
            output = torch.cat((output, out), dim=1)
        return output


