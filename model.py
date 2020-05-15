import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        support = torch.spmm(features, self.weight)
        feat = torch.spmm(adj, support)
        if self.bias is not None:
            return feat + self.bias
        else:
            return feat

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, num_feat, num_hidden1, num_hidden2, num_class, dropout = True):
        super(GCN, self).__init__()

        self.Graphconv1 = GraphConv(num_feat, num_hidden1)
        self.Graphconv2 = GraphConv(num_hidden1, num_hidden2)
        self.Graphconv3 = GraphConv(num_hidden2, num_class)
        self.dropout = dropout

    def forward(self, feat, adj):
        feat = F.relu(self.Graphconv1(feat, adj))
        if self.dropout:
            feat = F.dropout(feat, self.dropout, training = self.training)
        feat = F.relu(self.Graphconv2(feat, adj))
        if self.dropout:
            feat = F.dropout(feat, self.dropout, training = self.training)
        feat = self.Graphconv3(feat, adj)
        return F.log_softmax(feat, dim = 1)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)