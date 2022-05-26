import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)

        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output

### MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(9))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GraphGIN(nn.Module):
    def __init__(self, input_dim, out_feature_list, final_dropout, num_layers = 7, num_mlp_layers = 2):
        '''
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        '''

        super(GraphGIN, self).__init__()

        self.final_dropout = nn.Dropout(final_dropout)
        self.num_layers = num_layers
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        self.linear = nn.Linear(input_dim, out_feature_list[0])

        ###by linear function above, mlps are needed in hidden layers
        for layer in range(self.num_layers-1):
            self.mlps.append(MLP(num_mlp_layers, out_feature_list[0], out_feature_list[0], out_feature_list[0]))
            self.batch_norms.append(nn.BatchNorm1d(9))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linear_m = nn.Linear(out_feature_list[0], out_feature_list[0])
        self.linear_f = nn.Linear(out_feature_list[0], out_feature_list[1])

    def next_layer_eps(self, h, layer, adj):
        ###pooling neighboring nodes.
        if (layer == 0):
            pooled = torch.stack([self.linear(h) for _ in range(adj.size(1))], 1)
        else:
            pooled = torch.stack([h for _ in range(adj.size(1))], 1)
        
        pooled = torch.einsum('bijk,bikl->bijl', (adj, pooled))

        #Reweights the center node representation when aggregating it with its neighbors
        if (layer == 0):
            pooled = torch.sum(pooled, 1) + (1 + self.eps[layer]) * self.linear(h)
        else:
            pooled = torch.sum(pooled, 1) + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    def forward(self, input, adj, activation=None):
        h = input

        for layer in range(self.num_layers-1):
            h = self.next_layer_eps(h, layer, adj)

        output = torch.stack([self.linear_f(h) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + (1 + self.eps[layer]) * self.linear_f(h)
        output = activation(output) if activation is not None else output
        output = self.final_dropout(output)

        return output