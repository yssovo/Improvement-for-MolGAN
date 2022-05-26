import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, num_flow_layer, graph_size,
                 num_node_type, num_edge_type, args, nhid=128, nout=128):
        '''
        :param index_nod_edg:
        :param num_edge_type, virtual type included
        '''
        super(MaskedGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)
        self.graph_size = graph_size
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.args = args
        self.is_batchNorm = self.args.is_bn

        self.mask_node = nn.Parameter(mask_node.view(1, self.repeat_num, graph_size, 1), requires_grad=False)  # (1, repeat_num, n, 1)
        self.mask_edge = nn.Parameter(mask_edge.view(1, self.repeat_num, 1, graph_size, graph_size), requires_grad=False)  # (1, repeat_num, 1, n, n)

        self.index_select_edge = nn.Parameter(index_select_edge, requires_grad=False)  # (edge_step_length, 2)

        self.emb_size = nout
        self.hid_size = nhid
        self.num_flow_layer = num_flow_layer

        self.rgcn = RGCN(num_node_type, nhid=self.hid_size, nout=self.emb_size, edge_dim=self.num_edge_type-1,
                         num_layers=self.args.gcn_layer, dropout=0., normalization=False)

        if self.is_batchNorm:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.node_st_net = nn.ModuleList([ST_Net_Sigmoid(self.args, nout, self.num_node_type, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])

        self.edge_st_net = nn.ModuleList([ST_Net_Sigmoid(self.args, nout*3, self.num_edge_type, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])


    def forward(self, x, adj, x_deq, adj_deq):
        '''
        :param x:   (batch, N, 9)
        :param adj: (batch, 4, N, N)

        :param x_deq: (batch, N, 9)
        :param adj_deq:  (batch, edge_num, 4)
        :return:
        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        x_deq = x_deq.view(-1, self.num_node_type)  # (batch *N, 9)
        adj_deq = adj_deq.view(-1, self.num_edge_type) # (batch*(repeat_num-N), 4)


        for i in range(self.num_flow_layer):
            # update x_deq
            node_s, node_t = self.node_st_net[i](graph_emb_node)
            x_deq = x_deq * node_s + node_t

            if torch.isnan(x_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()

            # update adj_deq
            edge_s, edge_t = self.edge_st_net[i](graph_node_emb_edge)
            adj_deq = adj_deq * edge_s + edge_t

            if torch.isnan(adj_deq).any():
                raise RuntimeError(
                    'adj_deq has NaN entries after transformation at layer %d' % i)
            if i == 0:
                adj_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                adj_log_jacob += (torch.abs(edge_s) + 1e-20).log()

        x_deq = x_deq.view(batch_size, -1)  # (batch, N * 9)
        adj_deq = adj_deq.view(batch_size, -1)  # (batch, (repeat_num-N) * 4)

        x_log_jacob = x_log_jacob.view(batch_size, -1).sum(-1)  # (batch)
        adj_log_jacob = adj_log_jacob.view(batch_size, -1).sum(-1)  # (batch)
        return [x_deq, adj_deq], [x_log_jacob, adj_log_jacob]

    def _get_embs(self, x, adj):
        '''
        :param x of shape (batch, N, 9)
        :param adj of shape (batch, 4, N, N)
        :return: inputs for st_net_node and st_net_edge
        graph_emb_node of shape (batch*N, d)
        graph_emb_edge of shape (batch*(repeat-N), 3d)

        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        adj = adj[:, :3] # (batch, 3, N, N) TODO: check whether we have to use the 4-th slices(virtual bond) or not

        x = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1), torch.zeros([1]).cuda()).view(
            -1, self.graph_size, self.num_node_type)  # (batch*repeat_num, N, 9)

        adj = torch.where(self.mask_edge, adj.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1),
                          torch.zeros([1]).cuda()).view(
            -1, self.num_edge_type - 1, self.graph_size, self.graph_size)  # (batch*repeat_num, 3, N, N)

        node_emb = self.rgcn(x, adj)  # (batch*repeat_num, N, d)

        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch*repeat_num, N, d)

        node_emb = node_emb.view(batch_size, self.repeat_num, self.graph_size, -1) # (batch, repeat_num, N, d)


        graph_emb = torch.sum(node_emb, dim=2, keepdim=False) # (batch, repeat_num, d)

        #  input for st_net_node
        graph_emb_node = graph_emb[:, :self.graph_size].contiguous() # (batch, N, d)
        graph_emb_node = graph_emb_node.view(batch_size * self.graph_size, -1)  # (batch*N, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb[:, self.graph_size:].contiguous() # (batch, repeat_num-N, d)
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num-N, 1, d)

        all_node_emb_edge = node_emb[:, self.graph_size:] # (batch, repeat_num-N, N, d)

        index = self.index_select_edge.view(1, -1, 2, 1).repeat(batch_size, 1, 1,
                                        self.emb_size)  # (batch_size, repeat_num-N, 2, d)


        graph_node_emb_edge = torch.cat((torch.gather(all_node_emb_edge, dim=2, index=index), 
                                        graph_emb_edge),dim=2)  # (batch_size, repeat_num-N, 3, d)

        graph_node_emb_edge = graph_node_emb_edge.view(batch_size * (self.repeat_num - self.graph_size),
                                        -1)  # (batch_size * (repeat_num-N), 3*d)

        return graph_emb_node, graph_node_emb_edge