import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.utils import to_dense_batch


num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

class GraphTransformerPool(nn.Module):
    """
    A Transformer-based readout layer, modified to act as a pooling layer.
    
    It uses a [CLS] token, applies a Transformer Encoder, and then 
    returns the [CLS] token's embedding as the graph representation.
    """
    # CHANGED: Removed output_dim from __init__
    def __init__(self, emb_dim, n_head=8, n_layer=4, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        
        # A learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # The Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=n_head,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=n_layer
        )
        
        # --- CHANGED: REMOVED self.ffn ---
        
    def forward(self, h_nodes, data_batch):
        # h_nodes shape: [TotalNodes, emb_dim]
        # data_batch shape: [TotalNodes]
        
        dense_h, mask = to_dense_batch(h_nodes, data_batch)
        B, _L, _D = dense_h.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dense_h_with_cls = torch.cat([cls_tokens, dense_h], dim=1)
        
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=h_nodes.device)
        inverted_node_mask = ~mask
        final_mask = torch.cat([cls_mask, inverted_node_mask], dim=1)

        dense_h_with_cls = dense_h_with_cls.permute(1, 0, 2)
        
        transformer_out = self.transformer_encoder(
            dense_h_with_cls, 
            src_key_padding_mask=final_mask
        )
        
        transformer_out = transformer_out.permute(1, 0, 2)

        # h_global shape: [BatchSize, emb_dim]
        h_global = transformer_out[:, 0]  # Select the first token
        
        # --- CHANGED: Return h_global directly ---
        return h_global

class GCN(nn.Module):
    def __init__(self, task='classification', num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean', trans_n_layer=1, trans_n_head=12):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'transformer':
            self.pool = GraphTransformerPool(
                emb_dim=emb_dim, 
                n_head=trans_n_head, 
                n_layer=trans_n_layer, 
                dropout=drop_ratio
            )
        else:
            raise ValueError('Not defined pooling!')
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus(),
                nn.Linear(self.feat_dim//2, 2)
            )
        elif self.task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus(),
                nn.Linear(self.feat_dim//2, 1)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)

        return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


if __name__ == "__main__":
    model = GCN()
    print(model)
