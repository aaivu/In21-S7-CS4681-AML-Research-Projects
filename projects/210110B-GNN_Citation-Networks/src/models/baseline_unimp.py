import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from pgl.graph import Graph

def _safe_mean_aggregate(g: Graph, x: paddle.Tensor):
    """Homogeneous mean aggregation over incoming edges on the current subgraph."""
    # g.edges is numpy [E,2] with (src, dst)
    edges = g.edges
    if edges.size == 0:
        return x
    src = paddle.to_tensor(edges[:,0], dtype='int64')
    dst = paddle.to_tensor(edges[:,1], dtype='int64')
    msg = x[src]                                 # [E, F]
    # sum per dst (segment sum)
    max_dst = int(dst.max().numpy()[0]) if dst.numel() > 0 else (x.shape[0]-1)
    out = paddle.zeros([max_dst+1, x.shape[1]], dtype=x.dtype)
    out = paddle.scatter(out, dst.unsqueeze(1).expand([-1, x.shape[1]]), msg, overwrite=False)
    # count per node
    ones = paddle.ones([dst.shape[0], 1], dtype=x.dtype)
    cnt = paddle.zeros([max_dst+1, 1], dtype=x.dtype)
    cnt = paddle.scatter(cnt, dst.unsqueeze(1), ones, overwrite=False)
    cnt = paddle.clip(cnt, min=1.0)
    out = out / cnt
    # pad to all nodes
    if out.shape[0] < x.shape[0]:
        pad = paddle.zeros([x.shape[0]-out.shape[0], x.shape[1]], dtype=x.dtype)
        out = paddle.concat([out, pad], axis=0)
    return out

class GNNModel(nn.Layer):
    """
    Iter-1: UniMP-lite (homogeneous)
    - Inputs: (graph_list, feature, m2v_feature, label_y, label_idx)
    - Uses inner-most subgraph for local aggregation
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=2,
                 hidden_size=256,
                 drop=0.5,
                 m2v_dim=512,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.drop = drop

        # m2v projection and label embedding (UniMP-style label injection)
        self.m2v_proj   = nn.Linear(m2v_dim, input_size)
        self.label_emb  = nn.Embedding(num_class, input_size)
        self.label_mlp  = nn.Sequential(
            nn.Linear(2*input_size, hidden_size),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_size, input_size)
        )

        # shallow “transformer-less” stack (CPU-safe)
        self.inp = nn.Linear(input_size, hidden_size)
        self.blocks = nn.LayerList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norms  = nn.LayerList([nn.BatchNorm1D(hidden_size) for _ in range(num_layers+1)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_size, num_class)
        )
        self.dropout = nn.Dropout(drop)

    def _inject_labels(self, feat, label_y, label_idx):
        if label_idx.shape[0] == 0:
            return feat
        # gather current node feats at labeled indices
        f_lbl = paddle.gather(feat, label_idx)
        y_emb = self.label_emb(label_y)
        z = paddle.concat([y_emb, f_lbl], axis=1)
        z = self.label_mlp(z)
        # write back
        return paddle.scatter(feat, label_idx, z, overwrite=True)

    def forward(self, graph_list, feature, m2v_feature, label_y, label_idx):
        # unify features + m2v
        x = feature + self.m2v_proj(m2v_feature)

        # UniMP-style label injection at input
        x = self._inject_labels(x, label_y, label_idx)

        # pick inner-most subgraph (the last in list)
        g = graph_list[-1][0] if isinstance(graph_list[-1], (tuple, list)) else graph_list[-1]
        # 1 hop mean aggregate
        agg = _safe_mean_aggregate(g.numpy(), x) if isinstance(g, Graph) else _safe_mean_aggregate(g[0], x)

        h = self.inp(x + agg)         # simple residual: input + agg
        h = F.relu(self.norms[0](h))
        h = self.dropout(h)

        for i, (lin, bn) in enumerate(zip(self.blocks, self.norms[1:])):
            u = lin(h)
            u = F.relu(bn(u))
            h = h + self.dropout(u)    # residual

        return self.out(h)
