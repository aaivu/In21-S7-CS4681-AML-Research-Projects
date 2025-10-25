import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pgl.graph import Graph
import numpy as np

def _rel_mask(edges, edge_types, r):
    if edges.size == 0:
        return None
    m = (edge_types == r)
    if not m.any():
        return None
    return m

def _typed_mean_aggregate(g: Graph, x: paddle.Tensor, num_rels: int = 3):
    """Mean aggregate per relation, then sum across relations."""
    edges = g.edges
    if edges.size == 0:
        return x
    # edge_type saved in Graph.edge_feat['edge_type']
    e_types = g.edge_feat.get('edge_type', None)
    if e_types is None:
        # fallback: homogeneous
        return _hom_mean(g, x)

    e_types = e_types.astype(np.int32)
    src = paddle.to_tensor(edges[:,0], dtype='int64')
    dst = paddle.to_tensor(edges[:,1], dtype='int64')

    max_dst = int(dst.max().numpy()[0]) if dst.numel() > 0 else (x.shape[0]-1)
    out = paddle.zeros([max_dst+1, x.shape[1]], dtype=x.dtype)

    for r in range(num_rels):
        m = _rel_mask(edges, e_types, r)
        if m is None:
            continue
        midx = paddle.to_tensor(np.where(m)[0], dtype='int64')
        r_src = paddle.gather(src, midx)
        r_dst = paddle.gather(dst, midx)
        msg = x[r_src]
        # sum to dst for this relation
        buf = paddle.zeros_like(out)
        buf = paddle.scatter(buf, r_dst.unsqueeze(1).expand([-1, x.shape[1]]), msg, overwrite=False)
        # degree per node for this relation
        ones = paddle.ones([r_dst.shape[0], 1], dtype=x.dtype)
        deg = paddle.zeros([max_dst+1, 1], dtype=x.dtype)
        deg = paddle.scatter(deg, r_dst.unsqueeze(1), ones, overwrite=False)
        deg = paddle.clip(deg, min=1.0)
        out = out + buf / deg

    # pad to all nodes
    if out.shape[0] < x.shape[0]:
        pad = paddle.zeros([x.shape[0]-out.shape[0], x.shape[1]], dtype=x.dtype)
        out = paddle.concat([out, pad], axis=0)
    return out

def _hom_mean(g: Graph, x: paddle.Tensor):
    edges = g.edges
    if edges.size == 0:
        return x
    src = paddle.to_tensor(edges[:,0], dtype='int64')
    dst = paddle.to_tensor(edges[:,1], dtype='int64')
    msg = x[src]
    max_dst = int(dst.max().numpy()[0]) if dst.numel() > 0 else (x.shape[0]-1)
    out = paddle.zeros([max_dst+1, x.shape[1]], dtype=x.dtype)
    out = paddle.scatter(out, dst.unsqueeze(1).expand([-1, x.shape[1]]), msg, overwrite=False)
    ones = paddle.ones([dst.shape[0], 1], dtype=x.dtype)
    cnt = paddle.zeros([max_dst+1, 1], dtype=x.dtype)
    cnt = paddle.scatter(cnt, dst.unsqueeze(1), ones, overwrite=False)
    cnt = paddle.clip(cnt, min=1.0)
    out = out / cnt
    if out.shape[0] < x.shape[0]:
        pad = paddle.zeros([x.shape[0]-out.shape[0], x.shape[1]], dtype=x.dtype)
        out = paddle.concat([out, pad], axis=0)
    return out

class GNNModel(nn.Layer):
    """
    Iter-2: H-UniMP core (relation-aware SAGE projections)
    - Per-relation linear projections; typed mean-aggregation; residual stack.
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=2,
                 hidden_size=256,
                 drop=0.5,
                 m2v_dim=512,
                 edge_type=3,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.num_rels = edge_type
        self.drop = drop

        self.m2v_proj  = nn.Linear(m2v_dim, input_size)
        self.label_emb = nn.Embedding(num_class, input_size)
        self.label_mlp = nn.Sequential(
            nn.Linear(2*input_size, hidden_size),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_size, input_size)
        )

        # per-relation projections (input space) then typed aggregate
        self.rel_proj = nn.LayerList([nn.Linear(input_size, input_size) for _ in range(self.num_rels)])

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
        f_lbl = paddle.gather(feat, label_idx)
        y_emb = self.label_emb(label_y)
        z = paddle.concat([y_emb, f_lbl], axis=1)
        z = self.label_mlp(z)
        return paddle.scatter(feat, label_idx, z, overwrite=True)

    def forward(self, graph_list, feature, m2v_feature, label_y, label_idx):
        x = feature + self.m2v_proj(m2v_feature)
        x = self._inject_labels(x, label_y, label_idx)

        # inner-most subgraph
        g = graph_list[-1][0] if isinstance(graph_list[-1], (tuple, list)) else graph_list[-1]
        g = g.numpy() if hasattr(g, "numpy") else g

        # project per relation in input space and aggregate
        px = []
        for r in range(self.num_rels):
            px.append(self.rel_proj[r](x))
        # temporarily overwrite x with sum of typed aggregates
        agg_sum = _typed_mean_aggregate(g, px[0], self.num_rels)
        # (cheap trick) re-run with each projection and sum their aggregates
        for r in range(1, self.num_rels):
            agg_sum = agg_sum + _typed_mean_aggregate(g, px[r], self.num_rels)

        h = self.inp(x + agg_sum)
        h = F.relu(self.norms[0](h))
        h = self.dropout(h)
        for lin, bn in zip(self.blocks, self.norms[1:]):
            u = F.relu(bn(lin(h)))
            h = h + self.dropout(u)
        return self.out(h)
