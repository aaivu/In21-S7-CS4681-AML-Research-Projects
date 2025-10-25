import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pgl.graph import Graph


def _extract_graph(graph_like):
    """Return a PGL Graph no matter which tensor form was passed in."""
    if isinstance(graph_like, Graph):
        return graph_like
    if isinstance(graph_like, (tuple, list)) and len(graph_like) > 0:
        return _extract_graph(graph_like[0])
    if hasattr(graph_like, "numpy"):
        return graph_like.numpy()
    return graph_like


def _relation_mean(graph: Graph, features: paddle.Tensor, rel_id: int):
    """Mean aggregate messages restricted to a relation id."""
    edges = graph.edges
    if edges.size == 0:
        return paddle.zeros([features.shape[0], features.shape[1]], dtype=features.dtype)

    edge_types = graph.edge_feat.get("edge_type", None)
    if edge_types is None:
        mask = np.ones(edges.shape[0], dtype=bool)
    else:
        edge_types = edge_types.astype(np.int32)
        mask = (edge_types == rel_id)

    if not mask.any():
        return paddle.zeros([features.shape[0], features.shape[1]], dtype=features.dtype)

    edge_ids = np.where(mask)[0]
    src = paddle.to_tensor(edges[edge_ids, 0], dtype="int64")
    dst = paddle.to_tensor(edges[edge_ids, 1], dtype="int64")
    msg = paddle.gather(features, src)

    num_nodes = getattr(graph, "num_nodes", features.shape[0])
    if isinstance(num_nodes, paddle.Tensor):
        num_nodes = int(num_nodes.numpy()[0])
    num_nodes = max(num_nodes, features.shape[0])

    agg = paddle.zeros([num_nodes, features.shape[1]], dtype=features.dtype)
    agg = paddle.scatter(agg, dst.unsqueeze(1).expand([-1, features.shape[1]]), msg, overwrite=False)

    ones = paddle.ones([dst.shape[0], 1], dtype=features.dtype)
    deg = paddle.zeros([num_nodes, 1], dtype=features.dtype)
    deg = paddle.scatter(deg, dst.unsqueeze(1), ones, overwrite=False)
    deg = paddle.clip(deg, min=1.0)
    agg = agg / deg

    if agg.shape[0] < features.shape[0]:
        pad = paddle.zeros([features.shape[0] - agg.shape[0], features.shape[1]], dtype=features.dtype)
        agg = paddle.concat([agg, pad], axis=0)
    return agg


def _hom_mean(graph: Graph, features: paddle.Tensor):
    """Fallback mean aggregation when no relation ids are present."""
    edges = graph.edges
    if edges.size == 0:
        return paddle.zeros_like(features)

    src = paddle.to_tensor(edges[:, 0], dtype="int64")
    dst = paddle.to_tensor(edges[:, 1], dtype="int64")
    msg = paddle.gather(features, src)

    num_nodes = getattr(graph, "num_nodes", features.shape[0])
    if isinstance(num_nodes, paddle.Tensor):
        num_nodes = int(num_nodes.numpy()[0])
    num_nodes = max(num_nodes, features.shape[0])

    agg = paddle.zeros([num_nodes, features.shape[1]], dtype=features.dtype)
    agg = paddle.scatter(agg, dst.unsqueeze(1).expand([-1, features.shape[1]]), msg, overwrite=False)

    ones = paddle.ones([dst.shape[0], 1], dtype=features.dtype)
    deg = paddle.zeros([num_nodes, 1], dtype=features.dtype)
    deg = paddle.scatter(deg, dst.unsqueeze(1), ones, overwrite=False)
    deg = paddle.clip(deg, min=1.0)
    agg = agg / deg

    if agg.shape[0] < features.shape[0]:
        pad = paddle.zeros([features.shape[0] - agg.shape[0], features.shape[1]], dtype=features.dtype)
        agg = paddle.concat([agg, pad], axis=0)
    return agg


class GNNModel(nn.Layer):
    """
    H-UniMP mask-rate ablation head:
    - Relation-aware projections with learnable attention weights.
    - Uncertainty-gated residual updates.
    - Fixed label masking rate (p*) for ablation sweeps.
    """

    def __init__(
        self,
        input_size: int,
        num_class: int,
        num_layers: int = 2,
        hidden_size: int = 512,
        drop: float = 0.5,
        m2v_dim: int = 512,
        edge_type: int = 3,
        mask_rate: float = 0.2,
        use_label_gate: bool = True,
        use_residual_gate: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_rels = edge_type
        self.dropout = nn.Dropout(drop)
        self.mask_rate = float(np.clip(mask_rate, 0.0, 1.0))
        self.use_label_gate = use_label_gate
        self.use_residual_gate = use_residual_gate

        # Feature preparation
        self.m2v_proj = nn.Linear(m2v_dim, input_size)
        self.label_emb = nn.Embedding(num_class, input_size)
        self.label_mlp = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size, input_size),
        )

        if self.use_label_gate:
            self.label_gate = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Sigmoid(),
            )
        else:
            self.label_gate = None

        # Relation-aware projection weights
        self.rel_proj = nn.LayerList([nn.Linear(input_size, input_size) for _ in range(self.num_rels)])
        attn_scale = 1.0 / math.sqrt(max(input_size, 1))
        self.rel_scores = paddle.create_parameter(
            shape=[self.num_rels],
            dtype="float32",
            default_initializer=nn.initializer.Constant(attn_scale),
        )

        # Encoder stack
        self.inp = nn.Linear(input_size, hidden_size)
        self.blocks = nn.LayerList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norms = nn.LayerList([nn.BatchNorm1D(hidden_size) for _ in range(num_layers + 1)])

        if self.use_residual_gate:
            self.res_gates = nn.LayerList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        else:
            self.res_gates = None

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size, num_class),
        )

    def _inject_labels(self, feat, label_y, label_idx):
        if label_idx.shape[0] == 0:
            return feat

        f_lbl = paddle.gather(feat, label_idx)
        y_emb = self.label_emb(label_y)
        z = paddle.concat([y_emb, f_lbl], axis=1)
        z = self.label_mlp(z)

        keep_prob = 1.0 - self.mask_rate
        if keep_prob < 1.0:
            if self.training:
                keep = paddle.bernoulli(paddle.full([z.shape[0], 1], keep_prob, dtype=z.dtype))
            else:
                keep = paddle.full([z.shape[0], 1], keep_prob, dtype=z.dtype)
            z = z * keep

        if self.label_gate is not None:
            gate = self.label_gate(z)
            z = z * gate

        return paddle.scatter(feat, label_idx, z, overwrite=True)

    def _aggregate(self, graph_like, x):
        graph = _extract_graph(graph_like)
        if not isinstance(graph, Graph) or not hasattr(graph, "edges"):
            return paddle.zeros_like(x)

        if graph.edge_feat.get("edge_type", None) is None:
            return _hom_mean(graph, x)

        weights = F.softmax(self.rel_scores, axis=0)
        agg = paddle.zeros_like(x)
        for rel_id in range(self.num_rels):
            rel_feat = self.rel_proj[rel_id](x)
            rel_msg = _relation_mean(graph, rel_feat, rel_id)
            agg = agg + rel_msg * weights[rel_id]
        return agg

    def forward(self, graph_list, feature, m2v_feature, label_y, label_idx):
        x = feature + self.m2v_proj(m2v_feature)
        x = self._inject_labels(x, label_y, label_idx)

        g = graph_list[-1]
        aggregated = self._aggregate(g, x)

        h = self.inp(x + aggregated)
        h = F.relu(self.norms[0](h))
        h = self.dropout(h)

        for idx, (lin, bn) in enumerate(zip(self.blocks, self.norms[1:])):
            update = lin(h)
            update = F.relu(bn(update))
            if self.res_gates is not None:
                gate = F.sigmoid(self.res_gates[idx](h))
                update = update * gate
            h = h + self.dropout(update)

        return self.out(h)
