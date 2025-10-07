# temporal_nbfnet.py
from collections.abc import Sequence
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import core, layers, tasks
from torchdrug.layers import functional
from torchdrug.core import Registry as R


# ---------------------------
# Utilities
# ---------------------------

class SinusoidalTimeEncoder(nn.Module):
    """
    Classic sinusoidal encoding for scalar timestamps.
    Complexity: O(E * dim). Output shape: [E, dim].
    """
    def __init__(self, dim=32, min_scale=1.0, max_scale=10000.0):
        super().__init__()
        assert dim % 2 == 0, "time encoding dim must be even"
        self.dim = dim
        half = dim // 2
        inv_freq = torch.exp(-torch.linspace(math.log(min_scale), math.log(max_scale), steps=half))
        self.register_buffer("inv_freq", inv_freq)  # [half]

    def forward(self, t: torch.Tensor):
        # t: [E] or [E, 1]
        t = t.float().view(-1, 1)                    # [E, 1]
        ang = t * self.inv_freq.view(1, -1)          # [E, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [E, dim]
        return emb


def time_decay_weight(edge_time: torch.Tensor,
                      query_time: torch.Tensor,
                      mode: str = "exp",
                      half_life: float = 32.0,
                      window: int =5):
    """
    edge_time: [E] int
    query_time: [B] int
    returns: [E, B] in [0,1], zeroed if outside window or future
    """
    # Only use edges no later than the query time
    # broadcast to [E, B]
    dt = (query_time.unsqueeze(0) - edge_time.unsqueeze(1)).float()
    valid = dt >= 0
    if window is not None:
        valid = valid & (dt <= float(window))

    dt = torch.clamp(dt, min=0.0)
    if mode == "none":
        w = torch.ones_like(dt)
    elif mode == "linear":
        # linear decay over window; if no window, fall back to exp with long half life
        if window is None:
            w = 1.0 / (1.0 + dt / max(half_life, 1.0))
        else:
            w = torch.clamp(1.0 - dt / max(float(window), 1.0), min=0.0)
    elif mode == "exp":
        # w = 0.5 ** (dt / half_life)
        w = torch.exp(-math.log(2.0) * dt / max(half_life, 1e-6))
    else:
        raise ValueError(f"Unknown decay mode: {mode}")

    w = w * valid.float()
    return w  # [E, B]


# ---------------------------
# TGN-style Memory
# ---------------------------

class TGMemory(nn.Module):
    """
    Minimal TGN-style memory for nodes.

    - Lazy initialization: call init_memory(num_nodes, device)
    - Each node has a memory vector of size memory_dim.
    - Updates via a GRUCell from messages of size (msg_dim + time_dim).
    - time_encoder encodes Δt = event_time - last_update_time for each node.
    """
    def __init__(self, memory_dim, msg_dim, time_dim=16):
        super().__init__()
        self.memory_dim = memory_dim
        self.msg_dim = msg_dim
        self.time_dim = time_dim

        # Will be created on init_memory(...)
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.uint8))
        self.memory = None           # nn.Parameter of shape [N, memory_dim]
        self.last_update = None      # buffer [N] (float timestamps or ints)
        self.gru = nn.GRUCell(msg_dim + time_dim, memory_dim)
        self.time_encoder = SinusoidalTimeEncoder(time_dim)

    def init_memory(self, num_nodes: int, device=None):
        """
        Initialize memory for `num_nodes`. Call once after dataset known.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        # memory as parameter so optimizer will see it
        mem = torch.zeros(num_nodes, self.memory_dim, device=device)
        nn.init.xavier_uniform_(mem)
        self.memory = nn.Parameter(mem)
        last = torch.zeros(num_nodes, device=device).float()
        # store as buffer
        self.register_buffer("last_update", last)
        self._initialized.fill_(1)
        # move GRU params to device
        self.gru = self.gru.to(device)
        self.time_encoder = self.time_encoder.to(device)

    @property
    def initialized(self):
        return bool(self._initialized.item())

    def reset_memory(self):
        """
        Reinitialize memory and last_update (keeps sizes).
        """
        if not self.initialized:
            return
        with torch.no_grad():
            nn.init.xavier_uniform_(self.memory)
            self.last_update.zero_()

    def get_memory(self, nodes: torch.Tensor):
        """
        nodes: [K] long
        returns: [K, memory_dim]
        """
        assert self.initialized, "Memory not initialized. Call init_memory(num_nodes, device)."
        return self.memory[nodes]

    def compute_and_update(self, src_nodes: torch.LongTensor, dst_nodes: torch.LongTensor,
                           event_time: torch.Tensor, edge_emb: torch.Tensor):
        """
        Compute messages for src/dst and update memory in-place.

        src_nodes, dst_nodes: [B] longs (node ids) -- corresponds to positive edges
        event_time: [B] (timestamps)
        edge_emb: [B, Dmsg] (message embedding produced for that observed edge)
        """
        assert self.initialized, "Memory not initialized. Call init_memory(num_nodes, device)."
        device = self.memory.device
        B = edge_emb.size(0)
        # src updates
        if src_nodes is not None and len(src_nodes) > 0:
            src_nodes = src_nodes.to(device)
            et = event_time.to(device).float()
            last_src = self.last_update[src_nodes].to(device).float()
            dt_src = (et - last_src).clamp(min=0.0)  # [B]
            dt_emb = self.time_encoder(dt_src)       # [B, time_dim]
            msg_src = torch.cat([edge_emb.to(device), dt_emb], dim=-1)  # [B, msg+time]
            old_mem_src = self.memory[src_nodes]
            new_mem_src = self.gru(msg_src, old_mem_src)
            with torch.no_grad():
                self.memory.data[src_nodes] = new_mem_src
                self.last_update[src_nodes] = et
            # print("167 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")


        # dst updates (same edge_emb used; could be distinct)
        if dst_nodes is not None and len(dst_nodes) > 0:
            dst_nodes = dst_nodes.to(device)
            et = event_time.to(device).float()
            last_dst = self.last_update[dst_nodes].to(device).float()
            dt_dst = (et - last_dst).clamp(min=0.0)
            dt_emb = self.time_encoder(dt_dst)
            msg_dst = torch.cat([edge_emb.to(device), dt_emb], dim=-1)
            old_mem_dst = self.memory[dst_nodes]
            new_mem_dst = self.gru(msg_dst, old_mem_dst)
            with torch.no_grad():
                self.memory.data[dst_nodes] = new_mem_dst
                self.last_update[dst_nodes] = et
            # print("183 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")



# ---------------------------
# Temporal Relational Convolution
# ---------------------------

class GeneralizedRelationalConvTemporal(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
        # rotate disabled in fast path (needs complex split)
    }

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_relation,
                 query_input_dim,
                 message_func="rotate",
                 aggregate_func="pna",
                 layer_norm=False,
                 activation="relu",
                 dependent=True,
                 time_encode_dim=64,
                 time_decay="exp",
                 time_half_life=200.0,
                 time_window=None,
                 debug=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.debug = debug
        self.dropout = nn.Dropout(0.3)

        self.time_encoder = SinusoidalTimeEncoder(time_encode_dim) if time_encode_dim > 0 else None
        self.time_decay = time_decay
        self.time_half_life = time_half_life
        self.time_window = time_window

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # PNA builds stats; input doubles with boundary concat in original impl.
        # We keep same shapes and only enrich the MESSAGE with time encodings.
        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

        # to fuse time encoding with message
        if self.time_encoder is not None:
            self.time_fuse = nn.Linear(self.time_encoder.dim, input_dim, bias=True)
        else:
            self.time_fuse = None

    def _edge_filter_weights(self, graph):
        """
        Produce [E, B] weights according to edge_time vs graph.query_time,
        applying decay & window. graph must carry:
          - graph.edge_time: [E] long
          - graph.query_time: [B] long
        """
        assert hasattr(graph, "edge_time"), "graph.edge_time is required for temporal conv"
        assert hasattr(graph, "query_time"), "graph.query_time is required for temporal conv"
        w = time_decay_weight(
            graph.edge_time, graph.query_time,
            mode=self.time_decay,
            half_life=self.time_half_life,
            window=self.time_window
        )
        return w  # [E, B]

    def message(self, graph, input):
        if getattr(self, "debug", False):
            print(f"[DEBUG] GeneralizedRelationalConvTemporal.message called. input shape: {input.shape}")
        """
        input: [N, B, D]
        returns: [E, B, D] (message) + boundary handled in aggregate()
        """
        # print("[DEBUG] graph.num_relation:", graph.num_relation)
        # print("[DEBUG] self.num_relation:", self.num_relation)
        assert graph.num_relation == self.num_relation
        B = len(graph.query)

        node_in, _, relation = graph.edge_list.t()
        if self.dependent:
            # [B, R, D]
            relation_input = self.relation_linear(graph.query).view(B, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(B, -1, -1)
        # [R, B, D]
        relation_input = relation_input.transpose(0, 1)

        # Node and relation features for each edge
        node_input = input[node_in]             # [E, B, D]
        edge_input = relation_input[relation]   # [E, B, D]

        # If graph provides a memory tensor per node, fuse it.
        # graph.memory expected shape: [N, D] (node-level memory)
        if hasattr(graph, "memory") and graph.memory is not None:
            # memory[node_in]: [E, D] -> expand to [E, B, D]
            node_mem = graph.memory[node_in].unsqueeze(1)  # [E,1,D]
            node_input = node_input + node_mem  # broadcast to [E,B,D]

        if self.message_func == "transe":
            msg = edge_input + node_input
        elif self.message_func == "distmult":
            msg = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            msg_re = node_re * edge_re - node_im * edge_im
            msg_im = node_re * edge_im + node_im * edge_re
            msg = torch.cat([msg_re, msg_im], dim=-1)
        else:
            raise ValueError(f"Unknown message function `{self.message_func}`")

        # Time fusion (encoding edge_time, per-edge same across batch)
        if self.time_encoder is not None:
            te = self.time_encoder(graph.edge_time.view(-1))  # [E, Tdim]
            time_proj = self.time_fuse(te)                    # [E, D]
            msg = msg + time_proj.unsqueeze(1)                # [E, B, D] (broadcasted)

        # Per-(edge,query) temporal weight (mask + decay)
        ew = self._edge_filter_weights(graph).unsqueeze(-1)  # [E, B, 1]
        msg = msg * ew
        # print("330 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

        # Append learnable "boundary" will be handled in aggregate like original implementation
        return msg

    def aggregate(self, graph, message):
        if getattr(self, "debug", False):
            print(f"[DEBUG] GeneralizedRelationalConvTemporal.aggregate called. message : {message.shape}")
        """
        message: [E, B, D]
        returns: [N, B, ?] per original PNA or simple aggregations.
        """
        node_out = graph.edge_list[:, 1]
        # append boundary self edges
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        # edge weights: learned + implicit ones for boundary
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        # Broadcaster to [E+B, 1, 1]
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)

        # Expand message to include boundary = graph.boundary
        message = torch.cat([message, graph.boundary], dim=0)  # [(E+N), B, D]

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            maxv = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            minv = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), maxv.unsqueeze(-1), minv.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)   # [N, B, 4D]
            # degree_out + boundary
            degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)  # [N, B, 12D]
        else:
            raise ValueError(f"Unknown aggregation function `{self.aggregate_func}`")
        # print("375 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

        return update

    def message_and_aggregate(self, graph, input):
        # disable fast path in temporal setting due to per-query edge masks/weights
        return super().message_and_aggregate(graph, input)

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        output = self.dropout(output)
        return output


# ---------------------------
# Temporal NBFNet model (with memory)
# ---------------------------

@R.register("model.NBFNetTemporal")
class NeuralBellmanFordNetworkTemporal(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False,
                 num_beam=10, path_topk=10,
                 # temporal knobs
                 time_encode_dim=64, time_decay="exp", time_half_life=200.0, time_window=None,
                 debug=False,
                 # memory knobs
                 use_memory=True,
                 memory_time_dim=64):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.debug = debug

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            if self.debug:
                print(f"Initializing layer {i}: {self.dims[i]} -> {self.dims[i + 1]}")
            self.layers.append(
                GeneralizedRelationalConvTemporal(
                    self.dims[i], self.dims[i + 1], double_relation, self.dims[0],
                    message_func, aggregate_func, layer_norm, activation, dependent,
                    time_encode_dim=time_encode_dim, time_decay=time_decay,
                    time_half_life=time_half_life, time_window=time_window,
                    debug=self.debug
                )
            )

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

        # memory: lazy init (needs dataset.num_entity + device)
        self.use_memory = use_memory
        if use_memory:
            # create a TGMemory instance, but not initialized to nodes yet.
            # memory vector size = first hidden dim (so it can be added to node input)
            self.memory = TGMemory(memory_dim=128, msg_dim=self.dims[1], time_dim=memory_time_dim)
        else:
            self.memory = None

    # ---------------- temporal helpers ----------------

    def init_memory(self, num_nodes: int, device=None):
        """
        Initialize internal memory (call once, e.g. in task.preprocess).
        """
        if not self.use_memory:
            return
        if device is None:
            # attempt to infer device from model params
            device = next(self.parameters()).device
        self.memory.init_memory(num_nodes, device=device)

    def reset_memory(self):
        if self.memory is not None:
            self.memory.reset_memory()

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None, query_time=None):
        # identical to your version; temporal models still want to optionally drop trivial 1-hop edges.
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([
                    h_index.to(self.device),
                    t_index.to(self.device),
                    r_index.to(self.device)
                ], dim=-1)
            else:
                pattern = torch.stack([
                    h_index.to(self.device),
                    t_index.to(self.device)
                ], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # same as your original logic
    # Ensure all tensors are on the same device
        device = h_index.device
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True).to(device)
        h_index = h_index.to(device)
        t_index = t_index.to(device)
        r_index = r_index.to(device)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
        # unchanged except we preserve edge_time if present
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        edge_time = getattr(graph, "edge_time", None)
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=graph.device)])
            if edge_time is not None:
                loop_time = torch.full((graph.num_node,), edge_time.min(), device=self.device, dtype=edge_time.dtype)
                edge_time = torch.cat([edge_time, loop_time])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        new_graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                                num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        if edge_time is not None:
            with new_graph.edge():
                new_graph.edge_time = edge_time
        return new_graph

    def bellmanford(self, graph, h_index, r_index, query_time, separate_grad=False):
        """
        Same shape behavior as original, but we attach query_time so temporal convs
        can mask / decay edges per query.
        """
        query = self.query(r_index)                       # [B, D0]
        index = h_index.unsqueeze(-1).expand_as(query)    # [B, D0]
        # if self.debug:
        # print(h_index)
        # print(f"[DEBUG] bellmanford: h_index={h_index}, r_index.shape={r_index} , query_time={query_time}")
        # print(f"[DEBUG] bellmanford: graph.num_node={int(graph.num_node)}, query.shape={query.shape}")
        boundary = torch.zeros(int(graph.num_node), *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
            graph.query_time = query_time                # [B]
        with graph.node():
            # print(f"[DEBUG] boundary shape: {getattr(boundary, 'shape', type(boundary))}, graph.num_node: {graph.num_node}")

            graph.boundary = boundary
            # Inject node-level memory (if available and initialized)
            if self.use_memory and self.memory is not None and self.memory.initialized:
                # provide full memory [N, Dmem] to graph.node(); conv will index it
                graph.memory = self.memory.memory  # Parameter [N, D]
            else:
                object.__setattr__(graph, "memory", None)


        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            step_graph = graph.clone().requires_grad_() if separate_grad else graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    # ---------------- forward / visualize ----------------

    def forward(self, graph, h_index, t_index, r_index=None, query_time=None, all_loss=None, metric=None):
        if self.debug:
            print(f"[DEBUG] NeuralBellmanFordNetworkTemporal.forward called. h_index shape: {h_index.shape}, t_index shape: {t_index.shape}, r_index shape: {r_index.shape if r_index is not None else None}, query_time shape: {query_time.shape if query_time is not None else None}")
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index, query_time=query_time)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        assert query_time is not None, "Temporal model requires query_time [B]"

        out = self.bellmanford(graph, h_index[:, 0], r_index[:, 0], query_time=query_time)
        if self.debug:
            print(f"[DEBUG] Output from bellmanford: node_feature shape: {out['node_feature'].shape}")
        feature = out["node_feature"].transpose(0, 1)  # [B, N, F]
        if self.debug:
            print(f"[DEBUG] Feature after transpose: {feature.shape}")
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        if self.debug:
            print(f"[DEBUG] Index for gather: {index.shape}")
        feature = feature.gather(1, index)             # [B, K, F]
        if self.debug:
            print(f"[DEBUG] Feature after gather: {feature.shape}")

        # Update memory using the *positive* edges in the batch (index 0)
        # We update memory for the head and tail nodes that correspond to the positive sample only.
        # h_index and t_index are of shape [B, K] (K = 1 + negatives). The positive is at pos 0.
        try:
            if self.use_memory and self.memory is not None and self.memory.initialized:
                # positive nodes (first column)
                pos_h = h_index[:, 0].clone().detach().to(self.memory.memory.device)
                pos_t = t_index[:, 0].clone().detach().to(self.memory.memory.device)
                # choose a compact embedding to use as message - we take the corresponding features for positive
                pos_feat = feature[:, 0, :].detach()  # [B, F]
                # If memory dim differs from pos_feat dim, project to memory.msg_dim with a linear
                if pos_feat.size(-1) != self.memory.msg_dim:
                    # lazy create projector as attribute if missing
                    if not hasattr(self, "_mem_msg_proj"):
                        self._mem_msg_proj = nn.Linear(pos_feat.size(-1), self.memory.msg_dim).to(self.memory.memory.device)
                    pos_feat_proj = self._mem_msg_proj(pos_feat.to(self.memory.memory.device))
                else:
                    pos_feat_proj = pos_feat.to(self.memory.memory.device)
                # event times for update: choose query_time (length B)
                event_time = query_time.clone().to(self.memory.memory.device)
                # update memory (in-place)
                self.memory.compute_and_update(pos_h, pos_t, event_time, pos_feat_proj)
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Memory update failed: {e}")

        if self.symmetric:
            # assert (t_index[:, [0]] == t_index).all()
            out2 = self.bellmanford(graph, t_index[:, 0], r_index[:, 0], query_time=query_time)
            inv_feature = out2["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        if self.debug:
            print(f"[DEBUG] Score: {score}")
            print(f"[DEBUG] Returning score with shape: {score.view(shape).shape}")
        return score.view(shape)

    def visualize(self, graph, h_index, t_index, r_index, query_time):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index, r_index, query_time=query_time, separate_grad=True)
        feature = output["node_feature"]
        step_graphs = output["step_graphs"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = [g.edge_weight for g in step_graphs]
        edge_grads = autograd.grad(score, edge_weights)
        for g, eg in zip(step_graphs, edge_grads):
            with g.edge():
                g.edge_grad = eg
        distances, back_edges = self.beam_search_distance(step_graphs, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)
        return paths, weights



    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for graph in graphs:
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = torch.repeat_interleave(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths
