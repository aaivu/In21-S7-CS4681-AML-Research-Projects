# temporal_tasks_with_memory.py
import math
import torch
from torch.utils import data as torch_data
import torch.nn.functional as F

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.TemporalKnowledgeGraphCompletionMemory")
class TemporalKnowledgeGraphCompletionMemory(tasks.Task, core.Configurable):
    """
    Temporal Knowledge Graph Completion with memory-augmented NBFNet.
    Uses fact_graph (with edge_time) for negatives / filtering,
    and a TGN-style memory inside the model.
    """

    _option_members = ["criterion", "metric"]

    def __init__(self, model, criterion="bce",
                 metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=5, strict_negative=True, filtered_ranking=True,
                 full_batch_eval=True,
                 debug=False):
        super().__init__()
        self.model = model
        self.criterion = {"bce": 1.0} if isinstance(criterion, str) else criterion
        self.metric = metric
        self.num_negative = num_negative
        self.strict_negative = strict_negative
        self.filtered_ranking = filtered_ranking
        self.full_batch_eval = full_batch_eval
        self.debug = debug
        self.progress = 0

    def preprocess(self, train_set, valid_set, test_set):
        dataset = train_set.dataset if isinstance(train_set, torch_data.Subset) else train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("fact_graph", dataset.graph)  # carries edge_time
        return train_set, valid_set, test_set

    def reset(self):
        """Reset model memory between epochs."""
        if hasattr(self.model, "memory"):
            self.model.memory.reset_memory()

    @torch.no_grad()
    def _strict_negative(self, pos_h, pos_t, pos_r, pos_time):
        """
        Time-aware negatives that avoid any fact (h, t', r) or (h', t, r)
        that existed BEFORE OR AT pos_time.
        """
        B = len(pos_h)
        any = -torch.ones_like(pos_h)

        # Tail negatives
        pattern_t = torch.stack([pos_h, any, pos_r], dim=-1)
        edge_index, num_t_truth = self.fact_graph.match(pattern_t)
        t_truth = self.fact_graph.edge_list[edge_index, 1]
        t_time = self.fact_graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_t_truth)
        valid_truth = t_time <= pos_time[pos_index]

        t_mask = torch.ones(B, self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index[valid_truth], t_truth[valid_truth]] = 0
        t_mask[torch.arange(B, device=self.device), pos_h] = 0  # avoid self-loops
        t_cand = t_mask.nonzero()[:, 1]
        t_num = t_mask.sum(dim=-1)
        neg_t = functional.variadic_sample(t_cand, t_num, self.num_negative)

        # Head negatives
        pattern_h = torch.stack([any, pos_t, pos_r], dim=-1)
        edge_index, num_h_truth = self.fact_graph.match(pattern_h)
        h_truth = self.fact_graph.edge_list[edge_index, 0]
        h_time = self.fact_graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_h_truth)
        valid_truth = h_time <= pos_time[pos_index]

        h_mask = torch.ones(B, self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index[valid_truth], h_truth[valid_truth]] = 0
        h_mask[torch.arange(B, device=self.device), pos_t] = 0
        h_cand = h_mask.nonzero()[:, 1]
        h_num = h_mask.sum(dim=-1)
        neg_h = functional.variadic_sample(h_cand, h_num, self.num_negative)
        # print("88 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

        # Half tail, half head
        return torch.cat([neg_t, neg_h], dim=0)

    def predict(self, batch, all_loss=None, metric=None):
        """
        batch: LongTensor [B, 4] -> (h, t, r, time)
        returns logits: [B, K+1]
        """
        if isinstance(batch, list):
            batch = torch.stack(batch, dim=0)
        pos_h, pos_t, pos_r, pos_time = batch.t()
        B = len(batch)
        graph = self.fact_graph
        # print("103 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

        # evaluation mode: rank against all entities
        if all_loss is None:
            all_index = torch.arange(graph.num_node, device=self.device)
            preds_tail, preds_head = [], []
            # print("109 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")


            for neg_chunk in all_index.split(self.num_negative if not self.full_batch_eval else graph.num_node):
                r_index = pos_r.unsqueeze(-1).expand(-1, len(neg_chunk))
                q_time = pos_time
                h_index, t_index = torch.meshgrid(pos_h, neg_chunk)
                t_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time)
                preds_tail.append(t_pred)

            for neg_chunk in all_index.split(self.num_negative if not self.full_batch_eval else graph.num_node):
                r_index = pos_r.unsqueeze(-1).expand(-1, len(neg_chunk))
                q_time = pos_time
                t_index, h_index = torch.meshgrid(pos_t, neg_chunk)
                h_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time)
                preds_head.append(h_pred)

            return torch.stack([torch.cat(preds_tail, dim=-1),
                                torch.cat(preds_head, dim=-1)], dim=1).cpu()

        # training mode: negatives + positive
        else:
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h, pos_t, pos_r, pos_time)
            else:
                neg_index = torch.randint(self.num_entity, (2 * B, self.num_negative))
            # print("135 re Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

            # Tail: h fixed
            h_index_tail = pos_h.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index_tail = torch.cat([pos_t.unsqueeze(-1), neg_index[:B]], dim=1)
            r_index_tail = pos_r.unsqueeze(-1).repeat(1, self.num_negative + 1)
            q_time_tail = pos_time
            pred_tail = self.model(graph, h_index_tail, t_index_tail, r_index_tail, query_time=q_time_tail)

            # Head: t fixed
            t_index_head = pos_t.unsqueeze(-1).repeat(1, self.num_negative + 1)
            h_index_head = torch.cat([pos_h.unsqueeze(-1), neg_index[B:]], dim=1)
            r_index_head = pos_r.unsqueeze(-1).repeat(1, self.num_negative + 1)
            q_time_head = pos_time
            pred_head = self.model(graph, h_index_head, t_index_head, r_index_head, query_time=q_time_head)
            # print("150 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

            return torch.stack([pred_tail, pred_head], dim=1)

    def target(self, batch):
        """Produce time-filtered masks for ranking."""
        B = len(batch)
        graph = self.fact_graph
        pos_h, pos_t, pos_r, pos_time = batch.t()
        any = -torch.ones_like(pos_h)

        # Tail mask
        pattern_t = torch.stack([pos_h, any, pos_r], dim=-1)
        edge_index, num_t_truth = graph.match(pattern_t)
        t_truth = graph.edge_list[edge_index, 1]
        t_time = graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_t_truth)
        valid_truth = t_time <= pos_time[pos_index]
        t_mask = torch.ones(B, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index[valid_truth], t_truth[valid_truth]] = 0

        # Head mask
        pattern_h = torch.stack([any, pos_t, pos_r], dim=-1)
        edge_index, num_h_truth = graph.match(pattern_h)
        h_truth = graph.edge_list[edge_index, 0]
        h_time = graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_h_truth)
        valid_truth = h_time <= pos_time[pos_index]
        h_mask = torch.ones(B, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index[valid_truth], h_truth[valid_truth]] = 0

        return torch.stack([t_mask, h_mask], dim=1), torch.stack([pos_t, pos_h], dim=1)

    def evaluate(self, pred, target):
        mask, target = target
        pos_pred = pred.gather(-1, target.to(pred.device).unsqueeze(-1))
        ranking = torch.sum((pos_pred <= pred) & mask.to(pred.device), dim=-1) + 1
        ranking = torch.minimum(ranking[:, 0], ranking[:, 1])
        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                k = int(_metric.split("@")[1])
                score = (ranking <= k).float().mean()
            else:
                raise ValueError(f"Unknown metric {_metric}")
            metric[tasks._get_metric_name(_metric)] = score
        return metric

    def forward(self, batch):
        all_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        metric = {}

        # training path
        tr_logits = self.predict(batch, all_loss=all_loss)
        labels = torch.zeros_like(tr_logits)
        labels[:, :, 0] = 1.0  # positives at first column
        neg_weight = torch.ones_like(tr_logits)
        if tr_logits.size(-1) > 1:
            neg_weight[:, :, 1:] = 1.0 / (tr_logits.size(-1) - 1)

        bce = F.binary_cross_entropy_with_logits(tr_logits, labels, reduction="none")
        bce = (bce * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = bce.mean()
        # print("217 Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")

        out = {tasks._get_criterion_name("bce"): loss}
        return loss, out
