import torch
import torch.nn as nn
from torch.distributions.normal import Normal


################################################################################
# Dispatcher: sparse batching utils
################################################################################
class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()

        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


################################################################################
# Standard MLP expert
################################################################################
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


################################################################################
# LoRA Expert
################################################################################
class LoRAExpert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, rank=4):
        super().__init__()
        self.W_base = nn.Linear(input_size, hidden_size, bias=False)
        self.W_base.weight.requires_grad = False
        self.A = nn.Linear(input_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_eff = self.W_base(x) + self.B(self.A(x))
        return self.fc2(self.relu(x_eff))


################################################################################
# Modular MoE class
################################################################################
class MoE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        k=2,
        noisy_gating=True,
        mode='baseline',
        lora_rank=4
    ):
        """
        mode: 'baseline' (top-k noisy), 'switch' (top-1), 'lora', 'hybrid' (Switch+LoRA)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.k = k
        self.noisy_gating = noisy_gating
        self.mode = mode

        # experts
        if mode in ['baseline', 'switch']:
            self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        elif mode in ['lora', 'hybrid']:
            self.experts = nn.ModuleList([LoRAExpert(input_size, output_size, hidden_size, lora_rank) for _ in range(num_experts)])

        # gating parameters
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        # adjust k & noise for modes
        if mode in ['switch', 'hybrid']:
            self.k = 1
            self.noisy_gating = False

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def noisy_top_k_gating(self, x):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        k_clamped = min(int(self.k), int(self.num_experts))

        top_logits, top_indices = logits.topk(k_clamped, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]

        top_k_gates = torch.softmax(top_logits, dim=1).to(clean_logits.dtype)
        zeros = torch.zeros_like(clean_logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x)

        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates_per_expert = dispatcher.expert_to_gates()

        expert_outputs = []
        active_expert_indices = []

        for i, inp in enumerate(expert_inputs):
            if inp.size(0) > 0:  # skip empty expert batch
                expert_outputs.append(self.experts[i](inp))
                active_expert_indices.append(i)
            else:
                expert_outputs.append(torch.zeros(0, self.output_size, device=x.device))  # placeholder

        y = dispatcher.combine(expert_outputs)
        return y, loss
