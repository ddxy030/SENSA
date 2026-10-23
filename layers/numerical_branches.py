"""
作用：实现 MWP_numerical 的循环季节候选、稳健相对候选和样本级可靠性门控。
边界：九个 domain 共用同一结构；不读取 domain 名称、未来目标、未来文本或 Test 统计量。
组成：循环季节记忆、稳健尺度相对残差、可靠性监督与受限修正。
"""

import torch
import torch.nn as nn

from layers.numerical_core import HorizonQuery, LocalStateEncoder, zero_last_linear


class ReliabilityGate(nn.Module):
    """根据当前样本、候选预测和数值主干上下文预测候选可靠性。"""

    def __init__(self, d_model, gate_max, gate_bias):
        super().__init__()
        self.gate_max = float(gate_max)
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("gate_max must be in (0, 1]")
        self.candidate_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.network = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, float(gate_bias))

    def forward(self, query, candidate_context, base_context, stats_embedding, candidate):
        candidate_embedding = self.candidate_projection(candidate)
        features = torch.cat(
            (query, candidate_context, base_context, stats_embedding, candidate_embedding),
            dim=-1,
        )
        probability = torch.sigmoid(self.network(features))
        return self.gate_max * probability, probability


class CyclicSeasonalMemoryBranch(nn.Module):
    """从 Train 标签学习统一 366 槽循环记忆，预测未来日历相对当前日期的季节偏移。"""

    _CALENDAR = {
        "monthly": (0, 11.0, 12),
        "weekly": (1, 52.0, 53),
        "daily": (2, 365.0, 366),
    }

    def __init__(self, d_model, dropout, frequency_name, gate_max, gate_bias, clip_scale):
        super().__init__()
        if frequency_name not in self._CALENDAR:
            raise ValueError("Unknown frequency_name {!r}".format(frequency_name))
        self.channel, self.denominator, self.period = self._CALENDAR[frequency_name]
        self.memory_period = 366
        self.clip_scale = float(clip_scale)
        self.memory = nn.Parameter(torch.zeros(self.memory_period, 1))
        self.query = HorizonQuery(d_model, frequency_name, dropout)
        self.context = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.gate = ReliabilityGate(d_model, gate_max, gate_bias)

    def _read(self, mark):
        if mark.size(-1) <= self.channel:
            raise ValueError("Calendar mark lacks required channel {}".format(self.channel))
        cycle_index = torch.remainder(
            (mark[..., self.channel:self.channel + 1] + 0.5) * self.denominator,
            float(self.period),
        )
        index = cycle_index * (float(self.memory_period) / float(self.period))
        floor = torch.floor(index)
        left = floor.long().squeeze(-1)
        right = torch.remainder(left + 1, self.memory_period)
        fraction = index - floor
        return self.memory[left] * (1.0 - fraction) + self.memory[right] * fraction

    def smoothness(self):
        return (self.memory - torch.roll(self.memory, shifts=1, dims=0)).square().mean()

    def forward(self, base_tokens, base_context, stats_embedding, pred_len, full_y_mark, future_mark):
        if full_y_mark.size(1) <= int(pred_len):
            raise ValueError("y_mark must include at least one observed timestamp")
        anchor_mark = full_y_mark[:, -(int(pred_len) + 1), :]
        future_value = self._read(future_mark)
        anchor_value = self._read(anchor_mark).unsqueeze(1)
        raw = future_value - anchor_value
        candidate = self.clip_scale * torch.tanh(raw / max(self.clip_scale, 1e-6))
        query = self.query(
            base_tokens.size(0), pred_len, base_tokens.device, base_tokens.dtype, future_mark
        )
        context = self.context(torch.cat((query, candidate), dim=-1))
        gate, probability = self.gate(
            query, context, base_context, stats_embedding, candidate
        )
        return candidate, gate, probability, context, query


class RobustRelativeBranch(nn.Module):
    """用截尾局部差分尺度预测相对残差，抑制稀疏尖峰造成的尺度放大。"""

    def __init__(
        self,
        d_model,
        n_heads,
        state_layers,
        kernels,
        dilations,
        dropout,
        frequency_name,
        gate_max,
        gate_bias,
        clip_factor,
        scale_cap,
    ):
        super().__init__()
        self.clip_factor = float(clip_factor)
        self.scale_cap = float(scale_cap)
        self.encoder = LocalStateEncoder(
            d_model, state_layers, kernels, dilations, dropout
        )
        self.query = HorizonQuery(d_model, frequency_name, dropout)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        zero_last_linear(self.head)
        self.gate = ReliabilityGate(d_model, gate_max, gate_bias)

    @staticmethod
    def robust_scale(x):
        differences = x[:, 1:, :] - x[:, :-1, :]
        absolute = differences.abs()
        median = absolute.median(dim=1, keepdim=True).values
        cap = (3.0 * median).clamp_min(0.05)
        clipped = torch.minimum(absolute, cap)
        scale = torch.sqrt(clipped.square().mean(dim=1, keepdim=True) + 1e-6)
        return scale.clamp_min(0.05)

    def forward(self, x_num, state, base_context, stats_embedding, pred_len, future_mark):
        scale = self.robust_scale(x_num).clamp_max(self.scale_cap)
        relative = state["history"] / scale
        tokens = self.encoder(relative)
        query = self.query(
            x_num.size(0), pred_len, x_num.device, x_num.dtype, future_mark
        )
        context, _ = self.attention(query, tokens, tokens)
        context = self.norm(context)
        raw_normalized = self.head(context)
        bound = self.clip_factor * scale.expand(-1, int(pred_len), -1)
        candidate = bound * torch.tanh(raw_normalized / max(self.clip_factor, 1e-6))
        gate, probability = self.gate(
            query, context, base_context, stats_embedding, candidate
        )
        return candidate, gate, probability, context, query, scale
