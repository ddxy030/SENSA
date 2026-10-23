"""
作用：实现 MWP_numerical 数值主干的最后值残差状态、未来日历编码、逐预测步查询和局部状态编码。
原则：只使用统一冻结数据管线提供的历史数值与已知未来日期；不读取 domain ID、未来目标或未来文本。
统一性：月、周、日使用同一网络结构与参数，仅由 frozen 协议注入窗口长度和日历维度。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_last_linear(module):
    """把模块中最后一个 Linear 初始化为零，使新增修正分支初始严格为零。"""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return
    raise ValueError("No Linear layer found for zero initialization")


class CalendarEncoder(nn.Module):
    """把 frozen x_mark/y_mark 转换为固定维度的周期日历表示。"""

    MAX_MARK_DIM = 3
    _SEMANTICS = {
        "monthly": ((11.0,), (12.0,)),
        "weekly": ((30.0, 52.0), (31.0, 53.0)),
        "daily": ((6.0, 30.0, 365.0), (7.0, 31.0, 366.0)),
    }

    def __init__(self, frequency_name, d_model, dropout):
        super().__init__()
        if frequency_name not in self._SEMANTICS:
            raise ValueError("Unknown frequency_name {!r}".format(frequency_name))
        denominators, periods = self._SEMANTICS[frequency_name]
        denominator = torch.ones(self.MAX_MARK_DIM)
        period = torch.ones(self.MAX_MARK_DIM)
        mask = torch.zeros(self.MAX_MARK_DIM)
        for index, (denom, cycle) in enumerate(zip(denominators, periods)):
            denominator[index] = float(denom)
            period[index] = float(cycle)
            mask[index] = 1.0
        self.register_buffer("denominator", denominator, persistent=False)
        self.register_buffer("period", period, persistent=False)
        self.register_buffer("channel_mask", mask, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(self.MAX_MARK_DIM * 4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, marks):
        if marks is None or marks.ndim != 3 or marks.size(-1) > self.MAX_MARK_DIM:
            raise ValueError("Calendar marks must be [B, T, <=3]")
        padded = F.pad(marks, (0, self.MAX_MARK_DIM - marks.size(-1)))
        mask = self.channel_mask.to(dtype=marks.dtype).view(1, 1, -1)
        index = (padded + 0.5) * self.denominator.to(dtype=marks.dtype).view(1, 1, -1)
        phase = 2.0 * math.pi * index / self.period.to(dtype=marks.dtype).view(1, 1, -1)
        features = torch.cat(
            (
                padded * mask,
                torch.sin(phase) * mask,
                torch.cos(phase) * mask,
                mask.expand_as(padded),
            ),
            dim=-1,
        )
        return self.projection(features)


class HorizonQuery(nn.Module):
    """为每个预测步生成位置、频率与未来日历联合查询。"""

    FREQUENCY_CODE = {"monthly": 0.0, "weekly": 0.5, "daily": 1.0}

    def __init__(self, d_model, frequency_name, dropout, calendar_encoder=None):
        super().__init__()
        self.frequency_name = str(frequency_name)
        if self.frequency_name not in self.FREQUENCY_CODE:
            raise ValueError("Unknown frequency_name {!r}".format(frequency_name))
        self.position_projection = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.calendar_encoder = (
            calendar_encoder
            if calendar_encoder is not None
            else CalendarEncoder(self.frequency_name, d_model, dropout)
        )

    def forward(self, batch, pred_len, device, dtype, future_mark):
        pred_len = int(pred_len)
        if future_mark is None or tuple(future_mark.shape[:2]) != (batch, pred_len):
            raise ValueError("Future y_mark must be [B, pred_len, time_dim]")
        steps = torch.arange(1, pred_len + 1, device=device, dtype=dtype)
        relative = steps / float(max(1, pred_len))
        logarithmic = torch.log1p(steps) / math.log1p(float(max(1, pred_len)))
        code = torch.full_like(relative, self.FREQUENCY_CODE[self.frequency_name])
        features = torch.stack((relative, logarithmic, code), dim=-1)
        query = self.position_projection(features).unsqueeze(0).expand(batch, -1, -1)
        return query + self.calendar_encoder(future_mark)


class LastResidualState(nn.Module):
    """构造 D1 的最后值残差坐标及统一窗口统计量。"""

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = int(seq_len)

    def forward(self, x, pred_len):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, 1):
            raise ValueError("Expected x_num [B, {}, 1]".format(self.seq_len))
        last = x[:, -1:, :]
        residual = x - last
        differences = x[:, 1:, :] - x[:, :-1, :]
        local_scale = torch.sqrt(
            differences.square().mean(dim=1, keepdim=True) + 1e-6
        ).clamp_min(0.05)
        return {
            "history": residual,
            "last": last,
            "future_base": last.expand(-1, int(pred_len), -1),
            "window_mean": x.mean(dim=1, keepdim=True),
            "window_std": x.std(dim=1, keepdim=True, unbiased=False),
            "recent_jump": differences[:, -1:, :].abs(),
            "local_scale": local_scale,
            "relative_history": residual / local_scale,
        }


class _StateBlock(nn.Module):
    def __init__(self, d_model, kernels, dilations, dropout):
        super().__init__()
        if len(kernels) != len(dilations) or not kernels:
            raise ValueError("state kernels and dilations must be aligned")
        self.norm = nn.LayerNorm(d_model)
        self.branches = nn.ModuleList()
        for kernel, dilation in zip(kernels, dilations):
            kernel = int(kernel)
            dilation = int(dilation)
            if kernel <= 0 or kernel % 2 == 0 or dilation <= 0:
                raise ValueError("state kernels must be positive odd; dilations positive")
            self.branches.append(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding=((kernel - 1) * dilation) // 2,
                    groups=d_model,
                    bias=False,
                )
            )
        self.gate = nn.Conv1d(d_model * len(self.branches), d_model * 2, 1)
        self.output = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        residual = tokens
        hidden = self.norm(tokens).transpose(1, 2)
        hidden = torch.cat([branch(hidden) for branch in self.branches], dim=1)
        value, gate = self.gate(hidden).chunk(2, dim=1)
        hidden = torch.tanh(value) * torch.sigmoid(gate)
        tokens = residual + self.dropout(self.output(hidden)).transpose(1, 2)
        return tokens + self.ffn(self.ffn_norm(tokens))


class LocalStateEncoder(nn.Module):
    """可处理任意长度的一维残差序列，供主干和新增分支共享结构。"""

    def __init__(self, d_model, layers, kernels, dilations, dropout):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.position_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                _StateBlock(d_model, kernels, dilations, dropout)
                for _ in range(int(layers))
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, history):
        if history.ndim != 3 or history.size(-1) != 1:
            raise ValueError("History must be [B, T, 1]")
        batch, length, _ = history.shape
        position = torch.linspace(
            0.0,
            1.0,
            length,
            device=history.device,
            dtype=history.dtype,
        )
        position = torch.stack(
            (
                position,
                torch.log1p(position * length) / math.log1p(float(length)),
            ),
            dim=-1,
        ).view(1, length, 2).expand(batch, -1, -1)
        tokens = self.value_projection(history) + self.position_projection(position)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)


class StateDecoder(nn.Module):
    """使用未来日历查询从状态 token 解码 D1 预测残差。"""

    def __init__(
        self, d_model, n_heads, dropout, frequency_name, future_calendar_encoder=None
    ):
        super().__init__()
        self.query = HorizonQuery(
            d_model, frequency_name, dropout, calendar_encoder=future_calendar_encoder
        )
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens, pred_len, future_mark):
        queries = self.query(
            tokens.size(0), pred_len, tokens.device, tokens.dtype, future_mark
        )
        context, _ = self.attention(queries, tokens, tokens)
        context = self.norm(context)
        return self.head(context), context, queries
