"""
作用：最终冻结的 MWP_numerical 数值预测模型。
结构：最后值残差主干、局部多尺度状态编码、未来日历查询、循环季节候选、稳健相对候选和可靠性门控。
损失：按 Validation 锁定的 MSE-Huber 联合损失；九个 domain 共用同一模型代码，仅使用各自已冻结的参数。
因果性：推理只读取历史数值和已知未来日历，不读取未来目标、文本、Test 统计量或 domain 专用结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.numerical_core import CalendarEncoder, LastResidualState, LocalStateEncoder, StateDecoder
from layers.numerical_branches import CyclicSeasonalMemoryBranch, RobustRelativeBranch


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.label_len = int(configs.label_len)
        self.frequency_name = str(configs.frequency_name)
        self.cosine_eta_min_ratio = float(configs.cosine_eta_min_ratio)
        self.total_correction_clip = float(configs.total_correction_clip)

        self.shock_anchor_mode = str(
            getattr(configs, "shock_anchor_mode", "off")
        ).lower()

        self.shock_anchor_tau = float(
            getattr(configs, "shock_anchor_tau", 6.0)
        )

        if self.shock_anchor_mode not in {"off", "median", "winsor"}:
            raise ValueError(
                "shock_anchor_mode must be off/median/winsor"
            )

        if self.shock_anchor_tau <= 0.0:
            raise ValueError("shock_anchor_tau must be positive")
        self.candidate_loss_weight = float(configs.candidate_loss_weight)
        self.reliability_loss_weight = float(configs.reliability_loss_weight)
        self.safety_loss_weight = float(configs.safety_loss_weight)
        self.oracle_temperature = float(configs.oracle_temperature)
        self.oracle_margin = float(configs.oracle_margin)
        self.safety_margin = float(configs.safety_margin)
        self.correction_penalty = float(configs.correction_penalty)
        self.gate_penalty = float(configs.gate_penalty)
        self.seasonal_smoothness_penalty = float(configs.seasonal_smoothness_penalty)
        self._loss_cache = None

        d_model = int(configs.d_model)
        n_heads = int(configs.n_heads)
        d_ff = int(configs.d_ff)
        dropout = float(configs.dropout)
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if d_ff < d_model:
            raise ValueError("d_ff must be >= d_model")

        # 模块顺序与最终冻结实验一致，便于复现三种子结果。
        self.future_calendar_encoder = CalendarEncoder(
            self.frequency_name, d_model, dropout
        )
        self.state = LastResidualState(self.seq_len)
        self.state_encoder = LocalStateEncoder(
            d_model=d_model,
            layers=int(configs.state_layers),
            kernels=list(configs.state_kernel_sizes),
            dilations=list(configs.state_dilations),
            dropout=dropout,
        )
        self.state_decoder = StateDecoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            frequency_name=self.frequency_name,
            future_calendar_encoder=self.future_calendar_encoder,
        )
        self.stats_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.seasonal_memory = CyclicSeasonalMemoryBranch(
            d_model=d_model,
            dropout=dropout,
            frequency_name=self.frequency_name,
            gate_max=float(configs.gate_max),
            gate_bias=float(configs.gate_bias),
            clip_scale=float(configs.seasonal_clip_scale),
        )
        self.robust_relative = RobustRelativeBranch(
            d_model=d_model,
            n_heads=n_heads,
            state_layers=int(configs.branch_state_layers),
            kernels=list(configs.state_kernel_sizes),
            dilations=list(configs.state_dilations),
            dropout=dropout,
            frequency_name=self.frequency_name,
            gate_max=float(configs.gate_max),
            gate_bias=float(configs.gate_bias),
            clip_factor=float(configs.relative_clip_factor),
            scale_cap=float(configs.relative_scale_cap),
        )

        self.loss_mode = str(configs.loss_mode)
        self.hybrid_alpha = float(configs.hybrid_alpha)
        self.huber_delta = float(configs.huber_delta)
        self.safety_mae_ratio = float(configs.safety_mae_ratio)
        self.mae_safety_margin = float(configs.mae_safety_margin)
        self.active_domain_for_parameters = str(configs.active_domain_for_parameters)
        if self.active_domain_for_parameters != str(configs.domain):
            raise ValueError(
                "MWP_numerical parameter domain {} does not match run domain {}".format(
                    self.active_domain_for_parameters, configs.domain
                )
            )
        if self.loss_mode not in {
            "mse_reference", "hybrid_plain", "hybrid_plain_candidate",
            "hybrid_plain_candidate_oracle", "hybrid_full", "huber_full",
        }:
            raise ValueError("Unsupported frozen loss_mode {!r}".format(self.loss_mode))
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1]")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")

    def configure_optimizers(self, training):
        learning_rate = float(training["learning_rate"])
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(training["train_epochs"])),
            eta_min=learning_rate * self.cosine_eta_min_ratio,
        )
        return optimizer, scheduler

    @staticmethod
    def _validate_numeric_only(batch):
        mask = batch.get("text_mask")
        if mask is not None and bool(mask.any().item()):
            raise RuntimeError("MWP_numerical stage does not accept enabled text")
        for key in ("fact_text", "pred_text"):
            values = batch.get(key)
            if values is not None and any(value not in (None, "") for value in values):
                raise RuntimeError("MWP_numerical stage received non-empty {}".format(key))

    def _future_mark(self, y_mark):
        if y_mark is None or y_mark.ndim != 3 or y_mark.size(1) < self.pred_len + 1:
            raise ValueError("y_mark must include observed context and future pred_len")
        return y_mark[:, -self.pred_len:, :]

    def _stats_embedding(self, state, robust_scale):
        stats = torch.cat(
            (
                state["last"],
                state["window_mean"],
                state["window_std"],
                torch.log1p(robust_scale),
                state["recent_jump"] / robust_scale.clamp_min(0.05),
            ),
            dim=-1,
        )
        return self.stats_projection(stats).expand(-1, self.pred_len, -1)

    def _apply_shock_anchor(self, x_num, state):
        """
        只使用当前已经观测到的 input window 检测最后一个点是否为孤立 shock。

        off:
            原始 LastValue anchor，不做任何修改。

        median:
            若最后一点 robust z-score > tau，
            用前面历史点的 median 替换最后一点。

        winsor:
            若检测到 shock，只把最后一点截断到
            median +/- tau * robust_sigma。

        不读取未来值。
        """
        if self.shock_anchor_mode == "off":
            return state

        if x_num.size(1) < 4:
            return state

        history = x_num[:, :-1, :]
        last = x_num[:, -1:, :]

        median = history.median(
            dim=1, keepdim=True
        ).values

        mad = (
            history - median
        ).abs().median(
            dim=1, keepdim=True
        ).values

        robust_sigma = (
            1.4826 * mad
        ).clamp_min(0.05)

        score = (
            last - median
        ).abs() / robust_sigma

        active = (
            score > self.shock_anchor_tau
        ).to(dtype=x_num.dtype)

        if self.shock_anchor_mode == "median":
            robust_anchor = median

        elif self.shock_anchor_mode == "winsor":
            limit = (
                self.shock_anchor_tau *
                robust_sigma
            )

            robust_anchor = median + (
                last - median
            ).clamp(
                min=-limit,
                max=limit
            )

        else:
            return state

        anchor = (
            active * robust_anchor +
            (1.0 - active) * last
        )

        # 对检测出的孤立 shock，只修正最后一个观测值。
        # 然后重新构造 residual state，避免极端值继续污染编码器。
        clean_x = x_num.clone()

        clean_x[:, -1:, :] = anchor

        clean_state = self.state(
            clean_x,
            self.pred_len
        )

        clean_state["shock_score"] = score
        clean_state["shock_active"] = active

        return clean_state

    def _numeric_forward(self, x_num, y_mark):
        if x_num.ndim != 3 or tuple(x_num.shape[1:]) != (self.seq_len, 1):
            raise ValueError("Expected x_num [B, {}, 1]".format(self.seq_len))
        future_mark = self._future_mark(y_mark)
        state = self.state(x_num, self.pred_len)
        state = self._apply_shock_anchor(x_num, state)
        base_tokens = self.state_encoder(state["history"])
        base_residual, base_context, _ = self.state_decoder(
            base_tokens, self.pred_len, future_mark
        )
        base_prediction = state["future_base"] + base_residual

        robust_scale = self.robust_relative.robust_scale(x_num).clamp_max(
            self.robust_relative.scale_cap
        )
        stats_embedding = self._stats_embedding(state, robust_scale)

        seasonal_candidate, seasonal_gate, seasonal_probability, _, _ = self.seasonal_memory(
            base_tokens,
            base_context,
            stats_embedding,
            self.pred_len,
            y_mark,
            future_mark,
        )
        relative_candidate, relative_gate, relative_probability, _, _, scale = self.robust_relative(
            x_num,
            state,
            base_context,
            stats_embedding,
            self.pred_len,
            future_mark,
        )

        total = seasonal_gate * seasonal_candidate + relative_gate * relative_candidate
        total = self.total_correction_clip * torch.tanh(
            total / max(self.total_correction_clip, 1e-6)
        )
        prediction = base_prediction + total
        penalty = (
            self.seasonal_smoothness_penalty * self.seasonal_memory.smoothness()
            + self.correction_penalty * total.square().mean()
            + self.gate_penalty * torch.stack((seasonal_gate, relative_gate)).mean()
        )

        if self.training:
            self._loss_cache = {
                "base_prediction": base_prediction,
                "candidates": (seasonal_candidate, relative_candidate),
                "probabilities": (seasonal_probability, relative_probability),
            }
        else:
            self._loss_cache = None

        if tuple(prediction.shape[1:]) != (self.pred_len, 1):
            raise ValueError("Invalid prediction shape {}".format(tuple(prediction.shape)))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Prediction contains NaN or Inf")
        diagnostics = {
            "seasonal_gate": seasonal_gate.detach().mean(),
            "relative_gate": relative_gate.detach().mean(),
            "robust_scale": scale.detach().mean(),
        }
        return prediction, penalty, diagnostics

    def forward_batch(self, batch, decoder_input=None):
        self._validate_numeric_only(batch)
        prediction, penalty, diagnostics = self._numeric_forward(
            batch["x_num"], batch.get("y_mark")
        )
        return {
            "prediction": prediction,
            "aux_loss": penalty if self.training else prediction.new_zeros(()),
            "diagnostics": diagnostics,
        }

    def forward(self, x_num, x_mark=None, decoder_input=None, y_mark=None):
        prediction, _, _ = self._numeric_forward(x_num, y_mark)
        return prediction

    @staticmethod
    def _huber_element(error, delta):
        return F.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=float(delta)
        )

    def _joint_element(self, error, alpha):
        if alpha <= 0.0:
            return error.square()
        huber = self._huber_element(error, self.huber_delta)
        if alpha >= 1.0:
            return huber
        return (1.0 - alpha) * error.square() + alpha * huber

    def _component_alphas(self):
        if self.loss_mode == "mse_reference":
            return 0.0, 0.0, 0.0, False
        if self.loss_mode == "huber_full":
            return 1.0, 1.0, 1.0, True
        candidate_alpha = self.hybrid_alpha if self.loss_mode in {
            "hybrid_plain_candidate", "hybrid_plain_candidate_oracle", "hybrid_full"
        } else 0.0
        oracle_alpha = self.hybrid_alpha if self.loss_mode in {
            "hybrid_plain_candidate_oracle", "hybrid_full"
        } else 0.0
        return self.hybrid_alpha, candidate_alpha, oracle_alpha, self.loss_mode == "hybrid_full"

    def compute_loss(self, prediction, target, batch, stage):
        plain_alpha, candidate_alpha, oracle_alpha, dual_safety = self._component_alphas()
        plain = self._joint_element(prediction - target, plain_alpha).mean()
        if stage != "train" or self._loss_cache is None:
            return plain

        base = self._loss_cache["base_prediction"]
        candidates = self._loss_cache["candidates"]
        probabilities = self._loss_cache["probabilities"]
        candidate_loss = torch.stack([
            self._joint_element(base.detach() + candidate - target, candidate_alpha).mean()
            for candidate in candidates
        ]).mean()

        base_oracle_error = self._joint_element(base.detach() - target, oracle_alpha)
        reliability_losses = []
        for candidate, probability in zip(candidates, probabilities):
            candidate_oracle_error = self._joint_element(
                base.detach() + candidate.detach() - target, oracle_alpha
            )
            advantage = base_oracle_error - candidate_oracle_error - self.oracle_margin
            oracle = torch.sigmoid(
                advantage / max(self.oracle_temperature, 1e-6)
            ).detach()
            reliability_losses.append(
                F.binary_cross_entropy(
                    probability.clamp(1e-6, 1.0 - 1e-6), oracle
                )
            )
        reliability = torch.stack(reliability_losses).mean()

        final_error = prediction - target
        base_error = base.detach() - target
        mse_safety = torch.relu(
            final_error.square() - base_error.square().detach() - self.safety_margin
        ).mean()
        if dual_safety:
            mae_safety = torch.relu(
                final_error.abs() - base_error.abs().detach() - self.mae_safety_margin
            ).mean()
            safety = (
                (1.0 - self.safety_mae_ratio) * mse_safety
                + self.safety_mae_ratio * mae_safety
            )
        else:
            safety = mse_safety

        return (
            plain
            + self.candidate_loss_weight * candidate_loss
            + self.reliability_loss_weight * reliability
            + self.safety_loss_weight * safety
        )
