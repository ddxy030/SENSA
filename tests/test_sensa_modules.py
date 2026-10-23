import torch

from layers.sensa_modules import (
    MemoryFreeConditionalStateAdapter,
    PredictiveSemanticFactorizer,
    UniformHistoricalTextAggregator,
)


def test_factorizer_shape_and_finite():
    module = PredictiveSemanticFactorizer(
        text_dim=896, d_model=32, n_factors=4, dropout=0.0
    )
    x = torch.randn(2, 5, 896)
    y = module(x)
    assert y.shape == (2, 5, 4, 32)
    assert torch.isfinite(y).all()


def test_uniform_aggregator_empty_text_is_exact_zero():
    module = UniformHistoricalTextAggregator(d_model=32)
    factors = torch.randn(2, 5, 4, 32)
    timing = torch.randn(2, 5, 5)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    context, mass = module(factors, timing, mask, horizon_count=6)
    assert context.shape == (2, 6, 32)
    assert mass.shape == (2, 6, 1)
    assert torch.count_nonzero(context) == 0
    assert torch.count_nonzero(mass) == 0


def test_adapter_outputs_bounded_gate_and_zero_initial_delta():
    module = MemoryFreeConditionalStateAdapter(
        d_model=32, dropout=0.0, adapter_scale=1.0, gate_max=1.0, gate_bias=-2.0
    )
    tensors = [torch.randn(2, 6, 32) for _ in range(5)]
    semantic_mass = torch.ones(2, 6, 1)
    disagreement = torch.rand(2, 6, 1)
    delta, alpha, raw, effective = module(
        *tensors, semantic_mass, disagreement
    )
    assert delta.shape == (2, 6, 32)
    assert alpha.shape == raw.shape == effective.shape == (2, 6, 1)
    assert torch.count_nonzero(delta) == 0
    assert torch.all((alpha >= 0) & (alpha <= 1))
