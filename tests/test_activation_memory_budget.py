"""The --activation-memory-budget knob that base_train.py sets before torch.compile.

Below 1.0 the AOT partitioner stops optimising purely for runtime and solves a
knapsack for the cheapest recompute set that fits the ceiling, so a compiled model
must still build and differentiate under one.

python -m pytest tests/test_activation_memory_budget.py -v
"""

import pytest
import torch
import torch._functorch.config as functorch_config

from nanochat.gpt import GPT, GPTConfig

B, T = 2, 16


def make_model(n_layer=2):
    cfg = GPTConfig(sequence_len=T, vocab_size=64, n_layer=n_layer,
                    n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(cfg)
    model.init_weights()
    return model, cfg


def test_budget_config_is_reachable():
    """base_train.py sets these two by name; a rename upstream must fail loudly here."""
    assert isinstance(functorch_config.activation_memory_budget, float)
    assert functorch_config.activation_memory_budget_runtime_estimator in {
        "flops", "profile", "testing"
    }


@pytest.mark.slow
@pytest.mark.parametrize("budget", [1.0, 0.5])
def test_compiles_and_differentiates_under_budget(budget):
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    with functorch_config.patch(activation_memory_budget=budget,
                                activation_memory_budget_runtime_estimator="flops"):
        compiled = torch.compile(model, dynamic=False)
        compiled(idx, targets).backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.slow
def test_budget_changes_what_is_saved():
    """A tighter budget must save strictly fewer bytes, or the knob is doing nothing."""
    saved = {}
    for budget in (1.0, 0.3):
        model, cfg = make_model(n_layer=4)
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        torch._dynamo.reset()
        with functorch_config.patch(activation_memory_budget=budget,
                                    activation_memory_budget_runtime_estimator="flops"):
            total = 0

            def count(packed):
                nonlocal total
                total += packed.numel() * packed.element_size()
                return packed

            with torch.autograd.graph.saved_tensors_hooks(count, lambda x: x):
                torch.compile(model, dynamic=False)(idx, targets).backward()
            saved[budget] = total

    assert saved[0.3] < saved[1.0], saved
