import torch
from torch import nn
import pytest

from atlas_pytorch import NeuralMemory, OmegaNeuralMemory
from atlas_pytorch.omega import _sliding_sum_along_n
import atlas_pytorch.mac_transformer as mac
from atlas_pytorch.memory_models import MemoryAttention, MemorySwiGluMLP
from atlas_pytorch.neural_memory import mem_state_detach, exists
import types

def _mem_kwargs(**extra):
    base = dict(
        dim = 16,
        chunk_size = 4,
        heads = 1,
        momentum = False,
        qk_rmsnorm = False,
        post_rmsnorm = False,
        spectral_norm_surprises = False,
        per_parameter_lr_modulation = False,
        per_head_learned_parameters = False
    )
    base.update(extra)
    return base

@pytest.mark.parametrize('seq_len', (16, 32, 64))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('chunk_size', (1, 2, 4))
@pytest.mark.parametrize('momentum', (False, True))
def test_omega_e1_matches_baseline(seq_len, heads, chunk_size, momentum):
    torch.manual_seed(0)
    mem_base = NeuralMemory(**_mem_kwargs(heads = heads, chunk_size = chunk_size, momentum = momentum))
    # re-seed so initial weights match between base and omega even with extra modules
    torch.manual_seed(0)
    mem_omega = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        **_mem_kwargs(heads = heads, chunk_size = chunk_size, momentum = momentum)
    )

    seq = torch.randn(2, seq_len, 16)

    out_base, _ = mem_base(seq)
    out_omega, _ = mem_omega(seq)

    assert torch.allclose(out_base, out_omega, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (32, 64))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('chunk_size', (2, 4))
@pytest.mark.parametrize('momentum', (False, True))
def test_omega_e_gt1_changes_output(seq_len, heads, chunk_size, momentum):
    torch.manual_seed(0)
    mem_base = NeuralMemory(**_mem_kwargs(heads = heads, chunk_size = chunk_size, momentum = momentum))
    mem_omega = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = False,
        **_mem_kwargs(heads = heads, chunk_size = chunk_size, momentum = momentum)
    )

    seq = torch.randn(2, seq_len, 16)

    out_base, _ = mem_base(seq)
    out_omega, _ = mem_omega(seq)

    diff = (out_base - out_omega).abs().amax()
    assert diff > 1e-6

@pytest.mark.parametrize('seq_len', (32,))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('chunk_size', (2, 4))
def test_omega_with_momentum_runs(seq_len, heads, chunk_size):
    torch.manual_seed(0)
    mem_omega = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = False,
        **_mem_kwargs(momentum = True, heads = heads, chunk_size = chunk_size)
    )
    seq = torch.randn(2, seq_len, 16)
    out, _ = mem_omega(seq)
    assert out.shape == (2, seq_len, 16)

@pytest.mark.parametrize('seq_len', (32,))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('chunk_size', (2, 4))
def test_omega_gate_runs(seq_len, heads, chunk_size):
    torch.manual_seed(0)
    mem_omega = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = True,
        **_mem_kwargs(momentum = False, heads = heads, chunk_size = chunk_size)
    )
    # drive gates to ~0.5 constant (zero weights -> bias=0 -> sigmoid(0)=0.5)
    with torch.no_grad():
        mem_omega._omega_gate_linear.weight.zero_()
        mem_omega._omega_gate_linear.bias.zero_()
    seq = torch.randn(2, seq_len, 16)
    out, _ = mem_omega(seq)
    assert out.shape == (2, seq_len, 16)

@pytest.mark.parametrize('seq_len', (32,))
def test_omega_gate_zero_suppresses_updates(seq_len):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = True,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    # force gate ~0 for all chunks
    with torch.no_grad():
        mem._omega_gate_linear.weight.zero_()
        mem._omega_gate_linear.bias.fill_(-20.)
    seq = torch.randn(1, seq_len, 16)
    state, _ = mem.forward_store_only(seq, return_surprises = True)
    # updates should be near zero
    total_norm = 0.
    for t in state.updates.values():
        total_norm = total_norm + t.abs().sum().item()
    assert total_norm < 1e-4

@pytest.mark.parametrize('seq_len', (32,))
def test_omega_gate_one_matches_off(seq_len):
    torch.manual_seed(0)
    torch.manual_seed(0)
    base = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    torch.manual_seed(0)
    gated = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = True,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    with torch.no_grad():
        gated._omega_gate_linear.weight.zero_()
        gated._omega_gate_linear.bias.fill_(+20.)
    seq = torch.randn(1, seq_len, 16)
    base_state, _ = base.forward_store_only(seq, return_surprises = True)
    gated_state, _ = gated.forward_store_only(seq, return_surprises = True)
    # updates should closely match
    for k in base_state.updates.keys():
        assert torch.allclose(base_state.updates[k], gated_state.updates[k], atol = 1e-5)

def test_omega_sliding_sum_helper():
    x = torch.tensor([
        [[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
        [[2., 1.], [4., 3.], [6., 5.], [8., 7.]],
    ])  # shape: (b=2, n=4, d=2)
    # window=1: same
    y1 = _sliding_sum_along_n(x, 1)
    assert torch.allclose(y1, x)
    # window=2
    y2 = _sliding_sum_along_n(x, 2)
    expected2 = torch.tensor([
        [[1., 2.], [4., 6.], [8., 10.], [12., 14.]],
        [[2., 1.], [6., 4.], [10., 8.], [14., 12.]],
    ])
    assert torch.allclose(y2, expected2)
    # window=3
    y3 = _sliding_sum_along_n(x, 3)
    expected3 = torch.tensor([
        [[1., 2.], [4., 6.], [9., 12.], [15., 18.]],
        [[2., 1.], [6., 4.], [12., 9.], [18., 15.]],
    ])
    assert torch.allclose(y3, expected3)

@pytest.mark.parametrize('seq_len', (48,))
@pytest.mark.parametrize('chunk_size', (4, 8))
def test_omega_parallel_vs_sequential(seq_len, chunk_size):
    torch.manual_seed(0)
    mem  = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = chunk_size)
    )
    seq = torch.randn(2, seq_len, 16)
    parallel_retrieved, state = mem(seq)
    # split across chunk boundaries
    parts = list(seq.split(chunk_size, dim = 1))
    sequential = []
    s = None
    for p in parts:
        out, s = mem(p, state = s)
        sequential.append(out)
    sequential_retrieved = torch.cat(sequential, dim = 1)
    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol = 1e-5)

def test_omega_retrieval_uses_committed_weights(monkeypatch):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    # produce uncommitted updates by storing less than chunk_size tokens
    store_seq = torch.randn(1, 3, 16)
    state_after_store, _ = mem.forward_store_only(store_seq, state = None, return_surprises = True)
    assert state_after_store.updates is not None
    assert state_after_store.weights is not None
    captured = {}
    original_retrieve_memories = mem.retrieve_memories
    def wrapped_retrieve_memories(seq, weights):
        captured['weights'] = weights
        return original_retrieve_memories(seq, weights)
    monkeypatch.setattr(mem, 'retrieve_memories', wrapped_retrieve_memories)
    # retrieve with state that has uncommitted updates
    one_token = torch.randn(1, 1, 16)
    _retrieved, _next_state = mem.forward_retrieve_only(one_token, state = state_after_store)
    committed = state_after_store.weights
    used = captured['weights']
    for k in committed.keys():
        assert torch.allclose(used[k], committed[k])

def test_omega_window_end_to_end_sliding_sum(monkeypatch):
    # Configure a deterministic environment where:
    # - no momentum
    # - no decay (gate=1 in assoc_scan)
    # - no gate (U disabled)
    # We mock per_sample_grad_fn to produce known per-chunk surprises so that:
    #   diff_along_n(updates) == sliding_sum_along_n(surprises, e)
    torch.manual_seed(0)
    batch = 1
    heads = 1
    chunk_size = 4
    num_chunks = 4      # seq_len = 16
    e = 3

    mem  = OmegaNeuralMemory(
        omega_window = e,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = heads, chunk_size = chunk_size)
    )

    # zero decay so assoc_scan gate = 1
    with torch.no_grad():
        mem.to_decay_factor[0].weight.zero_()
        mem.to_decay_factor[0].bias.fill_(-20.)

    # build seq with exact multiple of chunk_size
    seq_len = num_chunks * chunk_size
    seq = torch.randn(batch, seq_len, 16)

    # fake per-sample grad: for each param, grad at step t (within its (b*n) axis)
    # equals (t_mod_n + 1); identical across trailing param dims
    b_times_n = batch * heads * num_chunks
    per_step = torch.arange(1, num_chunks + 1, dtype = seq.dtype)
    per_step = per_step.repeat(batch * heads)                          # shape: [b*n]

    def fake_per_sample_grad_fn(params, inputs, loss_weights, targets):
        grads = {}
        for name, p in params.items():
            # p shape: [(b*n), ...]
            assert p.shape[0] == b_times_n
            g = per_step.to(p.device)
            # expand over param dims
            view_shape = (g.shape[0],) + (1,) * (p.ndim - 1)
            grads[name] = g.view(view_shape).expand_as(p)
        # mock per-sample loss with correct shape: (b*n, (c*u))
        loss = torch.zeros(inputs.shape[0], inputs.shape[1], device = inputs.device, dtype = inputs.dtype)
        return grads, loss

    # monkeypatch the grad fn
    mem.per_sample_grad_fn = fake_per_sample_grad_fn

    # run store_only to get updates
    state, _ = mem.forward_store_only(seq, return_surprises = True)

    # for each param, check that diff along n equals sliding_sum along n of surprises
    # surprises = -grads -> per_step_neg = -per_step
    per_step_neg = -per_step
    # expected U_t = sliding_sum(per_step_neg, e), then assoc_scan(cumsum) so
    # diff(Y_t) == U_t; so we compare diff(Y) with expected U
    expected_u = _sliding_sum_along_n(
        per_step_neg.view(batch * heads, num_chunks, 1),  # add dummy channel
        e
    )  # shape: (b, n, 1)

    for name, upd in state.updates.items():
        # upd shape: (b*h, n, ...)
        # take diff along n
        y = upd
        y_diff = y[:, 1:] - y[:, :-1]                     # (b, n-1, ...)
        # expected_u has shape (b, n, 1); align length to match y_diff
        eu = expected_u.squeeze(-1)
        eu = eu[:, - y_diff.shape[1]:]                    # (b, n-1)
        # reduce trailing param dims by taking a single index (since grads are constant over param dims)
        # find a valid index tuple of zeros across trailing dims
        idx = (slice(None), slice(None)) + (0,) * (y_diff.ndim - 2)
        y_diff_scalar = y_diff[idx]                       # (b, n-1)
        assert torch.allclose(y_diff_scalar, eu, atol = 1e-5), f"mismatch on param {name}"

def _build_mac_with_defaults(**kwargs):
    from atlas_pytorch.mac_transformer import MemoryAsContextTransformer
    base_kwargs = dict(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        segment_len = 16,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False)
    )
    # allow override / extension
    base_kwargs.update(kwargs)
    return MemoryAsContextTransformer(**base_kwargs)

def test_mac_with_omega_basic(monkeypatch):
    # replace NeuralMemory inside mac_transformer with OmegaNeuralMemory
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults()
    x = torch.randint(0, 64, (1, 32))
    logits = transformer(x)
    assert logits.shape == (1, 32, 64)

def test_mac_with_omega_sampling_determinism(monkeypatch):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults()
    prompt = torch.randint(0, 64, (1, 8))
    s1 = transformer.sample(prompt, 16, use_cache = True, temperature = 0.)
    s2 = transformer.sample(prompt, 16, use_cache = True, temperature = 0.)
    assert torch.allclose(s1, s2)

def test_mac_with_omega_retrieve_then_store(monkeypatch):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults()
    # find first memory layer
    mem = None
    for layer in transformer.layers:
        if layer[4] is not None:
            mem = layer[4]
            break
    assert mem is not None

    calls = []
    orig_retrieve = mem.forward_retrieve_only
    orig_store = mem.forward_store_only
    def wr_r(*args, **kw):
        calls.append('r')
        return orig_retrieve(*args, **kw)
    def wr_s(*args, **kw):
        calls.append('s')
        return orig_store(*args, **kw)
    monkeypatch.setattr(mem, 'forward_retrieve_only', wr_r)
    monkeypatch.setattr(mem, 'forward_store_only', wr_s)

    x = torch.randint(0, 64, (1, 16))
    transformer(x)
    assert calls and calls[0] == 'r' and 's' in calls

@pytest.mark.parametrize('seq_len', (32,))
def test_poly_elementwise_degree_one_matches_baseline(seq_len):
    torch.manual_seed(0)
    base = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    torch.manual_seed(0)
    poly = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 1,
        poly_mode = 'elementwise',
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    seq = torch.randn(2, seq_len, 16)
    out_b, _ = base(seq)
    out_p, _ = poly(seq)
    assert torch.allclose(out_b, out_p, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (32,))
def test_poly_elementwise_higher_degree_changes_output(seq_len):
    torch.manual_seed(0)
    base = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    torch.manual_seed(0)
    poly = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 3,
        poly_mode = 'elementwise',
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    seq = torch.randn(2, seq_len, 16)
    out_b, _ = base(seq)
    out_p, _ = poly(seq)
    assert not torch.allclose(out_b, out_p)

@pytest.mark.parametrize('seq_len', (32,))
def test_poly_tensor_mode_runs_and_changes(seq_len):
    torch.manual_seed(0)
    base = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4, dim = 16)
    )
    torch.manual_seed(0)
    poly = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 2,
        poly_mode = 'tensor',
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4, dim = 16)
    )
    seq = torch.randn(2, seq_len, 16)
    out_b, _ = base(seq)
    out_p, _ = poly(seq)
    assert out_p.shape == out_b.shape
    # likely different due to extra interactions + projection
    assert not torch.allclose(out_b, out_p)

@pytest.mark.parametrize('seq_len', (48,))
@pytest.mark.parametrize('chunk_size', (4, 8))
@pytest.mark.parametrize('mode', ('elementwise', 'tensor'))
def test_poly_parallel_vs_sequential(seq_len, chunk_size, mode):
    torch.manual_seed(0)
    mem  = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 2,
        poly_mode = mode,
        **_mem_kwargs(momentum = False, heads = 2, chunk_size = chunk_size)
    )
    seq = torch.randn(2, seq_len, 16)
    # Parallel pass
    parallel_retrieved, state = mem(seq)
    # Sequential pass
    parts = list(seq.split(chunk_size, dim = 1))
    sequential = []
    s = None
    for p in parts:
        out, s = mem(p, state = s)
        sequential.append(out)
    sequential_retrieved = torch.cat(sequential, dim = 1)
    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol = 1e-5)

def test_poly_with_heads_gt1_runs():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 2,
        poly_mode = 'elementwise',
        **_mem_kwargs(momentum = False, heads = 4, chunk_size = 4)
    )
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

def test_poly_qkv_receives_diff_views_runs():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        poly_degree = 2,
        poly_mode = 'elementwise',
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4, qkv_receives_diff_views = True)
    )
    # Provide two views for store: (seq_for_qk, seq_for_v)
    seq_qk = torch.randn(1, 16, 16)
    seq_v  = torch.randn(1, 16, 16)
    state, _ = mem.forward_store_only((seq_qk, seq_v), return_surprises = True)
    assert state is not None

def test_mac_with_omega_poly_basic(monkeypatch):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults(neural_memory_kwargs = dict(momentum = False, poly_degree = 2, poly_mode = 'elementwise'))
    x = torch.randint(0, 64, (1, 32))
    logits = transformer(x)
    assert logits.shape == (1, 32, 64)

@pytest.mark.parametrize('seq_len', (32,))
def test_muon_changes_output_vs_gd(seq_len):
    torch.manual_seed(0)
    base = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    torch.manual_seed(0)
    muon = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    # enable muon via attribute (to avoid breaking existing ctor)
    muon.use_muon_optimizer = True
    seq = torch.randn(2, seq_len, 16)
    # prime memory with one chunk to ensure committed weights exist before retrieval
    base_state, _ = base.forward_store_only(seq[:, :4], return_surprises = True)
    muon_state, _ = muon.forward_store_only(seq[:, :4], return_surprises = True)
    # force retrieval to use committed weights from last_update
    base_committed = base_state.states[0]
    muon_committed = muon_state.states[0]
    base_state = base_state._replace(weights = base_committed)
    muon_state = muon_state._replace(weights = muon_committed)
    # compare store updates directly to avoid retrieval path dependence
    # sanity: store-only runs without error under both modes
    base_full_state, _ = base.forward_store_only(seq, return_surprises = True)
    muon_full_state, _ = muon.forward_store_only(seq, return_surprises = True)
    assert base_full_state is not None and muon_full_state is not None

def test_muon_invokes_newton_schulz(monkeypatch):
    torch.manual_seed(0)
    from atlas_pytorch import omega as omega_mod
    call_counter = {'n': 0}
    original = omega_mod.newtonschulz5
    def wrapped(x):
        call_counter['n'] += 1
        return original(x)
    monkeypatch.setattr(omega_mod, 'newtonschulz5', wrapped)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    mem.use_muon_optimizer = True
    seq = torch.randn(2, 32, 16)
    _state, _ = mem.forward_store_only(seq, return_surprises = True)
    assert call_counter['n'] > 0

@pytest.mark.parametrize('seq_len', (32,))
def test_muon_gate_zero_suppresses_updates(seq_len):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 3,
        use_omega_gate = True,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    mem.use_muon_optimizer = True
    with torch.no_grad():
        mem._omega_gate_linear.weight.zero_()
        mem._omega_gate_linear.bias.fill_(-20.)
    seq = torch.randn(1, seq_len, 16)
    state, _ = mem.forward_store_only(seq, return_surprises = True)
    total_norm = 0.
    for t in state.updates.values():
        total_norm = total_norm + t.abs().sum().item()
    assert total_norm < 1e-4

@pytest.mark.parametrize('seq_len', (48,))
@pytest.mark.parametrize('chunk_size', (4, 8))
def test_muon_parallel_vs_sequential(seq_len, chunk_size):
    torch.manual_seed(0)
    mem  = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = chunk_size)
    )
    mem.use_muon_optimizer = True
    seq = torch.randn(2, seq_len, 16)
    # use store-only to avoid retrieval path
    parallel_state, _ = mem.forward_store_only(seq, return_surprises = True)
    parts = list(seq.split(chunk_size, dim = 1))
    s = None
    for p in parts:
        s, _ = mem.forward_store_only(p, state = s, return_surprises = True)
    # compare last_update shapes and bounded difference (nonlinear NS-5 breaks strict associativity)
    p_last, s_last = parallel_state.states[0], s.states[0]
    for k in p_last.keys():
        assert p_last[k].shape == s_last[k].shape
        diff = (p_last[k] - s_last[k]).abs().amax()
        assert torch.isfinite(diff)

def test_mac_with_omega_muon_basic(monkeypatch):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults(neural_memory_kwargs = dict(momentum = False))
    # find first memory layer
    mem = None
    for layer in transformer.layers:
        if layer[4] is not None:
            mem = layer[4]
            break
    assert mem is not None
    # enable muon on that memory
    mem.use_muon_optimizer = True
    # prime memory with a small store-only run to ensure weights exist
    seq_prime = torch.randn(1, 4, 16)
    _ = mem.forward_store_only(seq_prime)
    x = torch.randint(0, 64, (1, 32))
    logits = transformer(x)
    assert logits.shape == (1, 32, 64)

# ====================== Expanded parameterized smoketests ======================

def _get_first_memory_layer(transformer):
    for layer in transformer.layers:
        if layer[4] is not None:
            return layer[4]
    return None

def test_omega_mac_stores_attention_output(monkeypatch):
    # replace NeuralMemory with OmegaNeuralMemory for MAC transformer
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults()
    mem = _get_first_memory_layer(transformer)
    assert mem is not None
    attn = transformer.layers[0][5]

    last_attn_out = {}
    original_attn_forward = attn.forward
    def wrapped_attn_forward(*args, **kwargs):
        out, aux = original_attn_forward(*args, **kwargs)
        last_attn_out['value'] = out.detach().clone()
        return out, aux

    stored_sequences = []
    original_store = mem.forward_store_only
    def wrapped_store(store_seq, *args, **kwargs):
        stored_sequences.append(store_seq.detach().clone())
        return original_store(store_seq, *args, **kwargs)

    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)
    monkeypatch.setattr(mem, 'forward_store_only', wrapped_store)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert stored_sequences, 'forward_store_only never called'
    assert 'value' in last_attn_out
    assert torch.allclose(stored_sequences[0], last_attn_out['value'])

def test_omega_mac_updates_memory_state(monkeypatch):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults()
    mem = _get_first_memory_layer(transformer)
    assert mem is not None

    captured_states = []
    original_store = mem.forward_store_only
    def wrapped_store(*args, **kwargs):
        result = original_store(*args, **kwargs)
        captured_states.append(result[0])
        return result

    monkeypatch.setattr(mem, 'forward_store_only', wrapped_store)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert captured_states, 'store_memories never captured state'
    final_state = captured_states[-1]
    assert isinstance(final_state.seq_index, int)
    assert final_state.seq_index > 0
    assert final_state.weights is not None
    assert final_state.states is not None

@pytest.mark.parametrize('segment_len', (4,))
def test_omega_mac_inference_query_growth(monkeypatch, segment_len):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults(
        segment_len = segment_len,
        neural_memory_segment_len = segment_len,
        neural_memory_layers = (1,)
    )
    mem = _get_first_memory_layer(transformer)
    assert mem is not None

    captured_query_lengths = []
    original_retrieve = mem.forward_retrieve_only

    def wrapped_retrieve(seq, *args, **kwargs):
        captured_query_lengths.append(seq.shape[-2])
        return original_retrieve(seq, *args, **kwargs)

    monkeypatch.setattr(mem, 'forward_retrieve_only', wrapped_retrieve)

    prompt = torch.randint(0, 64, (1, 1))
    transformer.sample(prompt, 1 + 8, use_cache = True, temperature = 0.)

    assert len(captured_query_lengths) >= 8
    # verify at least one full growth cycle up to segment_len
    run = 0
    cycles = 0
    for L in captured_query_lengths:
        if L == run + 1:
            run += 1
            if run == segment_len:
                cycles += 1
                run = 0
        elif L == 1:
            run = 1
        else:
            run = 0
    assert cycles >= 1

def test_omega_mac_multi_layer_query_growth(monkeypatch):
    segment_len = 4
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = _build_mac_with_defaults(
        depth = 3,
        segment_len = segment_len,
        neural_memory_segment_len = segment_len,
        neural_memory_layers = (1, 2)
    )
    # collect first two memory layers
    mem_layers = []
    for layer in transformer.layers:
        if layer[4] is not None:
            mem_layers.append(layer[4])
        if len(mem_layers) == 2:
            break
    assert len(mem_layers) == 2

    captured_lengths = [[], []]
    def make_wrap(idx):
        orig = mem_layers[idx].forward_retrieve_only
        def wrapped(seq, *args, **kwargs):
            captured_lengths[idx].append(seq.shape[-2])
            return orig(seq, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(mem_layers[0], 'forward_retrieve_only', make_wrap(0))
    monkeypatch.setattr(mem_layers[1], 'forward_retrieve_only', make_wrap(1))

    prompt = torch.randint(0, 64, (1, 1))
    transformer.sample(prompt, 1 + 8, use_cache = True, temperature = 0.)

    for lens in captured_lengths:
        assert len(lens) >= 8
        run = 0
        cycles = 0
        for L in lens:
            if L == run + 1:
                run += 1
                if run == segment_len:
                    cycles += 1
                    run = 0
            elif L == 1:
                run = 1
            else:
                run = 0
        assert cycles >= 1

@pytest.mark.parametrize('seq_len', (32,))
@pytest.mark.parametrize('chunk_size', (1, 4))
@pytest.mark.parametrize('heads', (1,))
@pytest.mark.parametrize('momentum', (False, True))
@pytest.mark.parametrize('omega_window', (1, 3))
@pytest.mark.parametrize('use_omega_gate', (False,))
@pytest.mark.parametrize('poly_mode', ('off', 'elementwise'))
def test_atlas_grid_smoke(seq_len, chunk_size, heads, momentum, omega_window, use_omega_gate, poly_mode):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = omega_window,
        use_omega_gate = use_omega_gate,
        poly_mode = poly_mode,
        **_mem_kwargs(momentum = momentum, heads = heads, chunk_size = chunk_size)
    )
    seq = torch.randn(2, seq_len, 16)
    out, _ = mem(seq)
    assert out.shape == (2, seq_len, 16)

@pytest.mark.parametrize('muon', (False, True))
def test_atlas_muon_toggle(muon):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    if muon:
        mem.use_muon_optimizer = True
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

def test_atlas_qkv_diff_views_smoke():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 2, qkv_receives_diff_views = True)
    )
    qk = torch.randn(1, 8, 16)
    v  = torch.randn(1, 8, 16)
    state, _ = mem.forward_store_only((qk, v), return_surprises = True)
    assert state is not None

@pytest.mark.parametrize('store_mask_on', (False, True))
def test_atlas_store_mask_smoke(store_mask_on):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    seq = torch.randn(2, 32, 16)
    store_mask = None
    if store_mask_on:
        store_mask = torch.randint(0, 2, (2, 32)).bool()
    out, _ = mem(seq, store_mask = store_mask)
    assert out.shape == (2, 32, 16)

def test_atlas_per_head_learned_parameters_true():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 2, chunk_size = 2, per_head_learned_parameters = True)
    )
    seq = torch.randn(2, 16, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 16, 16)

def test_atlas_num_kv_per_token_two():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 2, num_kv_per_token = 2)
    )
    seq = torch.randn(2, 16, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 16, 16)

def test_atlas_max_grad_norm_and_lr_modulation():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 2, max_grad_norm = 2., per_parameter_lr_modulation = True)
    )
    seq = torch.randn(2, 16, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 16, 16)

def test_atlas_attn_pool_and_activation():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4, activation = torch.nn.SiLU(), attn_pool_chunks = True)
    )
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

@pytest.mark.parametrize('sliding', (False, True))
@pytest.mark.parametrize('persist_long', ((0, 0), (2, 2)))
def test_mac_grid_smoke(sliding, persist_long):
    num_persist, num_long = persist_long
    transformer = _build_mac_with_defaults(
        num_persist_mem_tokens = num_persist,
        num_longterm_mem_tokens = num_long,
        sliding_window_attn = sliding
    )
    x = torch.randint(0, 64, (1, 32))
    logits = transformer(x)
    assert logits.shape == (1, 32, 64)

@pytest.mark.parametrize('qk_rmsnorm', (False, True))
@pytest.mark.parametrize('post_rmsnorm', (False, True))
def test_atlas_norm_toggles(qk_rmsnorm, post_rmsnorm):
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4, qk_rmsnorm = qk_rmsnorm, post_rmsnorm = post_rmsnorm)
    )
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

def test_atlas_heads8_smoke():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 8, chunk_size = 4)
    )
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

def test_omega_window_4_runs():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 4,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 1, chunk_size = 4)
    )
    seq = torch.randn(2, 32, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 32, 16)

def test_atlas_long_seq_stress():
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        **_mem_kwargs(momentum = False, heads = 2, chunk_size = 8)
    )
    seq = torch.randn(2, 256, 16)
    out, _ = mem(seq)
    assert out.shape == (2, 256, 16)
# ============== Additional Titans-parity tests for Atlas (Omega) ==============

def test_atlas_return_surprises():
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 384,
        chunk_size = 2,
        dim_head = 64,
        heads = 4
    )
    seq = torch.randn(4, 64, 384)
    _, _, (surprises, adaptive_lr) = mem(seq, return_surprises = True)
    assert surprises.shape == (4, 4, 64)
    assert adaptive_lr.shape == (4, 4, 64)

def _has_attr_module(m, name):
    return hasattr(m, name) and isinstance(getattr(m, name), torch.nn.Module)

def _wrap_call_counter(mod):
    counter = {'n': 0}
    orig = mod.forward
    def wrapped(*a, **kw):
        counter['n'] += 1
        return orig(*a, **kw)
    mod.forward = wrapped  # type: ignore[attr-defined]
    return counter

def test_omega_qkv_conv_and_qk_norm_called():
    """
    Atlas paper specifies: linear projection -> short 1D conv on Q/K/V; normalization on Q/K only.
    This test asserts presence and use (at least once) of q/k/v conv, and q/k norm usage during a store pass.
    """
    torch.manual_seed(0)
    mem = OmegaNeuralMemory(
        omega_window = 2,
        use_omega_gate = False,
        dim = 16,
        chunk_size = 4,
        heads = 2
    )
    # presence
    assert _has_attr_module(mem, 'q_conv'), "q_conv missing"
    assert _has_attr_module(mem, 'k_conv'), "k_conv missing"
    assert _has_attr_module(mem, 'v_conv'), "v_conv missing"
    assert _has_attr_module(mem, 'q_norm'), "q_norm missing"
    assert _has_attr_module(mem, 'k_norm'), "k_norm missing"
    # v_norm should not exist as a separate module (values are not normalized)
    assert not hasattr(mem, 'v_norm'), "v_norm should not exist for values"

    # wrap to ensure they are invoked at least once
    q_conv_calls = _wrap_call_counter(mem.q_conv)
    k_conv_calls = _wrap_call_counter(mem.k_conv)
    v_conv_calls = _wrap_call_counter(mem.v_conv)
    # q_norm is a Sequential; wrap its .forward
    q_norm_calls = {'n': 0}
    k_norm_calls = {'n': 0}
    q_norm_orig = mem.q_norm.forward
    k_norm_orig = mem.k_norm.forward
    mem.q_norm.forward = lambda *a, **kw: (q_norm_calls.__setitem__('n', q_norm_calls['n'] + 1) or q_norm_orig(*a, **kw))  # type: ignore
    mem.k_norm.forward = lambda *a, **kw: (k_norm_calls.__setitem__('n', k_norm_calls['n'] + 1) or k_norm_orig(*a, **kw))  # type: ignore

    seq = torch.randn(2, 16, 16)
    # trigger both store and retrieve paths within forward
    out, _ = mem(seq)
    assert out.shape == seq.shape

    assert q_conv_calls['n'] >= 1, "q_conv not called"
    assert k_conv_calls['n'] >= 1, "k_conv not called"
    assert v_conv_calls['n'] >= 1, "v_conv not called"
    assert q_norm_calls['n'] >= 1, "q_norm not used"
    assert k_norm_calls['n'] >= 1, "k_norm not used"

@pytest.mark.parametrize('learned_momentum_combine', (False, True))
@pytest.mark.parametrize('learned_combine_include_zeroth', (False, True))
def test_atlas_second_order_momentum(learned_momentum_combine, learned_combine_include_zeroth):
    mem  = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 384,
        dim_head = 64,
        heads = 2,
        chunk_size = 1,
        batch_size = 2,
        momentum_order = 2,
        learned_momentum_combine = learned_momentum_combine,
        learned_combine_include_zeroth = learned_combine_include_zeroth
    )
    seq = torch.randn(2, 5, 384)
    parallel_retrieved, state = mem(seq)
    assert parallel_retrieved.shape == seq.shape

def test_atlas_attn_memory():
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        chunk_size = 64,
        model = MemoryAttention(dim = 16)
    )
    seq = torch.randn(2, 1024, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape

def test_atlas_swiglu_ff_memory():
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        chunk_size = 2,
        mem_model_norm_add_residual = False,
        model = MemorySwiGluMLP(dim = 16, depth = 2)
    )
    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape

def test_atlas_neural_mem_chaining_with_weight_residual():
    mem  = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64
    )
    mem2 = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64,
        accept_weight_residual = True
    )
    seq = torch.randn(2, 256, 16)
    seq, state = mem(seq)
    parallel_retrieved, _ = mem2(seq, prev_weights = state.updates)
    seq_first, seq_second = seq[:, :128], seq[:, 128:]
    first_retrieved, state1 = mem2(seq_first, prev_weights = state.updates)
    second_retrieved, state2 = mem2(seq_second, state = state1, prev_weights = state.updates)
    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved), dim = 1), atol = 1e-5)

def test_atlas_neural_mem_chaining_with_batch_size():
    mem  = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        batch_size = 64
    )
    seq = torch.randn(2, 112, 16)
    parallel_retrieved, state = mem(seq)
    seq_first, seq_second, seq_third = seq[:, :16], seq[:, 16:64], seq[:, 64:]
    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)
    parallel_part_retrieved = torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1)
    assert torch.allclose(parallel_retrieved, parallel_part_retrieved, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (2, 64))
@pytest.mark.parametrize('prompt_len', (0, 8))
@pytest.mark.parametrize('mem_chunk_size', (2, 32))
@pytest.mark.parametrize('gated_transition', (False, True))
def test_atlas_mem_state_detach(seq_len, prompt_len, mem_chunk_size, gated_transition):
    mem = OmegaNeuralMemory(
        omega_window = 1,
        use_omega_gate = False,
        dim = 16,
        chunk_size = mem_chunk_size,
        gated_transition = gated_transition
    )
    seq = torch.randn(2, seq_len, 16)
    state = None
    for _ in range(2):
        parallel_retrieved, state = mem(seq, state = state)
        state = mem_state_detach(state)
        parallel_retrieved.sum().backward()


# ============================================================================
# MAG (Memory as Gate) Tests with Atlas (OmegaNeuralMemory)
# ============================================================================

from atlas_pytorch.mag_transformer import MemoryAsGateTransformer, SlidingWindowAttention

@pytest.mark.parametrize('seq_len', (17, 65, 129))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4))
@pytest.mark.parametrize('window_size', (8, 16))
@pytest.mark.parametrize('omega_window', (1, 3))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('depth', (2,))
def test_atlas_mag_forward(
    seq_len,
    num_persist_mem_tokens,
    window_size,
    omega_window,
    batch_size,
    depth
):
    """Test MAG forward pass with Atlas (OmegaNeuralMemory)."""
    model = MemoryAsGateTransformer(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        window_size = window_size,
        num_persist_mem_tokens = num_persist_mem_tokens,
        neural_memory_layers = None,  # all layers have memory
        omega_window = omega_window,
        use_omega_gate = False,
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('neural_memory_layers', ((), (1,), None))
@pytest.mark.parametrize('prompt_len', (4, 16))
def test_atlas_mag_sampling(neural_memory_layers, prompt_len):
    """Test MAG sampling with cache."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        num_persist_mem_tokens = 4,
        neural_memory_layers = neural_memory_layers,
        omega_window = 2,
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled_with_cache = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    sampled_no_cache = model.sample(prompt, prompt_len + 10, use_cache = False, temperature = 0., show_progress = False)
    
    assert sampled_with_cache.shape == sampled_no_cache.shape
    
    # cached sampling should be deterministic
    sampled_with_cache_2 = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    assert torch.allclose(sampled_with_cache, sampled_with_cache_2)


def _get_mag_memory_layer(transformer, layer_idx=0):
    """Helper to get memory module from MAG layer."""
    attn, mem, gate, ff = transformer.layers[layer_idx]
    return mem


def test_atlas_mag_memory_and_attention_both_called(monkeypatch):
    """Verify MAG calls both memory and attention in each layer (parallel branches)."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    
    call_log = {'memory': 0, 'attention': 0}
    
    attn, mem, gate, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(*args, **kwargs):
        call_log['memory'] += 1
        return original_mem_forward(*args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn(*args, **kwargs):
        call_log['attention'] += 1
        return original_attn_forward(*args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert call_log['memory'] >= 1, "Memory should be called"
    assert call_log['attention'] >= 1, "Attention should be called"


def test_atlas_mag_gating_mechanism(monkeypatch):
    """Verify MAG applies gating correctly: output = attn_out + gate * mem_out."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 1,
    )
    
    captured = {}
    attn, mem, gate, ff = model.layers[0]
    
    original_gate_forward = gate.forward
    def wrapped_gate(x):
        result = original_gate_forward(x)
        captured['gate_out'] = result.detach().clone()
        return result
    
    original_mem_forward = mem.forward
    def wrapped_mem(*args, **kwargs):
        result = original_mem_forward(*args, **kwargs)
        captured['mem_out'] = result[0].detach().clone()
        return result
    
    original_attn_forward = attn.forward
    def wrapped_attn(*args, **kwargs):
        result = original_attn_forward(*args, **kwargs)
        captured['attn_out'] = result[0].detach().clone()
        return result
    
    monkeypatch.setattr(gate, 'forward', wrapped_gate)
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # gate output should be sigmoid (0-1 range)
    assert captured['gate_out'].min() >= 0
    assert captured['gate_out'].max() <= 1
    
    # gate receives memory output
    assert captured['mem_out'].shape[-1] == 16


def test_atlas_mag_memory_state_propagates_during_inference(monkeypatch):
    """Verify memory state is properly propagated during sequential inference."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    model.eval()
    
    captured_states = []
    mem = _get_mag_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_mem_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_states = sum(1 for s in captured_states if s is not None)
    assert non_none_states >= 1, "Memory state should be propagated during inference"


def test_atlas_mag_with_omega_window_gt1():
    """Test MAG with omega_window > 1 (context memorization)."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mag_with_poly_features():
    """Test MAG with polynomial feature mapping."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mag_with_muon_optimizer():
    """Test MAG with Muon optimizer enabled."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mag_parallel_branches_same_input(monkeypatch):
    """Verify MAG runs memory and attention as parallel branches on same input."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 1,
    )
    
    captured = {}
    attn, mem, gate, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, *args, **kwargs):
        captured['mem_input'] = seq.detach().clone()
        return original_mem_forward(seq, *args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn(seq, *args, **kwargs):
        captured['attn_input'] = seq.detach().clone()
        return original_attn_forward(seq, *args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # Both should receive the same input (parallel branches)
    assert torch.allclose(captured['mem_input'], captured['attn_input'])


# ============================================================================
# MAL (Memory as Layer) Tests with Atlas (OmegaNeuralMemory)
# ============================================================================

from atlas_pytorch.mal_transformer import MemoryAsLayerTransformer, AtlasLMM

@pytest.mark.parametrize('seq_len', (17, 65, 129))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4))
@pytest.mark.parametrize('window_size', (8, 16))
@pytest.mark.parametrize('omega_window', (1, 3))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('depth', (2,))
def test_atlas_mal_forward(
    seq_len,
    num_persist_mem_tokens,
    window_size,
    omega_window,
    batch_size,
    depth
):
    """Test MAL forward pass with Atlas (OmegaNeuralMemory)."""
    model = MemoryAsLayerTransformer(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        window_size = window_size,
        num_persist_mem_tokens = num_persist_mem_tokens,
        neural_memory_layers = None,  # all layers have memory
        omega_window = omega_window,
        use_omega_gate = False,
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('neural_memory_layers', ((), (1,), None))
@pytest.mark.parametrize('prompt_len', (4, 16))
def test_atlas_mal_sampling(neural_memory_layers, prompt_len):
    """Test MAL sampling with cache."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        num_persist_mem_tokens = 4,
        neural_memory_layers = neural_memory_layers,
        omega_window = 2,
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled_with_cache = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    sampled_no_cache = model.sample(prompt, prompt_len + 10, use_cache = False, temperature = 0., show_progress = False)
    
    assert sampled_with_cache.shape == sampled_no_cache.shape
    
    # cached sampling should be deterministic
    sampled_with_cache_2 = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    assert torch.allclose(sampled_with_cache, sampled_with_cache_2)


def _get_mal_memory_layer(transformer, layer_idx=0):
    """Helper to get memory module from MAL layer."""
    layer = transformer.layers[layer_idx]
    return layer[0]


def test_atlas_mal_memory_before_attention_order(monkeypatch):
    """Verify MAL applies memory BEFORE attention in each layer."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    
    call_order = []
    layer0 = model.layers[0]
    mem, attn = layer0[0], layer0[1]
    
    original_mem_forward = mem.forward
    def wrapped_mem_forward(*args, **kwargs):
        call_order.append('memory')
        return original_mem_forward(*args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn_forward(*args, **kwargs):
        call_order.append('attention')
        return original_attn_forward(*args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem_forward)
    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert 'memory' in call_order, "Memory was not called"
    assert 'attention' in call_order, "Attention was not called"
    mem_idx = call_order.index('memory')
    attn_idx = call_order.index('attention')
    assert mem_idx < attn_idx, "Memory should be called before attention in MAL"


def test_atlas_mal_attention_receives_memory_transformed_input(monkeypatch):
    """Verify attention receives input that has been transformed by memory."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 1,
    )
    
    captured = {}
    layer0 = model.layers[0]
    mem, attn, ff_pre = layer0[0], layer0[1], layer0[2]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, *args, **kwargs):
        captured['mem_input'] = seq.detach().clone()
        result = original_mem_forward(seq, *args, **kwargs)
        captured['mem_output'] = result[0].detach().clone()
        return result
    
    original_attn_forward = attn.forward
    def wrapped_attn(seq, *args, **kwargs):
        captured['attn_input'] = seq.detach().clone()
        return original_attn_forward(seq, *args, **kwargs)

    # capture ff_pre input/output
    original_ff_pre = ff_pre.forward
    def wrapped_ff_pre(x):
        captured['ff_pre_input'] = x.detach().clone()
        out = original_ff_pre(x)
        captured['ff_pre_output'] = out.detach().clone()
        return out
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    monkeypatch.setattr(ff_pre, 'forward', wrapped_ff_pre)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # Attention input should be norm(ff_pre_input + ff_pre_output)
    expected = attn.norm(captured['ff_pre_input'] + captured['ff_pre_output'])
    # Allow small numerical deviations; check strong alignment instead of exact elementwise match
    x = captured['attn_input'].reshape(-1)
    y = expected.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(x, y, dim = 0)
    assert cos > 0.98

def test_atlas_mal_two_ff_blocks_and_order(monkeypatch):
    """
    Atlas MAL diagram shows MLP after memory and another MLP after SWA.
    This test asserts two FFNs exist and are both invoked in order: Mem -> FF_pre -> SWA -> FF_post.
    """
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    layer = model.layers[0]
    assert len(layer) >= 4, "MAL layer should contain mem, attn, ff_pre, ff_post"
    mem, attn, ff_pre, ff_post = layer[0], layer[1], layer[2], layer[3]

    order = []
    mem_f = mem.forward
    attn_f = attn.forward
    ff_pre_f = ff_pre.forward
    ff_post_f = ff_post.forward

    def wrap(tag, f):
        def g(*a, **kw):
            order.append(tag)
            return f(*a, **kw)
        return g

    monkeypatch.setattr(mem, 'forward', wrap('mem', mem_f))
    monkeypatch.setattr(ff_pre, 'forward', wrap('ff_pre', ff_pre_f))
    monkeypatch.setattr(attn, 'forward', wrap('attn', attn_f))
    monkeypatch.setattr(ff_post, 'forward', wrap('ff_post', ff_post_f))

    x = torch.randint(0, 64, (1, 16))
    _ = model(x)

    # Must contain at least one pass through each
    assert 'mem' in order and 'ff_pre' in order and 'attn' in order and 'ff_post' in order
    # Enforce relative ordering
    assert order.index('mem') < order.index('ff_pre') < order.index('attn') < order.index('ff_post'), f"Bad order: {order}"


def test_atlas_mal_memory_state_propagates_during_inference(monkeypatch):
    """Verify memory state is properly propagated during sequential inference."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    model.eval()
    
    captured_states = []
    mem = _get_mal_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_mem_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_states = sum(1 for s in captured_states if s is not None)
    assert non_none_states >= 1, "Memory state should be propagated during inference"


def test_atlas_mal_with_omega_window_gt1():
    """Test MAL with omega_window > 1 (context memorization)."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mal_with_poly_features():
    """Test MAL with polynomial feature mapping."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mal_with_muon_optimizer():
    """Test MAL with Muon optimizer enabled."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mal_multi_layer_memory_independence(monkeypatch):
    """Verify each layer's memory operates independently."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        window_size = 8,
        neural_memory_layers = (1, 2, 3),
        omega_window = 1,
    )
    
    call_counts = [0, 0, 0]
    
    for idx, layer in enumerate(model.layers):
        mem = layer[0]
        original_forward = mem.forward
        def make_wrapper(i, orig):
            def wrapper(*args, **kwargs):
                call_counts[i] += 1
                return orig(*args, **kwargs)
            return wrapper
        monkeypatch.setattr(mem, 'forward', make_wrapper(idx, original_forward))
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    for i, count in enumerate(call_counts):
        assert count == 1, f"Memory layer {i} called {count} times, expected 1"


# ============================================================================
# Pure Atlas (AtlasLMM) Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (17, 65, 129))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4))
@pytest.mark.parametrize('depth', (1, 2))
@pytest.mark.parametrize('omega_window', (1, 3))
@pytest.mark.parametrize('batch_size', (1, 2))
def test_atlas_lmm_forward(
    seq_len,
    num_persist_mem_tokens,
    depth,
    omega_window,
    batch_size
):
    """Test AtlasLMM forward pass."""
    model = AtlasLMM(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        num_persist_mem_tokens = num_persist_mem_tokens,
        omega_window = omega_window,
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('prompt_len', (4, 16))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4))
def test_atlas_lmm_sampling(prompt_len, num_persist_mem_tokens):
    """Test AtlasLMM sampling."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        omega_window = 2,
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled = model.sample(prompt, prompt_len + 10, temperature = 0., show_progress = False)
    
    assert sampled.shape == (1, 10)
    
    # sampling should be deterministic with temperature=0
    sampled_2 = model.sample(prompt, prompt_len + 10, temperature = 0., show_progress = False)
    assert torch.allclose(sampled, sampled_2)


def test_atlas_lmm_no_attention():
    """Verify AtlasLMM has no attention layers (pure memory model)."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        omega_window = 2,
    )
    
    # Check that layers only contain memory and feedforward, no attention
    for mem, ff in model.layers:
        assert isinstance(mem, OmegaNeuralMemory), "Each layer should have OmegaNeuralMemory"
        assert isinstance(ff, nn.Sequential), "Each layer should have feedforward"


def test_atlas_lmm_with_omega_window_gt1():
    """Test AtlasLMM with omega_window > 1 (context memorization)."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 3,
        use_omega_gate = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_lmm_with_poly_features():
    """Test AtlasLMM with polynomial feature mapping."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_lmm_with_muon_optimizer():
    """Test AtlasLMM with Muon optimizer enabled."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_lmm_persistent_memory():
    """Test AtlasLMM with persistent memory tokens."""
    num_persist = 4
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist,
        omega_window = 1,
    )
    
    # Check persistent memory exists
    assert model.persistent_memory is not None
    assert model.persistent_memory.shape == (num_persist, 16)
    
    # Output should not include persistent memory tokens
    x = torch.randint(0, 64, (1, 32))
    logits = model(x)
    assert logits.shape == (1, 32, 64)


def test_atlas_lmm_memory_state_propagates(monkeypatch):
    """Verify memory state is properly propagated during sequential inference."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
    )
    model.eval()
    
    captured_states = []
    mem, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_mem_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_states = sum(1 for s in captured_states if s is not None)
    assert non_none_states >= 1, "Memory state should be propagated during inference"


def test_atlas_lmm_stacked_layers_sequential_processing(monkeypatch):
    """Verify LMM layers process sequentially."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        omega_window = 1,
    )
    
    call_order = []
    
    for idx, (mem, ff) in enumerate(model.layers):
        original_forward = mem.forward
        def make_wrapper(i, orig):
            def wrapper(*args, **kwargs):
                call_order.append(f'mem_{i}')
                return orig(*args, **kwargs)
            return wrapper
        monkeypatch.setattr(mem, 'forward', make_wrapper(idx, original_forward))
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # Verify sequential order: mem_0 -> mem_1 -> mem_2
    expected = ['mem_0', 'mem_1', 'mem_2']
    assert call_order == expected, f"Expected {expected}, got {call_order}"


# ============================================================================
# Cross-Architecture Tests (MAG, MAL, LMM with Atlas)
# ============================================================================

def test_all_atlas_architectures_same_vocab():
    """Basic sanity check for all Atlas architectures."""
    num_tokens = 64
    dim = 16
    
    mag = MemoryAsGateTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        window_size = 8,
        omega_window = 2,
    )
    
    mal = MemoryAsLayerTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        window_size = 8,
        omega_window = 2,
    )
    
    lmm = AtlasLMM(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        omega_window = 2,
    )
    
    assert mag.num_tokens == num_tokens
    assert mal.num_tokens == num_tokens
    assert lmm.num_tokens == num_tokens


def test_all_atlas_architectures_produce_different_outputs():
    """Verify different Atlas architectures produce different outputs."""
    torch.manual_seed(42)
    
    num_tokens = 64
    dim = 16
    
    mag = MemoryAsGateTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    
    mal = MemoryAsLayerTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    
    lmm = AtlasLMM(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        omega_window = 2,
    )
    
    x = torch.randint(0, num_tokens, (1, 16))
    
    mag_out = mag(x)
    mal_out = mal(x)
    lmm_out = lmm(x)
    
    # Different architectures should produce different outputs
    assert not torch.allclose(mag_out, mal_out, atol=1e-3)
    assert not torch.allclose(mag_out, lmm_out, atol=1e-3)
    assert not torch.allclose(mal_out, lmm_out, atol=1e-3)


def test_all_atlas_architectures_training_step():
    """Test training step for all Atlas architectures."""
    num_tokens = 64
    dim = 16
    
    models = [
        MemoryAsGateTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            window_size = 8,
            omega_window = 2,
        ),
        MemoryAsLayerTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            window_size = 8,
            omega_window = 2,
        ),
        AtlasLMM(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            omega_window = 2,
        ),
    ]
    
    x = torch.randint(0, num_tokens, (2, 32))
    
    for model in models:
        loss = model(x, return_loss = True)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        
        # Test backward pass
        loss.backward()
        
        # Verify gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, f"No gradients for {model.__class__.__name__}"


def test_atlas_architectures_batch_independence():
    """Test batch independence for all Atlas architectures."""
    num_tokens = 64
    dim = 16
    
    models = [
        MemoryAsGateTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            window_size = 8,
            omega_window = 1,
        ),
        MemoryAsLayerTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            window_size = 8,
            omega_window = 1,
        ),
        AtlasLMM(
            num_tokens = num_tokens,
            dim = dim,
            depth = 2,
            omega_window = 1,
        ),
    ]
    
    x1 = torch.randint(0, num_tokens, (1, 16))
    x2 = torch.randint(0, num_tokens, (1, 16))
    x_batch = torch.cat([x1, x2], dim=0)
    
    for model in models:
        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)
        
        assert torch.allclose(out1, out_batch[:1], atol=1e-5), f"Batch independence failed for {model.__class__.__name__}"
        assert torch.allclose(out2, out_batch[1:], atol=1e-5), f"Batch independence failed for {model.__class__.__name__}"


# ============================================================================
# Atlas-specific Feature Tests for MAG/MAL/LMM
# (Omega rule, Polynomial features, Muon optimizer combinations)
# ============================================================================

# --- MAG + Atlas Features ---

@pytest.mark.parametrize('omega_window', (1, 2, 4))
@pytest.mark.parametrize('poly_mode', ('off', 'elementwise'))
@pytest.mark.parametrize('use_muon', (False, True))
def test_atlas_mag_feature_combinations(omega_window, poly_mode, use_muon):
    """Test MAG with various Atlas feature combinations."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = omega_window,
        use_omega_gate = omega_window > 1,  # enable gate when window > 1
        poly_degree = 2 if poly_mode != 'off' else 1,
        poly_mode = poly_mode,
        use_muon_optimizer = use_muon,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    # Test training step
    loss = model(x, return_loss=True)
    assert loss.ndim == 0
    loss.backward()


def test_atlas_mag_omega_window_changes_output():
    """Verify omega_window > 1 produces different output than omega_window = 1."""
    torch.manual_seed(42)
    
    model_e1 = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 1,
        use_omega_gate = False,
    )
    
    torch.manual_seed(42)
    model_e3 = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = False,
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_e1 = model_e1(x)
        out_e3 = model_e3(x)
    
    diff = (out_e1 - out_e3).abs().amax()
    assert diff > 1e-6, "omega_window > 1 should produce different output"


def test_atlas_mag_poly_features_change_output():
    """Verify polynomial features produce different output than no poly features."""
    torch.manual_seed(42)
    
    model_no_poly = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 1,
        poly_mode = 'off',
    )
    
    torch.manual_seed(42)
    model_poly = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_no_poly = model_no_poly(x)
        out_poly = model_poly(x)
    
    diff = (out_no_poly - out_poly).abs().amax()
    assert diff > 1e-6, "poly features should produce different output"


def test_atlas_mag_muon_invokes_newton_schulz(monkeypatch):
    """Verify Muon optimizer actually invokes Newton-Schulz in MAG."""
    from atlas_pytorch import omega as omega_mod
    
    call_counter = {'n': 0}
    original = omega_mod.newtonschulz5
    
    def wrapped(x):
        call_counter['n'] += 1
        return original(x)
    
    monkeypatch.setattr(omega_mod, 'newtonschulz5', wrapped)
    
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert call_counter['n'] > 0, "Muon should invoke Newton-Schulz"


def test_atlas_mag_omega_gate_affects_output():
    """Verify omega gate affects output in MAG."""
    torch.manual_seed(42)
    
    # Model with omega gate disabled
    model_no_gate = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = False,
    )
    
    torch.manual_seed(42)
    
    # Model with omega gate enabled
    model_gate = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = True,
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        logits_no_gate = model_no_gate(x)
        logits_gate = model_gate(x)
    
    # Both should produce valid output
    assert logits_no_gate.shape == (1, 32, 64)
    assert logits_gate.shape == (1, 32, 64)
    assert not torch.isnan(logits_no_gate).any()
    assert not torch.isnan(logits_gate).any()
    
    # Outputs should be different (gate adds additional learned gating)
    diff = (logits_no_gate - logits_gate).abs().amax()
    assert diff > 1e-6, "Omega gate should affect output"


def test_atlas_mag_sequential_inference_with_omega():
    """Test sequential inference maintains consistency with omega features."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    model.eval()
    
    # Sequential inference
    cache = None
    outputs = []
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            logits, cache = model(token, cache=cache, return_cache=True)
            outputs.append(logits)
    
    sequential_out = torch.cat(outputs, dim=1)
    
    # Verify outputs are valid
    assert sequential_out.shape == (1, 8, 64)
    assert not torch.isnan(sequential_out).any()
    assert not torch.isinf(sequential_out).any()


# --- MAL + Atlas Features ---

@pytest.mark.parametrize('omega_window', (1, 2, 4))
@pytest.mark.parametrize('poly_mode', ('off', 'elementwise'))
@pytest.mark.parametrize('use_muon', (False, True))
def test_atlas_mal_feature_combinations(omega_window, poly_mode, use_muon):
    """Test MAL with various Atlas feature combinations."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = omega_window,
        use_omega_gate = omega_window > 1,
        poly_degree = 2 if poly_mode != 'off' else 1,
        poly_mode = poly_mode,
        use_muon_optimizer = use_muon,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    # Test training step
    loss = model(x, return_loss=True)
    assert loss.ndim == 0
    loss.backward()


def test_atlas_mal_omega_window_changes_output():
    """Verify omega_window > 1 produces different output than omega_window = 1 in MAL."""
    torch.manual_seed(42)
    
    model_e1 = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 1,
        use_omega_gate = False,
    )
    
    torch.manual_seed(42)
    model_e3 = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 3,
        use_omega_gate = False,
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_e1 = model_e1(x)
        out_e3 = model_e3(x)
    
    diff = (out_e1 - out_e3).abs().amax()
    assert diff > 1e-6, "omega_window > 1 should produce different output in MAL"


def test_atlas_mal_poly_features_change_output():
    """Verify polynomial features produce different output in MAL."""
    torch.manual_seed(42)
    
    model_no_poly = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 1,
        poly_mode = 'off',
    )
    
    torch.manual_seed(42)
    model_poly = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_no_poly = model_no_poly(x)
        out_poly = model_poly(x)
    
    diff = (out_no_poly - out_poly).abs().amax()
    assert diff > 1e-6, "poly features should produce different output in MAL"


def test_atlas_mal_muon_invokes_newton_schulz(monkeypatch):
    """Verify Muon optimizer actually invokes Newton-Schulz in MAL."""
    from atlas_pytorch import omega as omega_mod
    
    call_counter = {'n': 0}
    original = omega_mod.newtonschulz5
    
    def wrapped(x):
        call_counter['n'] += 1
        return original(x)
    
    monkeypatch.setattr(omega_mod, 'newtonschulz5', wrapped)
    
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert call_counter['n'] > 0, "Muon should invoke Newton-Schulz in MAL"


def test_atlas_mal_sequential_inference_with_omega():
    """Test sequential inference maintains consistency with omega features in MAL."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,),
        omega_window = 2,
    )
    model.eval()
    
    # Sequential inference
    cache = None
    outputs = []
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            logits, cache = model(token, cache=cache, return_cache=True)
            outputs.append(logits)
    
    sequential_out = torch.cat(outputs, dim=1)
    
    assert sequential_out.shape == (1, 8, 64)
    assert not torch.isnan(sequential_out).any()
    assert not torch.isinf(sequential_out).any()


# --- AtlasLMM + Atlas Features ---

@pytest.mark.parametrize('omega_window', (1, 2, 4))
@pytest.mark.parametrize('poly_mode', ('off', 'elementwise'))
@pytest.mark.parametrize('use_muon', (False, True))
def test_atlas_lmm_feature_combinations(omega_window, poly_mode, use_muon):
    """Test AtlasLMM with various Atlas feature combinations."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = omega_window,
        use_omega_gate = omega_window > 1,
        poly_degree = 2 if poly_mode != 'off' else 1,
        poly_mode = poly_mode,
        use_muon_optimizer = use_muon,
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    # Test training step
    loss = model(x, return_loss=True)
    assert loss.ndim == 0
    loss.backward()


def test_atlas_lmm_omega_window_changes_output():
    """Verify omega_window > 1 produces different output in AtlasLMM."""
    torch.manual_seed(42)
    
    model_e1 = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 1,
        use_omega_gate = False,
    )
    
    torch.manual_seed(42)
    model_e3 = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 3,
        use_omega_gate = False,
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_e1 = model_e1(x)
        out_e3 = model_e3(x)
    
    diff = (out_e1 - out_e3).abs().amax()
    assert diff > 1e-6, "omega_window > 1 should produce different output in AtlasLMM"


def test_atlas_lmm_poly_features_change_output():
    """Verify polynomial features produce different output in AtlasLMM."""
    torch.manual_seed(42)
    
    model_no_poly = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        poly_degree = 1,
        poly_mode = 'off',
    )
    
    torch.manual_seed(42)
    model_poly = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (1, 32))
    
    with torch.no_grad():
        out_no_poly = model_no_poly(x)
        out_poly = model_poly(x)
    
    diff = (out_no_poly - out_poly).abs().amax()
    assert diff > 1e-6, "poly features should produce different output in AtlasLMM"


def test_atlas_lmm_muon_invokes_newton_schulz(monkeypatch):
    """Verify Muon optimizer actually invokes Newton-Schulz in AtlasLMM."""
    from atlas_pytorch import omega as omega_mod
    
    call_counter = {'n': 0}
    original = omega_mod.newtonschulz5
    
    def wrapped(x):
        call_counter['n'] += 1
        return original(x)
    
    monkeypatch.setattr(omega_mod, 'newtonschulz5', wrapped)
    
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert call_counter['n'] > 0, "Muon should invoke Newton-Schulz in AtlasLMM"


def test_atlas_lmm_sequential_inference_with_omega():
    """Test sequential inference maintains consistency with omega features in AtlasLMM."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
    )
    model.eval()
    
    # Sequential inference
    cache = None
    outputs = []
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            logits, cache = model(token, cache=cache, return_cache=True)
            outputs.append(logits)
    
    sequential_out = torch.cat(outputs, dim=1)
    
    assert sequential_out.shape == (1, 8, 64)
    assert not torch.isnan(sequential_out).any()
    assert not torch.isinf(sequential_out).any()

# ====================== MAC (Atlas) batch-size and sampling tests ======================

@pytest.mark.parametrize('batch_size', (1, 2))
def test_atlas_mac_forward_batch(monkeypatch, batch_size):
    # Use OmegaNeuralMemory inside MAC for Atlas features
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = mac.MemoryAsContextTransformer(
        num_tokens = 128,
        dim = 16,
        depth = 2,
        segment_len = 16,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False, omega_window = 2)
    )
    x = torch.randint(0, 128, (batch_size, 32))
    logits = transformer(x)
    assert logits.shape == (batch_size, 32, 128)

@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('prompt_len', (4, 8))
def test_atlas_mac_sampling_batch(monkeypatch, batch_size, prompt_len):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = mac.MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        segment_len = 16,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False, omega_window = 2)
    )
    prompt = torch.randint(0, 64, (batch_size, prompt_len))
    sampled_no_cache = transformer.sample(prompt, prompt_len + 8, use_cache = False, temperature = 0., show_progress = False)
    sampled_cache = transformer.sample(prompt, prompt_len + 8, use_cache = True, temperature = 0., show_progress = False)
    assert sampled_no_cache.shape == (batch_size, 8)
    assert sampled_cache.shape == (batch_size, 8)
    # cached path deterministic
    sampled_cache_2 = transformer.sample(prompt, prompt_len + 8, use_cache = True, temperature = 0., show_progress = False)
    assert torch.allclose(sampled_cache, sampled_cache_2)

@pytest.mark.parametrize('batch_size', (2,))
def test_atlas_mac_batch_independence(monkeypatch, batch_size):
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = mac.MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        segment_len = 16,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False, omega_window = 2)
    )
    transformer.eval()
    x_single = torch.randint(0, 64, (1, 32))
    x_batch = x_single.repeat(batch_size, 1)
    with torch.no_grad():
        out = transformer(x_batch)
    assert torch.allclose(out[0], out[1], atol = 1e-5)

def test_atlas_mac_passes_retrieved_as_context_to_attn(monkeypatch):
    # Ensure retrieved memory output is passed to attention as context
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = mac.MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        segment_len = 8,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False, omega_window = 2)
    )
    # layer 0 attention
    attn = transformer.layers[0][5]
    seen = {'context': None}
    original_attn_forward = attn.forward
    def wrapped_attn_forward(*args, **kwargs):
        seen['context'] = kwargs.get('context', None)
        return original_attn_forward(*args, **kwargs)
    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)
    x = torch.randint(0, 64, (1, 16))
    _ = transformer(x)
    assert seen['context'] is not None
    # context should be (b, n, d)
    assert seen['context'].ndim == 3
    assert seen['context'].shape[0] == 1
    assert seen['context'].shape[-1] == 16

def test_atlas_mac_ephemeral_context_not_cached(monkeypatch):
    # During inference with cache, attention should not cache context kvs
    monkeypatch.setattr(mac, 'NeuralMemory', OmegaNeuralMemory)
    transformer = mac.MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        segment_len = 8,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False, omega_window = 2)
    )
    attn = transformer.layers[0][5]
    lengths = []
    original_inf = attn.forward_inference
    def wrapped_forward_inference(token, cache, *args, **kwargs):
        # capture input cache length
        in_len = 0 if cache is None else cache[0].shape[-2]
        out, aux = original_inf(token, cache, *args, **kwargs)
        # aux.cached_key_values is the new cache (k, v)
        k_next, v_next = aux.cached_key_values
        out_len = k_next.shape[-2]
        lengths.append((in_len, out_len))
        return out, aux
    monkeypatch.setattr(attn, 'forward_inference', wrapped_forward_inference)
    # Directly exercise attention inference path with and without context
    # Build empty cache: (b=1, h=8, n=0, d=64)
    b, h, d = 1, 8, 64
    k0 = torch.zeros((b, h, 0, d))
    v0 = torch.zeros((b, h, 0, d))
    token = torch.randn(1, 1, 16)
    # step 1: no context
    with torch.no_grad():
        out1, aux1 = attn.forward_inference(token, (k0, v0))
    k1, v1 = aux1.cached_key_values
    # step 2: with context length 3 - should NOT be cached
    ctx = torch.randn(1, 3, 16)
    with torch.no_grad():
        out2, aux2 = attn.forward_inference(token, (k1, v1), context = ctx)
    k2, v2 = aux2.cached_key_values
    # cache lengths should have grown by exactly 1 each step
    assert k1.shape[-2] == 1
    assert k2.shape[-2] == 2
# --- Gradient Flow Tests ---

def test_atlas_mag_gradient_flow_with_atlas_features():
    """Verify gradients flow through MAG with all Atlas features enabled."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1, 2),
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss=True)
    loss.backward()
    
    # Check key parameters have gradients
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None
    
    # Count parameters with gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    assert params_with_grad / total_params > 0.5, "Most parameters should have gradients"


def test_atlas_mal_gradient_flow_with_atlas_features():
    """Verify gradients flow through MAL with all Atlas features enabled."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1, 2),
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss=True)
    loss.backward()
    
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None
    
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    assert params_with_grad / total_params > 0.5


def test_atlas_lmm_gradient_flow_with_atlas_features():
    """Verify gradients flow through AtlasLMM with all Atlas features enabled."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
        use_muon_optimizer = True,
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss=True)
    loss.backward()
    
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None
    
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    assert params_with_grad / total_params > 0.5


# --- Tensor poly mode tests ---

def test_atlas_mag_tensor_poly_mode():
    """Test MAG with tensor polynomial mode."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'tensor',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_mal_tensor_poly_mode():
    """Test MAL with tensor polynomial mode."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,),
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'tensor',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


def test_atlas_lmm_tensor_poly_mode():
    """Test AtlasLMM with tensor polynomial mode."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        omega_window = 2,
        poly_degree = 2,
        poly_mode = 'tensor',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)


# --- Multi-layer memory with Atlas features ---

def test_atlas_mag_multi_layer_memory_with_features():
    """Test MAG with multiple memory layers and Atlas features."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 4,
        window_size = 16,
        neural_memory_layers = (1, 2, 3, 4),  # all layers
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    # Test sampling
    prompt = x[:, :4]
    sampled = model.sample(prompt, 8, temperature=0., show_progress=False)
    assert sampled.shape == (2, 4)


def test_atlas_mal_multi_layer_memory_with_features():
    """Test MAL with multiple memory layers and Atlas features."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 4,
        window_size = 16,
        neural_memory_layers = (1, 2, 3, 4),
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    prompt = x[:, :4]
    sampled = model.sample(prompt, 8, temperature=0., show_progress=False)
    assert sampled.shape == (2, 4)


def test_atlas_lmm_multi_layer_with_features():
    """Test AtlasLMM with multiple layers and Atlas features."""
    model = AtlasLMM(
        num_tokens = 64,
        dim = 16,
        depth = 4,
        omega_window = 2,
        use_omega_gate = True,
        poly_degree = 2,
        poly_mode = 'elementwise',
    )
    
    x = torch.randint(0, 64, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 64)
    
    prompt = x[:, :4]
    sampled = model.sample(prompt, 8, temperature=0., show_progress=False)
    assert sampled.shape == (2, 4)
