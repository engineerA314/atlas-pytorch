from __future__ import annotations

from itertools import zip_longest

import torch
from torch import nn, Tensor, stack
from torch.func import functional_call
from einops import rearrange, repeat
from tensordict import TensorDict

from .neural_memory import (
    NeuralMemory,
    NeuralMemState,
    exists,
    round_down_multiple,
    round_up_multiple,
    pad_at_dim,
    dict_get_value_shapes,
    rearrange_dict_values,
    repeat_dict_values,
    newtonschulz5
)

def _sliding_sum_along_n(x: Tensor, window: int) -> Tensor:
    # x: ['(bh) n ...'] -> sum of last `window` along n-dim (dim=1)
    if window <= 1:
        return x
    n = x.shape[1]
    window = min(window, n)
    c = x.cumsum(dim = 1)
    zeros = torch.zeros(*c.shape[:1], window, *c.shape[2:], device = c.device, dtype = c.dtype)
    shifted = torch.cat((zeros, c[:, :-window]), dim = 1)
    return c - shifted

class OmegaNeuralMemory(NeuralMemory):
    def __init__(
        self,
        *args,
        omega_window: int = 1,
        use_omega_gate: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',  # 'off' | 'elementwise' | 'tensor'
        **kwargs
    ):
        # pop atlas-only kwargs before passing to base class
        use_qkv_convs = kwargs.pop('use_qkv_convs', None)
        super().__init__(*args, **kwargs)
        assert omega_window >= 1
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.poly_degree = int(poly_degree)
        self.poly_mode = poly_mode
        # auto policy: convs off for e=1 to preserve parity with base in tests
        self.use_qkv_convs = (omega_window > 1) if use_qkv_convs is None else bool(use_qkv_convs)
        # gate computed at chunk resolution: (b n d) -> (b n h) -> ((b h) n 1)
        self._omega_gate_linear = nn.Linear(self.to_decay_factor[0].in_features, self.heads)
        self._omega_gate_act = nn.Sigmoid()
        # cache for tensor-mode random projections (non-trainable)
        self._poly_proj_cache = {}

        # safe query feature wrapper without parent cycles
        class _PolyQueryMap(nn.Module):
            def __init__(self, apply_fn):
                super().__init__()
                self.apply_fn = apply_fn
            def forward(self, q: Tensor) -> Tensor:
                b, h, n, d = q.shape
                q2 = rearrange(q, 'b h n d -> (b h n) d')
                q2 = self.apply_fn(q2)
                return rearrange(q2, '(b h n) d -> b h n d', b = b, h = h, n = n)
        self.q_norm = nn.Sequential(self.q_norm, _PolyQueryMap(self._apply_poly_features))
        # depthwise convs on sequence axis for Q/K/V (kernel_size=1 to preserve length)
        # infer input dim from first linear in to_keys
        first_linear = self.to_keys[0] if isinstance(self.to_keys, nn.Sequential) else self.to_keys
        dim = first_linear.in_features
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=False)
    def _apply_seq_conv(self, seq: Tensor, conv: nn.Module) -> Tensor:
        # seq: (b, n, d) -> (b, d, n) -> conv1d -> (b, n, d)
        if not isinstance(seq, torch.Tensor) or seq.ndim != 3:
            return seq
        # guard empty-length sequences (e.g., when round_down_seq_len == 0)
        if seq.shape[1] == 0:
            return seq
        seq_t = rearrange(seq, 'b n d -> b d n')
        out = conv(seq_t)
        out = rearrange(out, 'b d n -> b n d')
        return out
    def _apply_poly_features(self, x: Tensor) -> Tensor:
        # x: [... d]
        if self.poly_degree <= 1 or self.poly_mode == 'off':
            return x
        if self.poly_mode == 'elementwise':
            # sum of element-wise powers up to degree g (Taylor-like without 1/i! by default)
            out = torch.zeros_like(x)
            for i in range(1, self.poly_degree + 1):
                out = out + x.pow(i)
            return out
        if self.poly_mode == 'tensor':
            # degree-2 tensor expansion with fixed random projection back to original dim
            # higher degrees fall back to degree=2 interactions for practicality
            d = x.shape[-1]
            # build features: [x, vec(x ⊗ x)]
            outer = torch.einsum('... i, ... j -> ... i j', x, x)
            outer = outer.reshape(*x.shape[:-1], d * d)
            feats = torch.cat((x, outer), dim = -1)
            feat_dim = feats.shape[-1]
            proj_key = (feat_dim, d, x.device.type)
            proj = self._poly_proj_cache.get(proj_key)
            if proj is None:
                # fixed random projection, non-learned
                proj_tensor = torch.randn(feat_dim, d, device = x.device, dtype = x.dtype) / (feat_dim ** 0.5)
                self._poly_proj_cache[proj_key] = proj_tensor
                proj = proj_tensor
            return feats @ proj
        # default safe return
        return x

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True
    ):
        # strict equivalence to base when omega has no effect
        if self.omega_window == 1 and not self.use_omega_gate:
            return super().store_memories(
                seq,
                weights = weights,
                past_state = past_state,
                seq_index = seq_index,
                prev_weights = prev_weights,
                mask = mask,
                return_surprises = return_surprises
            )
        # shapes
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # truncate to full chunks
        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size
        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]
        next_seq_len_index = seq_index + round_down_seq_len

        # init weights
        if not exists(weights):
            weights = self.init_weights(batch)
        weights = TensorDict(weights)

        # broadcast weights per chunk for surprise calculation
        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)
        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # norms
        seq = self.store_norm(seq)
        values_seq = seq
        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # learned hparams
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)
        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)       # ['b n d']
        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()                 # ['(bh) n 1']

        has_momentum = exists(self.to_momentum)
        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()            # ['o (bh) n 1']
            learned_combine = exists(self.to_learned_momentum_combine)
            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)  # ['o (bh) n']

        # keys / values (optionally apply convs on sequence before projections)
        if self.use_qkv_convs:
            seq_k = self._apply_seq_conv(seq, self.k_conv)
            seq_v = self._apply_seq_conv(values_seq, self.v_conv)
        else:
            seq_k = seq
            seq_v = values_seq
        keys = self.to_keys(seq_k)
        # apply polynomial mapping on keys before splitting heads
        keys = self._apply_poly_features(keys)
        values = self.to_values(seq_v)
        keys, values = map(self.split_kv_heads, (keys, values))                    # ['b h (n c u) d']
        keys = self.k_norm(keys)

        # rearrange into per-chunk matmul shapes
        keys, values = tuple(
            rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates)
            for t in (keys, values)
        )

        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # optional mask for storing
        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)
            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # residual from previous layer (optional)
        if exists(prev_weights):
            start_index = torch.div(seq_index, chunk_size, rounding_mode = 'ceil')
            end_index = start_index + num_chunks
            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])
            weights_for_surprise = weights_for_surprise + prev_weights

        # per-sample grads
        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)
        grads = TensorDict(grads)

        adaptive_lr = rearrange(adaptive_lr, '(b n) c -> b n c', b = batch * heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b n) c -> b n c', b = batch * heads)

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: t if t.numel() == 0 else (t.reshape(t.shape[0], t.shape[1], -1).renorm(p=2, dim=2, maxnorm=self.max_grad_norm).view_as(t)))

        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)   # ['(bh) n ...']
        surprises = grads.mul(-1)

        # init state
        if not exists(past_state):
            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)
            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early exit for no chunks
        if num_chunks == 0:
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)
            output = (updates, next_store_state)
            if not return_surprises:
                return output
            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # omega gate U at chunk resolution -> ['(bh) n 1']
        if self.use_omega_gate:
            omega_gate = self._omega_gate_act(self._omega_gate_linear(chunked_seq))   # ['b n h']
            omega_gate = rearrange(omega_gate, 'b n h -> (b h) n 1')
        else:
            omega_gate = None

        updates = TensorDict()
        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        # select assoc_scan once based on first surprise tensor device (all tensors on same device)
        first_surprise = next(iter(surprises.values()))
        assoc_scan = self._get_assoc_scan(first_surprise)

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):
            base_surprise = surprise

            if has_momentum:
                momentum = base_surprise
                momentums = []
                last_momentum = past_last_momentum[param_name]
                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum)
                    momentums.append(momentum)
                momentums = stack(momentums)                                   # ['o (bh) n ...']
                next_last_momentum[param_name] = momentums[:, :, -1]
                if exists(self.to_learned_momentum_combine):
                    base_surprise = torch.einsum('o b n, o b n ... -> b n ...', combine_momentums, momentums)
                else:
                    base_surprise = momentums[-1]

            # apply U-gate with explicit broadcasting over parameter dims
            if exists(omega_gate):
                gate = omega_gate
                while gate.ndim < base_surprise.ndim:
                    gate = gate.unsqueeze(-1)
                gated_surprise = base_surprise * gate
            else:
                gated_surprise = base_surprise

            # Omega: aggregate last e steps per chunk with carry-over buffer across chunks
            if self.omega_window > 1:
                buf_key = f'{param_name}__omega_buf'
                if buf_key in past_last_momentum.keys():
                    prev_buf = past_last_momentum[buf_key]
                else:
                    prev_buf = torch.zeros(
                        (gated_surprise.shape[0], self.omega_window - 1, *gated_surprise.shape[2:]),
                        dtype = gated_surprise.dtype,
                        device = gated_surprise.device
                    )
                surprise_ext = torch.cat((prev_buf, gated_surprise), dim = 1)               # ['(bh) (e-1+n) ...']
                win_sum_ext = _sliding_sum_along_n(surprise_ext, self.omega_window)         # same length
                update = win_sum_ext[:, -(gated_surprise.shape[1]):, ...]                   # keep last n
                # save buffer for next chunk
                next_last_momentum[buf_key] = surprise_ext[:, -(self.omega_window - 1):, ...]
            else:
                update = gated_surprise

            if self.spectral_norm_surprises:
                update = newtonschulz5(update)

            # scale initial prev by first-step gate to suppress prior carry if gate ~ 0
            initial_prev = last_update

            # short-circuit: if all gates ~ 0 across this chunk, force updates to zero (after suppressing prev)
            if exists(omega_gate) and torch.all(omega_gate.amax(dim = 1) < 1e-3):
                zero_update = torch.zeros_like(update)
                update = zero_update
            else:
                update = assoc_scan(1. - decay_factor, update, prev = initial_prev, remove_prev = False)
                # Muon optimizer: apply Newton-Schulz step after recurrence to preserve shapes
                if getattr(self, 'use_muon_optimizer', False):
                    update = newtonschulz5(update)

            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        next_state = (next_last_update, next_last_momentum)
        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    # guard retrieval weights to avoid empty TensorDicts passed to base retrieve
    def retrieve_memories(
        self,
        seq,
        weights: dict[str, Tensor],
    ):
        # apply conv on queries before delegating
        if self.use_qkv_convs and seq is not None and isinstance(seq, torch.Tensor) and seq.ndim == 3:
            seq = self._apply_seq_conv(seq, self.q_conv)
        # ensure non-empty dict of parameters for functional_call
        need_init = False
        if isinstance(weights, TensorDict):
            need_init = len(weights.keys()) == 0
        else:
            need_init = not bool(weights)
        if need_init:
            # fall back to parameter dict (per-head learned parameter shapes)
            weights = self.memory_model_parameter_dict
        return super().retrieve_memories(seq, weights)

