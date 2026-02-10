"""
Memory as Layer (MAL) Transformer and Pure Atlas (LMM) for Atlas

From Titans paper Section 4.3 + Atlas extensions:

MAL:
- Neural memory is used as a layer BEFORE attention
- Memory compresses past/current context, then attention operates on compressed representation
- Uses sliding window attention
- Structure: x -> OmegaNeuralMemory -> Attention -> output
- Supports Omega rule and Muon optimizer

Pure Atlas (LMM):
- Only uses OmegaNeuralMemory, no attention
- Tests the memory module's standalone capability
- From paper: "a long-term memory module should still be a powerful model even without short-term memory"
- Supports Omega rule, polynomial features, and Muon optimizer
"""

from __future__ import annotations
from typing import Callable

from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import repeat, rearrange, pack, unpack
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend

from atlas_pytorch.omega import OmegaNeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# sliding window attention (simpler version for MAL)

class SlidingWindowAttention(Module):
    def __init__(
        self,
        dim,
        window_size,
        num_persist_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        
        dim_inner = dim_head * heads
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal = True, **attend_kwargs)
        
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)
        
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        # persistent memory
        self.num_persist_mem_tokens = num_persist_mem_tokens
        if num_persist_mem_tokens > 0:
            self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))
        else:
            self.persistent_memory = None
    
    def forward(
        self,
        seq,
        cache = None,
    ):
        batch, seq_len = seq.shape[:2]
        
        # check if we have valid cache (not None and contains tensors)
        has_cache = exists(cache) and cache[0] is not None
        
        # normalize
        seq = self.norm(seq)
        
        # qkv projection
        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))
        
        # handle caching for inference
        if has_cache:
            ck, cv = cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)
        
        next_cache = (k, v)
        
        # rotary embeddings
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        
        # sliding window masking
        attend_kwargs = dict()
        
        if not has_cache and seq_len > self.window_size:
            # create sliding window mask
            device = seq.device
            q_idx = torch.arange(seq_len, device = device).view(1, 1, seq_len, 1)
            k_idx = torch.arange(k.shape[-2], device = device).view(1, 1, 1, -1)
            
            # causal + window constraint
            causal_mask = q_idx >= k_idx
            window_mask = (q_idx - k_idx) <= self.window_size
            mask = causal_mask & window_mask
            
            # persistent memory always visible
            if self.num_persist_mem_tokens > 0:
                mask = F.pad(mask, (self.num_persist_mem_tokens, 0), value = True)
            
            attend_kwargs['mask'] = mask
        
        # add persistent memory
        if exists(self.persistent_memory):
            pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)
            k = cat((pmk, k), dim = -2)
            v = cat((pmv, v), dim = -2)
        
        # attention
        out, _ = self.attend(q, k, v, **attend_kwargs)
        
        out = self.merge_heads(out)
        out = self.to_out(out)
        
        # trim cache to window size for efficiency (only during inference)
        if has_cache:
            max_cache_len = self.window_size + self.num_persist_mem_tokens
            next_cache = (
                next_cache[0][..., -max_cache_len:, :],
                next_cache[1][..., -max_cache_len:, :]
            )
        
        return out, next_cache


# MAL Transformer with Atlas (OmegaNeuralMemory)

class MemoryAsLayerTransformer(Module):
    """
    Memory as Layer (MAL) architecture from Titans paper with Atlas extensions.
    
    Memory is applied as a layer before attention:
    x -> OmegaNeuralMemory -> SlidingWindowAttention -> output
    
    This is similar to hybrid architectures like H3 where
    recurrent models are stacked with attention.
    
    Atlas extensions:
    - Supports omega_window > 1 for context memorization
    - Supports polynomial feature mappings
    - Supports Muon optimizer for memory updates
    """
    
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        window_size = 64,
        num_persist_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        token_emb: Module | None = None,
        # Atlas-specific kwargs
        omega_window: int = 1,
        use_omega_gate: bool = False,
        poly_degree: int = 2,
        poly_mode: str = 'polysketch',
        use_muon_optimizer: bool = False,
    ):
        super().__init__()
        
        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)
        
        self.token_emb = token_emb
        self.num_tokens = num_tokens
        self.dim = dim
        
        # layers
        self.layers = ModuleList([])
        
        layers = tuple(range(1, depth + 1))
        
        # default: all layers have neural memory
        if not exists(neural_memory_layers):
            neural_memory_layers = layers
        
        for layer in layers:
            # neural memory layer (applied first)
            mem = None
            if layer in neural_memory_layers:
                mem = OmegaNeuralMemory(
                    dim = dim,
                    chunk_size = 1,  # token-by-token, no segmentation
                    model = deepcopy(neural_memory_model) if exists(neural_memory_model) else None,
                    omega_window = omega_window,
                    use_omega_gate = use_omega_gate,
                    poly_degree = poly_degree,
                    poly_mode = poly_mode,
                    **neural_memory_kwargs
                )
                
                # enable Muon optimizer if requested
                if use_muon_optimizer:
                    mem.use_muon_optimizer = True
            
            # sliding window attention (applied after memory)
            attn = SlidingWindowAttention(
                dim = dim,
                window_size = window_size,
                num_persist_mem_tokens = num_persist_mem_tokens,
                dim_head = dim_head,
                heads = heads,
            )
            
            # feedforwards: one after memory (pre-attn), one after attention (post-attn)
            ff_pre = FeedForward(dim = dim, mult = ff_mult)
            ff_post = FeedForward(dim = dim, mult = ff_mult)
            
            self.layers.append(ModuleList([mem, attn, ff_pre, ff_post]))
        
        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)
        
        # for device tracking
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        
        self.window_size = window_size
        self.num_persist_mem_tokens = num_persist_mem_tokens
    
    @torch.no_grad()
    def sample(
        self,
        prompt,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(min_p = 0.1),
        show_progress = True,
        use_cache = True
    ):
        was_training = self.training
        self.eval()
        
        prompt_seq_len = prompt.shape[-1]
        out = prompt.clone()
        
        cache = None
        
        progress = tqdm.tqdm(range(seq_len - prompt_seq_len), desc = 'sampling', disable = not show_progress)
        
        for _ in progress:
            if use_cache:
                logits, cache = self.forward(out[:, -1:], cache = cache, return_cache = True)
            else:
                logits = self.forward(out)
                logits = logits[:, -1:]
            
            logits = filter_fn(logits[:, -1], **filter_kwargs)
            sampled = gumbel_sample(logits, temperature = temperature)
            out = cat((out, sampled), dim = -1)
        
        self.train(was_training)
        return out[..., prompt_seq_len:]
    
    def forward(
        self,
        x,
        return_loss = False,
        cache = None,
        return_cache = False,
    ):
        """
        Forward pass for MAL with Atlas (OmegaNeuralMemory).
        
        Args:
            x: input token ids (batch, seq_len)
            return_loss: if True, compute cross-entropy loss
            cache: tuple of (attn_caches, mem_states) from previous forward
            return_cache: if True, return cache for next forward
            
        Returns:
            logits or loss, optionally with cache
            cache format: (attn_caches, mem_states) where
                - attn_caches: list of (k, v) tuples for each layer
                - mem_states: list of NeuralMemState for each layer
        """
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]
        
        batch, seq_len = x.shape
        
        # token embedding
        x = self.token_emb(x)
        
        # handle caching - cache is now (attn_caches, mem_states)
        if not exists(cache):
            attn_caches = [None for _ in self.layers]
            mem_states = [None for _ in self.layers]
        else:
            attn_caches, mem_states = cache
        
        next_attn_caches = []
        next_mem_states = []
        
        # process through layers
        for layer_idx, (mem, attn, ff_pre, ff_post) in enumerate(self.layers):
            attn_cache = attn_caches[layer_idx]
            mem_state = mem_states[layer_idx]
            
            # memory layer first (if exists)
            if exists(mem):
                mem_out, next_mem_state = mem(x, state = mem_state)
                next_mem_states.append(next_mem_state)
                x = x + mem_out  # residual connection
            else:
                next_mem_states.append(None)
            
            # pre-attention feedforward
            x = x + ff_pre(x)
            
            # then sliding window attention
            attn_out, next_attn_cache = attn(x, cache = attn_cache)
            next_attn_caches.append(next_attn_cache)
            
            x = x + attn_out
            
            # post-attention feedforward
            x = x + ff_post(x)
        
        # final norm and logits
        x = self.norm(x)
        logits = self.to_logits(x)
        
        if not return_loss:
            if return_cache:
                return logits, (next_attn_caches, next_mem_states)
            return logits
        
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)


# Pure Atlas (LMM only) with OmegaNeuralMemory

class AtlasLMM(Module):
    """
    Pure Atlas - Long-term Memory Module only (no attention).
    
    From paper Section 4.3:
    "a long-term memory module should still be a powerful model 
    even without short-term memory (i.e., attention)"
    
    This tests the OmegaNeuralMemory's standalone capability as a sequence model.
    
    Atlas extensions:
    - Supports omega_window > 1 for context memorization
    - Supports polynomial feature mappings
    - Supports Muon optimizer for memory updates
    """
    
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        num_persist_mem_tokens = 0,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        ff_mult = 4,
        token_emb: Module | None = None,
        # Atlas-specific kwargs
        omega_window: int = 1,
        use_omega_gate: bool = False,
        poly_degree: int = 2,
        poly_mode: str = 'polysketch',
        use_muon_optimizer: bool = False,
    ):
        super().__init__()
        
        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)
        
        self.token_emb = token_emb
        self.num_tokens = num_tokens
        self.dim = dim
        
        # persistent memory as learnable prefix
        self.num_persist_mem_tokens = num_persist_mem_tokens
        if num_persist_mem_tokens > 0:
            self.persistent_memory = nn.Parameter(torch.randn(num_persist_mem_tokens, dim) * 0.02)
        else:
            self.persistent_memory = None
        
        # stack of memory layers with feedforward
        self.layers = ModuleList([])
        
        for _ in range(depth):
            mem = OmegaNeuralMemory(
                dim = dim,
                chunk_size = 1,  # token-by-token
                model = deepcopy(neural_memory_model) if exists(neural_memory_model) else None,
                omega_window = omega_window,
                use_omega_gate = use_omega_gate,
                poly_degree = poly_degree,
                poly_mode = poly_mode,
                **neural_memory_kwargs
            )
            
            # enable Muon optimizer if requested
            if use_muon_optimizer:
                mem.use_muon_optimizer = True
            
            ff = FeedForward(dim = dim, mult = ff_mult)
            
            self.layers.append(ModuleList([mem, ff]))
        
        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)
        
        # for device tracking
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
    
    @torch.no_grad()
    def sample(
        self,
        prompt,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(min_p = 0.1),
        show_progress = True,
        use_cache = True  # LMM uses memory state cache
    ):
        was_training = self.training
        self.eval()
        
        prompt_seq_len = prompt.shape[-1]
        out = prompt.clone()
        
        cache = None
        
        progress = tqdm.tqdm(range(seq_len - prompt_seq_len), desc = 'sampling', disable = not show_progress)
        
        for _ in progress:
            if use_cache:
                logits, cache = self.forward(out[:, -1:], cache = cache, return_cache = True)
            else:
                logits = self.forward(out)
                logits = logits[:, -1:]
            
            logits = filter_fn(logits[:, -1], **filter_kwargs)
            sampled = gumbel_sample(logits, temperature = temperature)
            out = cat((out, sampled), dim = -1)
        
        self.train(was_training)
        return out[..., prompt_seq_len:]
    
    def forward(
        self,
        x,
        return_loss = False,
        cache = None,
        return_cache = False,
    ):
        """
        Forward pass for AtlasLMM (Pure Atlas with OmegaNeuralMemory).
        
        Args:
            x: input token ids (batch, seq_len)
            return_loss: if True, compute cross-entropy loss
            cache: list of NeuralMemState for each layer from previous forward
            return_cache: if True, return cache for next forward
            
        Returns:
            logits or loss, optionally with cache
            cache format: list of NeuralMemState for each layer
        """
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]
        
        batch, seq_len = x.shape
        
        # token embedding
        x = self.token_emb(x)
        
        # prepend persistent memory if exists
        if exists(self.persistent_memory):
            persist = repeat(self.persistent_memory, 'n d -> b n d', b = batch)
            x = cat((persist, x), dim = 1)
        
        # handle caching
        if not exists(cache):
            mem_states = [None for _ in self.layers]
        else:
            mem_states = cache
        
        next_mem_states = []
        
        # process through memory layers
        for layer_idx, (mem, ff) in enumerate(self.layers):
            mem_state = mem_states[layer_idx]
            
            mem_out, next_mem_state = mem(x, state = mem_state)
            next_mem_states.append(next_mem_state)
            
            x = x + mem_out
            x = x + ff(x)
        
        # remove persistent memory tokens from output
        if exists(self.persistent_memory):
            x = x[:, self.num_persist_mem_tokens:]
        
        # final norm and logits
        x = self.norm(x)
        logits = self.to_logits(x)
        
        if not return_loss:
            if return_cache:
                return logits, next_mem_states
            return logits
        
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)

