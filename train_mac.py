import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from atlas_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)
import atlas_pytorch.mac_transformer as mac
from atlas_pytorch.omega import OmegaNeuralMemory
import argparse

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4                      # set smaller for more granularity for learning rate / momentum etc
NEURAL_MEM_BATCH_SIZE = 128                     # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
NEURAL_MEM_SPEC_NORM_SURPRISES = True           # applying lessons from Muon optimizer to surprise updates, by spectral norming the surprises

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

# perf related

USE_FLEX_ATTN = True  # will be overridden after argparse (default: enabled for L4 GPU)
USE_FAST_INFERENCE = False

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# -----------------------------------------------------------------------------
# CLI to select model variant: 'titans' (baseline), 'omeganet', or 'atlas'
# OmegaNet = OmegaNeuralMemory with SGD; Atlas = OmegaNeuralMemory + Muon
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='titans', choices=['titans', 'omeganet', 'atlas'])
parser.add_argument('--omega-window', type=int, default=2)
parser.add_argument('--use-omega-gate', action='store_true', default=False)
parser.add_argument('--poly-mode', type=str, default='elementwise', choices=['off', 'elementwise', 'tensor'])
parser.add_argument('--poly-degree', type=int, default=2)
# device / runtime overrides
parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps', 'auto'])
parser.add_argument('--num-batches', type=int, default=NUM_BATCHES)
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('--grad-accum', type=int, default=GRADIENT_ACCUMULATE_EVERY)
parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('--validate-every', type=int, default=VALIDATE_EVERY)
parser.add_argument('--generate-every', type=int, default=GENERATE_EVERY)
parser.add_argument('--prime-length', type=int, default=PRIME_LENGTH)
parser.add_argument('--generate-length', type=int, default=GENERATE_LENGTH)
parser.add_argument('--should-generate', action='store_true', default=True)
parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
parser.add_argument('--data-path', type=str, default='./data/enwik8.gz')
# model size overrides
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dim-head', type=int, default=64)
# wandb toggle
parser.add_argument('--wandb', action='store_true', help='Enable wandb logging if installed')
parser.add_argument('--no-accelerated-scan', action='store_true', help='Disable accelerated assoc_scan backend (triton)')
parser.add_argument('--no-flex-attn', action='store_true', help='Disable flex attention (for environments without triton support)')
args, _ = parser.parse_known_args()

# wandb experiment tracker (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

wandb_log = lambda data: None
if args.wandb:
    if WANDB_AVAILABLE:
        wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME
        wandb.run.save()
        wandb_log = wandb.log
    else:
        print("wandb not installed; skipping wandb logging.")

MODEL_TYPE = args.model
OMEGA_WINDOW = args.omega_window
USE_FLEX_ATTN = not args.no_flex_attn
USE_OMEGA_GATE = args.use_omega_gate
POLY_MODE = args.poly_mode
POLY_DEGREE = args.poly_degree
USE_ACCELERATED_SCAN = not args.no_accelerated_scan

NUM_BATCHES = args.num_batches
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATE_EVERY = args.grad_accum
LEARNING_RATE = args.learning_rate
VALIDATE_EVERY = args.validate_every
GENERATE_EVERY = args.generate_every
PRIME_LENGTH = args.prime_length
GENERATE_LENGTH = args.generate_length
SHOULD_GENERATE = args.should_generate
SEQ_LEN = args.seq_len

# memory model

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = args.dim_head  # must match dim_head passed to neural_memory_kwargs
    )
else:
    neural_memory_model = MemoryMLP(
        dim = args.dim_head,  # must match dim_head passed to neural_memory_kwargs
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-context transformer

# swap in Omega memory for omeganet / atlas
if MODEL_TYPE in ('omeganet', 'atlas'):
    mac.NeuralMemory = OmegaNeuralMemory

# base kwargs shared across variants
neural_memory_kwargs = dict(
    dim_head = args.dim_head,
    heads = args.heads,
    attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
    qk_rmsnorm = NEURAL_MEM_QK_NORM,
    momentum = NEURAL_MEM_MOMENTUM,
    momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
    default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
    use_accelerated_scan = USE_ACCELERATED_SCAN,
    per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR,
    spectral_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES
)

# add omega-specific knobs for omeganet / atlas
if MODEL_TYPE in ('omeganet', 'atlas'):
    neural_memory_kwargs.update(
        dict(
            omega_window = OMEGA_WINDOW,
            use_omega_gate = USE_OMEGA_GATE,
            poly_mode = POLY_MODE,
            poly_degree = POLY_DEGREE
        )
    )

def pick_device():
    if args.device and args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()

model = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = args.dim,
    depth = args.depth,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = neural_memory_kwargs
).to(device)

# enable Muon optimizer for Atlas by toggling flag on each memory layer
if MODEL_TYPE == 'atlas':
    for layer in model.layers:
        mem = layer[4]
        if mem is not None:
            mem.use_muon_optimizer = True

# prepare enwik8 data

with gzip.open(args.data_path) as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN, device)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, device)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb_log(dict(loss = loss.item()))

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
        output_str = decode_tokens(sample[0])
        print(output_str)
