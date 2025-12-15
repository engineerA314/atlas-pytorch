"""
Training script for MAL (Memory as Layer) Transformer with Atlas.

MAL uses OmegaNeuralMemory as a layer before sliding window attention.
Memory compresses context, then attention operates on compressed representation.
Supports Omega rule, polynomial features, and Muon optimizer.
"""

import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from atlas_pytorch import (
    MemoryAsLayerTransformer,
    MemoryMLP,
    MemoryAttention
)
import argparse

# constants (overridable via CLI)
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)           # layers with neural memory
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 64                         # sliding window size

# Atlas-specific settings (defaults; overridable via CLI)
OMEGA_WINDOW = 2                         # context window for Omega rule
USE_OMEGA_GATE = False                   # use learned U gate for Omega
POLY_DEGREE = 1                          # polynomial feature degree
POLY_MODE = 'off'                        # 'off', 'elementwise', 'tensor'
USE_MUON_OPTIMIZER = False               # use Muon optimizer for memory

# experiment related

PROJECT_NAME = 'atlas-mal-transformer'
RUN_NAME = f'mal - window {WINDOW_SIZE}, omega {OMEGA_WINDOW}, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False

# CLI overrides
parser = argparse.ArgumentParser()
parser.add_argument('--omega-window', type=int, default=OMEGA_WINDOW)
parser.add_argument('--use-omega-gate', action='store_true', default=USE_OMEGA_GATE)
parser.add_argument('--poly-mode', type=str, default=POLY_MODE, choices=['off', 'elementwise', 'tensor'])
parser.add_argument('--poly-degree', type=int, default=POLY_DEGREE)
parser.add_argument('--use-muon-optimizer', action='store_true', default=USE_MUON_OPTIMIZER)
parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps', 'auto'])
parser.add_argument('--num-batches', type=int, default=NUM_BATCHES)
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('--grad-accum', type=int, default=GRADIENT_ACCUMULATE_EVERY)
parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('--validate-every', type=int, default=VALIDATE_EVERY)
parser.add_argument('--generate-every', type=int, default=GENERATE_EVERY)
parser.add_argument('--prime-length', type=int, default=PRIME_LENGTH)
parser.add_argument('--generate-length', type=int, default=GENERATE_LENGTH)
parser.add_argument('--should-generate', action='store_true', default=SHOULD_GENERATE)
parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
parser.add_argument('--data-path', type=str, default='./data/enwik8.gz')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dim-head', type=int, default=64)
# wandb toggle
parser.add_argument('--wandb', action='store_true', help='Enable wandb logging if installed')
parser.add_argument('--use-accelerated-scan', action='store_true', help='Enable accelerated assoc_scan backend when available')
args, _ = parser.parse_known_args()

OMEGA_WINDOW = args.omega_window
USE_OMEGA_GATE = args.use_omega_gate
POLY_MODE = args.poly_mode
POLY_DEGREE = args.poly_degree
USE_MUON_OPTIMIZER = args.use_muon_optimizer
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
USE_ACCELERATED_SCAN = args.use_accelerated_scan

# perf related

#
USE_FAST_INFERENCE = True

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

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# memory model

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-layer transformer with Atlas

def pick_device():
    if args.device and args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()

model = MemoryAsLayerTransformer(
    num_tokens = 256,
    dim = args.dim,
    depth = args.depth,
    window_size = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_model = neural_memory_model,
    # Atlas-specific kwargs
    omega_window = OMEGA_WINDOW,
    use_omega_gate = USE_OMEGA_GATE,
    poly_degree = POLY_DEGREE,
    poly_mode = POLY_MODE,
    use_muon_optimizer = USE_MUON_OPTIMIZER,
    neural_memory_kwargs = dict(
        dim_head = args.dim_head,
        heads = args.heads,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
    )
).to(device)

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
val_dataset = TextSamplerDataset(data_val, SEQ_LEN, device)
train_loader = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

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

