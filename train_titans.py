"""
Training script for Pure Titans (LMM - Long-term Memory Module only).

Pure Titans uses only neural memory without attention.
This tests the memory module's standalone capability as a sequence model.
From paper: "a long-term memory module should still be a powerful model 
even without short-term memory (i.e., attention)"
"""

import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from titans_pytorch import (
    TitansLMM,
    MemoryMLP,
    MemoryAttention
)

# CLI (minimal) - align with other scripts: accelerated scan is opt-in
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use-accelerated-scan', action='store_true', help='Enable accelerated assoc_scan backend when available')
parser.add_argument('--wandb', action='store_true', help='Enable wandb logging if installed')
args, _ = parser.parse_known_args()

# constants

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
MODEL_DEPTH = 8                           # number of memory layers
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False

# experiment related

PROJECT_NAME = 'titans-pure-lmm'
RUN_NAME = f'pure titans - depth {MODEL_DEPTH}, persist {NUM_PERSIST_MEM}'
WANDB_ONLINE = False

# perf related

USE_ACCELERATED_SCAN = args.use_accelerated_scan

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

# instantiate pure titans (LMM only)

model = TitansLMM(
    num_tokens = 256,
    dim = 384,
    depth = MODEL_DEPTH,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
    )
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
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

        # LMM doesn't use traditional KV cache
        sample = model.sample(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)

