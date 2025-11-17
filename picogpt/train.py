from dataclasses import dataclass
import inspect
import math
import os
import torch
import numpy as np
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import argparse

parser = argparse.ArgumentParser(description='PicoGPT Training')

# Run configurations
parser.add_argument('--baseline', action='store_true', help='Run baseline configuration (current code)')
parser.add_argument('--run-1', action='store_true', help='Run configuration 1 (same as baseline)')
parser.add_argument('--run-2', action='store_true', help='Run configuration 2 (Muon + RoPE + Data fixes)')

# Individual feature flags
parser.add_argument('--use-muon', action='store_true', help='Use Muon optimizer instead of AdamW')
parser.add_argument('--use-rope', action='store_true', help='Use RoPE instead of learned positional embeddings')
parser.add_argument('--data-fixes', action='store_true', help='Enable data shuffling and periodicity fixes')

args = parser.parse_args()

# Initialize defaults
if not (args.baseline or args.run_1 or args.run_2 or args.use_muon or args.use_rope or args.data_fixes):
    # if no args, default to baseline
    args.use_muon = False
    args.use_rope = False
    args.data_fixes = False
# if individual flags are set without run configs they remain as parsed
# run configurations (these override individual flags)
if args.run_1 or args.baseline:
    args.use_muon = False
    args.use_rope = False
    args.data_fixes = False
elif args.run_2:
    args.use_muon = True
    args.use_rope = True
    args.data_fixes = True


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 #number of tokens 50,000 bpe merges, 256 byte tokens, 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12
    n_embd: int = 768 # embedding dim 
    use_rope: bool = False #use rotary pos embedding instead of learned
    learning_rate = 6e-4 #gpt-3-small max_lr

class RoPEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al. (2021) https://arxiv.org/pdf/2104.09864 """

    def __init__(self, dim, max_seq_len=2048, base=10000): #2048 for inference flexibility
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        #pre compute freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        #if seq length changes update the cos/sin vaules
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        #rotate half the hidden dims of the input
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos, sin = self._update_cos_sin_cache(seq_len, q.device, q.dtype)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query , value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        #out projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.PICOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_rope = config.use_rope

        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rope = RoPEmbedding(head_dim, max_seq_len=config.block_size)

        #more of a mask but following openai naming 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1,1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, emedding dim (n_embd)
        #nh is number of heads, hs is head size, c is number of channels nh*hs
        #e.g. gpt-2 124M has nh = 12, hs=64, so channels are 768=nh*hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, nh, T, hs)
        
        #regular self-attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]  == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)

        #apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)
        
        #flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) #reassemble all head outputs side by side
        #output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh') #relu but not exactly flat tail  
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.PICOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        transformer_dict = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd), #token embedding
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        }
    
        # add learned positional embeddings if not using RoPE
        if not config.use_rope:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
    
        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight tying scheme
        self.transformer.wte.weight = self.lm_head.weight #copies the data pointer which is the same
        
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'PICOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence length of {T}, block size is {self.config.block_size}'
        
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B. T, n_embd)
        
        #add pos info (either learned or RoPE handles it in attention)
        if self.config.use_rope:
            x = tok_emb #rope is applied in attention layers
        else:
            #forward token and use pos embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
            pos_emb = self.transformer.wpe(pos) #pos embeddings of shape (T, n_embd)
            x = tok_emb + pos_emb

        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device, use_muon=False):
        #start with all canidate params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        if use_muon:
            #separate params: Muon for 2d weights, AdamW for everything else
            muon_params = []
            adamw_params = []

            for name, param in param_dict.items():
                #muon: 2d weight matricies (not embeddings or final layer)
                if (param.ndim == 2 and
                    'wte' not in name and
                    'wpe' not in name and
                    'lm_head' not in name):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
                
            num_muon_params = sum(p.numel() for p in muon_params)
            num_adamw_params = sum(p.numel() for p in adamw_params)
            print(f"Muon optimizer: {len(muon_params)} tensors with {num_muon_params:,} parameters")
            print(f"AdamW optimizer: {len(adamw_params)} tensors with {num_adamw_params:,} parameters")
        
            muon_optimizer = torch.optim.Muon(
                muon_params,
                lr=0.02,  # ~30x higher than adamw
                momentum=0.95,  # high
                nesterov=True,  # nesterov momentum
                ns_steps=5 
            )

            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device
            print(f"using fused adamw: {use_fused}")
            adamw_optimizer = torch.optim.AdamW(
                adamw_params,
                lr=learning_rate, #passed-in lr (6e-4)
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=weight_decay,
                fused=use_fused
            )

            return {'muon': muon_optimizer, 'adamw': adamw_optimizer}

        else:
            #Pure Adamw: seperate by dimensionality for weight decay
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # decay matmul participants and embeddings
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] #dont decay layernorms and biases
            
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,}, parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:}, parameters")

            #create adamw and use kernel fusion if available (some pytorch versions dont have it and isnt default)
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device
            print(f"using fused adamw: {use_fused}")
            
            optimizer = torch.optim.AdamW(
                optim_groups, 
                lr=learning_rate, 
                betas=(0.9, 0.95), 
                eps=1e-8, 
                fused=use_fused
            ) #gpt-3 paper optimizations

            return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface for testing"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use a "Conv1D" module, but we are transposing the weights to a vanilla linear 
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

def load_tokens(filename):
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

class DataLoader:
    #overly simple dataloader with distributed loading if needed
    def __init__(self, B, T, process_rank, num_processes, split, seed=42, shuffle=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seed=seed
        self.shuffle=shuffle
        assert split in {'train', 'val'}

        #get shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        if self.shuffle:
            self.rng = np.random.RandomState(seed)
            self.shard_indices = list(range(len(self.shards)))
            self.rng.shuffle(self.shard_indices)
        else:
            self.shard_indices = list(range(len(self.shards)))

        #state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
        self.current_pos = self.B * self.T * self.process_rank

    
    def reset(self):
        self.current_shard = 0
        if self.shuffle:
            self.rng.shuffle(self.shard_indices)
        self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
        self.current_pos = self.B * self.T * self.process_rank

    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos+B*T+1]
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets
        #advance tensor pos
        self.current_pos += B * T * self.num_processes
        #reset if loading the next batch would be out of dataset bounds
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            #reshuffle at epoch completion
            if self.shuffle and self.current_shard == 0:
                self.rng.shuffle(self.shard_indices)
            self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y

def log_model_stats(model, step):
    """Log model parameter statistics"""
    if master_process and step % 1000 == 0:  # Every 1000 steps
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get parameter norms
        param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/param_norm": param_norm,
            "step": step
        })

#----------------------------------------------------------------------------
"""
Run scripts:
    if running a distributed run: torchrun --standalong --nproc_per_node={GPUCOUNT} train.py

    elif running a single gpu: uv run train.py
"""

#support distributed training with DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "need cuda to run ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #logging, checkpointing, etc
else:
    #vanilla single gpu run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True #only gpu is the master
    #device detection -- should be cuda for my case
    device = "cpu" #good default
    if torch.cuda.is_available():
        device = "cuda" #work on my workstation
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # work on my macbook
    print(f"using device:", device)


torch.manual_seed(1337) #reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#gradient accumulation setup
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens to match gpt-3 paper
B = 16   #micro batch size (64 on an 8gpu a100 training run)
T = 1024 #sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure that the total batch size is dividisble by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", shuffle=args.data_fixes)
val_loader = DataLoader(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", shuffle=False) #dont shuffle on validation

torch.set_float32_matmul_precision("high") #TensorFloat32 avaliable on my 5060ti blackwell

#create model
# --- model = GPT.from_pretrained('gpt2') --- load og gpt2 weights but not necessary right now
model = GPT(GPTConfig(vocab_size=50304, use_rope=args.use_rope)) #overriding the vocab size with a easier number for ops
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4 #gpt-3-small max lr
min_lr = max_lr * 0.1
warmup_steps = 715 #match gpt-3 warmup schedule
max_steps = 19073 #match token shards
def get_adamw_lr(it):
    #linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:
        return min_lr
    #in between the warmup and steps end, use cosine decaying down to the min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

if master_process:
    # Determine run name based on configuration
    if args.baseline or args.run_1:
        run_name = "baseline"
    elif args.run_2:
        run_name = "run-2-full"
    else:
        # Custom configuration
        features = []
        if args.use_muon: features.append("muon")
        if args.use_rope: features.append("rope")
        if args.data_fixes: features.append("datafixes")
        run_name = "-".join(features) if features else "baseline"
    
    wandb.init(
        entity="yarbrough-labs",
        project="PicoGPT",
        name=run_name,
        config={
            # Model hyperparameters
            "model_type": "gpt",
            "n_layer": raw_model.config.n_layer,
            "n_head": raw_model.config.n_head,
            "n_embd": raw_model.config.n_embd,
            "vocab_size": raw_model.config.vocab_size,
            "block_size": raw_model.config.block_size,
            
            # Training hyperparameters
            "batch_size": B,
            "sequence_length": T,
            "total_batch_size": total_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "weight_decay": 0.1,
            
            # Optimization config
            "use_muon": args.use_muon,
            "use_rope": args.use_rope,
            "data_fixes": args.data_fixes,
            "optimizer": "muon+adamw" if args.use_muon else "adamw",
            
            # Muon specific (if used)
            "muon_lr": 0.02 if args.use_muon else None,
            "muon_momentum": 0.95 if args.use_muon else None,
            
            # System
            "device": device,
            "ddp": ddp,
            "ddp_world_size": ddp_world_size if ddp else 1,
            "precision": "bfloat16",
        }
    )
    
    # Optionally watch model (can be heavy, comment out if too much)
    # wandb.watch(raw_model, log="all", log_freq=1000)
    
    print(f"\n✓ Weights & Biases initialized: {run_name}")
    print(f"  Project: PicoGPT")
    print(f"  Run URL: {wandb.run.get_url()}\n")


#optimize
if args.use_muon:
    optimizers = raw_model.configure_optimizers(
        weight_decay = 0.1,
        learning_rate=6e-4, #used in adamW component
        device=device,
        use_muon=True
    )
    muon_optimizer = optimizers['muon']
    adamw_optimizer = optimizers['adamw']

    if master_process:
        print("\n=== Using Muon + AdamW Hybrid ===")
        print("Muon: constant LR = 0.02")
        print("AdamW: scheduled LR starting at 6e-4")
        print("=" * 35 + "\n")
else:
    optimizer = raw_model.configure_optimizers(
        weight_decay = 0.1,
        learning_rate=6e-4, 
        device=device,
        use_muon=False
    )

    if master_process:
        print("\n=== Using Pure AdamW ===")

#training loop
for step in range(max_steps):
    t0 = time.time()

    #validation loss split
    if step % 250 == 0: #every 250 steps validate
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision bf16 and tf32 
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            wandb.log({
                "val/loss": val_loss_accum.item(),
                "step": step
            })

    if step % 5000 == 0 or step == max_steps - 1:
        if master_process:
            if args.use_muon:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'muon_optimizer': muon_optimizer.state_dict(),
                    'adamw_optimizer': adamw_optimizer.state_dict(),
                    'step': step,
                    'config': raw_model.config
                }
            else:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': raw_model.config
                }
            torch.save(checkpoint, f"checkpoint_step_{step}.pt")
            wandb.save(f"checkpoint_step_{step}.pt")
            wandb.log({"checkpoint/step": step})

    model.train()

    if args.use_muon:
        muon_optimizer.zero_grad()
        adamw_optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    loss_accum = 0.0

    #gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision bf16 and tf32 
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps #scaling down by the accum steps to recover the normalizer
        loss_accum += loss.detach()
        if ddp: #sync the last step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #average the loss accum in distrbuted processing
    
    #gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #cliping global norm as seen in gpt-3 (kinda hacky)
    
    #determine and set appropriate learning rate for this iter
    if args.use_muon:
        #muon uses a constant LR
        #only need to update adamw lr
        adamw_lr = get_adamw_lr(step)
        for param_group in adamw_optimizer.param_groups:
            param_group['lr'] = adamw_lr
    
        muon_optimizer.step()
        adamw_optimizer.step()
        lr = adamw_lr #logging
    else:
        lr = get_adamw_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 #time difference in mill
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = (tokens_processed) / (dt / 1000) #seconds
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        
        # Log to wandb
        wandb.log({
            "train/loss": loss_accum.item(),
            "train/learning_rate": lr,
            "train/grad_norm": norm,
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time_ms": dt,
            "step": step
        })

        # Log model stats periodically
        log_model_stats(raw_model, step)
        
        # Log Muon LR separately if using Muon
        if args.use_muon:
            wandb.log({
                "train/muon_lr": 0.02,
                "train/adamw_lr": lr,
                "step": step
            })
if ddp: #cleanup
    destroy_process_group()

if master_process:
    wandb.finish()
    print("\n✓ Training complete! Weights & Biases run finished.")