from dataclasses import dataclass
import math
import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 #number of tokens 50,000 bpe merges, 256 byte tokens, 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12
    n_embd: int = 768 # embedding dim

    learning_rate = 3e-4


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
        
        #self-attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]  == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
        
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #pos embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
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
        #forward token and pos embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) #pos embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B. T, n_embd)
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

class DataLoader:
    #overly simple dataloader
    def __init__(self, B, T):
        self.B = B
        self.T = T

        #at init load tokens from disk into mem
        with open('../pico-shakespere/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) #left on cpu 
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        self.current_pos = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos+B*T+1]
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets
        #advance tensor pos
        self.current_pos += B * T
        #reset if loading the next batch would be out of dataset bounds
        if self.current_pos + (B * T + 1) > len(self.tokens):
            self.current_pos = 0
        return x, y

#----------------------------------------------------------------------------
import time
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

train_loader = DataLoader(B=16, T=1024) 
torch.set_float32_matmul_precision("high") #TensorFloat32 avaliable on my 5060ti blackwell
#get logits
# --- model = GPT.from_pretrained('gpt2') --- load og gpt2 weights but not necessary right now
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

#optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=GPTConfig.learning_rate)
step_count = 50
torch.cuda.empty_cache() #fix cache issues
for i in range(step_count):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision bf16 and tf32 
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 #time difference in mill
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

#print(f'final loss from {step_count} steps is {loss.item()} ')

import sys; sys.exit(0) # -- debug --

# num_return_sequences = 5
# max_length = 30

# #prefix tokens as example in the dev notebook
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# #generation (slightly inefficent)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     #forward model to get logits
#     with torch.no_grad():
#         logits, _ = model(x)
#         #take the logits at the last position
#         logits = logits[:, -1, :]
#         #get probabilities
#         probs = F.softmax(logits, dim=-1)
#         #do top-k sampling of 50 (example did the hugging face default)
#         #topk_probs becomes (5, 50), topk_indicies is (5, 50), 
#         topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
#         #select a token
#         ix = torch.multinomial(topk_probs, 1)
#         #gather the corresponding indicies
#         xcol = torch.gather(topk_indicies, -1, ix)
#         #append the token to the sequence
#         x = torch.cat((x, xcol), dim=1)

# #print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)