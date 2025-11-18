from dataclasses import dataclass
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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

