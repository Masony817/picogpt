import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.nn import functional as F  # pyright: ignore[reportMissingImports]

#hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#seed 
torch.manual_seed(1337)

#read the dataset in
with open("input.txt", 'r', encoding = 'utf-8') as f:
    text = f.read()

#create the vocab set and size
chars = sorted(list(set(text)))
vocab_size = len(chars)

#char mapping for ints (tokenization)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder string input for a list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of ints into string

#dataset encoding and splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest as val
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #make a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    #should make more explicit but for this its fine
    out = {}
    model.eval()  # pyright: ignore[reportUndefinedVariable]
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y  = get_batch(split)
            logits, loss  = model(X, Y)  # pyright: ignore[reportUndefinedVariable]
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # pyright: ignore[reportUndefinedVariable]
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    #communication layers

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """simple linear layer followed by a relu nonlinearity"""
    #per token layer

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 4x dimension increase to match attention is all you need paper
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd), #proj layer
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """full transformer block: communication -> computation"""
    
    def __init__(self, n_embd, n_head):
        #n_embd is the embedding dim, n_head: desired head count
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #pre-norm variant (per-token)
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x

#overly simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and target are both (b,t) tensor of ints
        tok_emb = self.token_embedding_table(idx) # (b, t, c )
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (t, c )
        x = tok_emb + pos_emb # (b, t, c )
        x = self.blocks(x) # (b, t, c)
        logits = self.lm_head(x) # (b, t, vocab_size )

        if targets is None:
            loss = None
        else:
            #reshape to fit torch cross entropy dim needs
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is a (b,t) array of indicies in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get the probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append samled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m =  model.to(device)

#pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#train loop
for iter in range(max_iters):

    #every once in a while evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))