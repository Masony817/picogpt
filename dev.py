import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.nn import functional as F  # pyright: ignore[reportMissingImports]

#hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
n_embd = 32
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
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y  = get_batch(split)
            logits, loss  = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #idx and target are both (b,t) tensor of ints
        tok_emb = self.token_embedding_table(idx) # (b, t, c ) batch of 4, time of 8 (context size), channel of 65 (vocab size)
        logits = self.lm_head(tok_emb) # (b, t, vocab_size )

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
            # get predictions
            logits, loss = self(idx)
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