from django.shortcuts import render
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 64

n_head = 4

n_layer = 4
#using 20% dropout percentage
dropout = 0.05
# ------------
#using dropout = 20%

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = """<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	XY	R	-20	-20	5	5
E	E	-25
S	XY	R	15	-20	5	5
E	E	-25
S	XY	R	-20	15	5	5
E	E	-25
S	XY	R	15	15	5	5
E	E	-25
S	XY	R	-20	15	40	5
E	E	35
<end>
<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	XY	R	-20	-20	40	40
E	E	-25
S	XY	R	-20	15	40	5
E	E	35
<end>
<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	XY	C	-15	-15	5
E	E	-25
S	XY	C	-15	15	5
E	E	-25
S	XY	C	15	15	5
E	E	-25
S	XY	C	15	-15	5
E	E	-25
S	XY	R	-20	15	40	5
E	E	35
<end>
<begin>
S	XY	C	0	0	40
E	E	5
S	XY	R	-2.5	-2.5	5	5
E	E	-25
S	YZ	R	-12.5	-25	25	2.5
E	E	2.5
S	YZ	R	-12.5	-25	25	2.5
E	E	-2.5
S	YZ	R	-2.5	-25	5	2.5
E	E	12.5
S	YZ	R	-2.5	-25	5	2.5
E	E	-12.5
S	YZ	R	0	-5	25	5
E	E	2.5
S	YZ	R	0	-5	25	5
E	E	-2.5
S	YZ	R	20	0	5	30
E	E	2.5
S	YZ	R	20	0	5	30
E	E	-2.5
S	YZ	R	20	15	5	15
E	E	15
S	YZ	R	20	15	5	15
E	E	-15
<end>
<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	YZ	R	-20	-25	5	25
E	E	20
S	YZ	R	-20	-25	5	25
E	E	-20
S	YZ	R	20	-25	5	60
E	E	20
S	YZ	R	20	-25	5	60
E	E	-20
S	ZX	R	-15	10	5	20
E	C	25
S	ZX	R	10	10	5	20
E	C	25
<end>
<begin>
S	XY	R	-20	-40	40	40
E	E	5
S	YZ	R	-40	-25	5	25
E	E	20
S	YZ	R	-40	-25	5	25
E	E	-20
S	YZ	R	0	-25	5	60
E	E	20
S	YZ	R	0	-25	5	60
E	E	-20
S	ZX	C	0	35	40
E	E	5
S	ZX	C	0	-25	20
E	C	5
S	ZX	C	0	-25	20
E	C	-40
<end>
<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	XY	R	-20	-20	5	5
E	E	-25
S	XY	R	15	-20	5	5
E	E	-25
S	XY	R	-20	15	5	5
E	E	-25
S	XY	R	15	15	5	5
E	E	-25
S	XY	R	-20	15	40	5
E	E	35
<end>
<begin>
S	XY	R	-20	-20	40	40
E	E	5
S	XY	R	-20	-20	40	40
E	E	-25
S	XY	R	-20	15	40	5
E	E	35
<end>
"""

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def index(request):
    generated_output = ""
    train_input = ""
    if request.method == 'POST':
        train_input = request.POST.get('train_input', '')
        model = BigramLanguageModel()
        path1="D:\\Fusion 360 Scripts\\Python Django Practice\\Front-End\\DIFM_AI_TDL_V1"
        model.load_state_dict(torch.load("DIFM_AI_TDL_V1_Model1", weights_only=True, map_location ='cpu'))
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = model.to(device)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_output = decode(m.generate(context, max_new_tokens=2000)[0].tolist())

    return render(request, 'difm_app/index.html', {
        'train_input': train_input,
        'generated_output': generated_output
    })
