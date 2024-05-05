import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
from Tokenizer import naive_tokenizer

tokenizer = naive_tokenizer()
tokenizer.fit()

#hyperparameters for the model to be imported from hyperparameter yaml file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class hyperparameters:
    def __init__(self):
        self.batch_size = 64
        self.block_size = 256
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 3e-4
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout =  0.2
    
    def load_hyperparameters(self):
        with open('hyper parameters.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.batch_size = data['batch_size']
        self.block_size = data['block_size']
        self.max_iters = data['max_iters']
        self.eval_interval = data['eval_interval']
        self.learning_rate = data['learning_rate']
        self.eval_iters = data['eval_iters']
        self.n_embd = data['n_embd']
        self.n_head = data['n_head']
        self.n_layer = data['n_layer']
        self.dropout = data['dropout']

class Head(nn.Module,hyperparameters):
    #one head of self attention

    def __init__(self, head_size):
        super().__init__()
        hyperparameters.load_hyperparameters(self)
        self.key = nn.Linear(self.n_embd, head_size, bias = False)
        self.query = nn.Linear(self.n_embd, head_size, bias = False)
        self.value = nn.Linear(self.n_embd, head_size, bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(self.block_size,self.block_size)))
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  #B x T x Head_size
        q = self.query(x) #B x T x Head_size
    

        wei = q @ k.transpose(-1,-2) * C**(-0.5) #B,T,C @ B,C,T = B,T,T
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #B,T,T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) #B,T,C
        out = wei @ v #B,T,T @ B,T,C = B,T,C

        return out
    
class MUltiHeadAttention(nn.Module,hyperparameters):
    '''multiple heads of self attention in parallel'''

    def __init__(self, n_heads, head_size):
        super().__init__()
        hyperparameters.load_hyperparameters(self)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #B,T,C*n_heads
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module,hyperparameters):

    def __init__(self,n_embd):
        super().__init__()
        hyperparameters.load_hyperparameters(self)
        self.net = nn.Sequential(nn.Linear(n_embd,4*n_embd),
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd,n_embd) ,
                                 nn.Dropout(self.dropout),
                                 )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module,hyperparameters):
    '''transformer block : communication followed by computation'''

    def __init__(self,n_embd, n_head):
        super().__init__()
        hyperparameters.load_hyperparameters(self)
        head_size = n_embd//n_head
        self.sa_heads = MUltiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)  
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# create  bigram model
class BigramLanguageModel(nn.Module,hyperparameters):

    def __init__(self):
        super().__init__()
        hyperparameters.load_hyperparameters(self)

        self.token_embedding_table = nn.Embedding(tokenizer.vocab_size,self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size,self.n_embd)
        
        self.blocks = nn.Sequential(
            *[Block(self.n_embd, n_head = self.n_head) for _ in range(self.n_layer)]
        )
        self.ln_f = nn.LayerNorm(self.n_embd) #final layer norm
        self.lm_head  = nn.Linear(self.n_embd, tokenizer.vocab_size) 
    
    def forward(self, idx, targets = None):
        # idx and targets are of shape B, T
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #B,  T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else:

            B , T , C = logits.shape
            
            logits = logits.view(B*T,C)
            
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.block_size:] #B, T

            logits , loss = self(idx_cond) #B, T, vocab_size

            logits = logits[:, -1, :] #B, vocab_size
            probs = F.softmax(logits, dim = -1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)
        
        return idx
    
