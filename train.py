import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel
from Tokenizer import naive_tokenizer
from model import hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyperparameters
hyp = hyperparameters()
hyp.load_hyperparameters()


# load data
with open('input.txt','r') as f:
    text = f.read()

tokenizer = naive_tokenizer()
tokenizer.fit()
tokenizer.save_json()


# create dataset
data = torch.tensor(tokenizer.encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# create dataloader
def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-hyp.block_size, (hyp.batch_size,))
    x = torch.stack([data[i:i+hyp.block_size] for i in ix])
    y = torch.stack([data[i+1:i+hyp.block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(hyp.eval_iters)
        for k in range(hyp.eval_iters):
            X, Y = get_batch(split)
            logits , loss = model(X,Y)
            losses[k]=loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr =float(hyp.learning_rate))


for iter in range(hyp.max_iters):
    if iter % hyp.eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

    xb , yb = get_batch('train')

    logits , loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save model
torch.save(model.state_dict(), 'weights.pth')