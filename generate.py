import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel
from Tokenizer import naive_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel()
m = model.to(device)

tokenizer = naive_tokenizer()
tokenizer.load_json()

# load model
m.load_state_dict(torch.load('weights.pth'))

#if trained on gpu and run on cpu use this
#m.load_state_dict(torch.load('weights.pth', map_location = 'cpu'))


context  = torch.zeros((1,1), dtype = torch.long, device = device)
print(tokenizer.decode(m.generate(context, max_new_tokens = 500)[0].tolist())) 

