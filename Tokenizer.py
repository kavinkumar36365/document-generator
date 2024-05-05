import json

class naive_tokenizer:
    def __init__(self):
       self.stoi = {}
       self.itos = {}
       self.vocab_size = 0
    
    def fit(self):
        with open('input.txt','r') as f:
            text = f.read()
        chars =sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
    
    def encode(self, s):
        return [ self.stoi[ch] for ch in s]
    
    def decode(self, l):
        return ''.join([ self.itos[str(i)] for i in l])
       
    def load_json(self):
        with open('stoi.json') as f:
            self.stoi = json.load(f)
        with open('itos.json') as f:
            self.itos = json.load(f)
        with open('vocab_size.json') as f:
            self.vocab_size = json.load(f)
    
    def save_json(self):
        with open('stoi.json','w') as f:
            json.dump(self.stoi,f)
        with open('itos.json','w') as f:
            json.dump(self.itos,f)
        with open('vocab_size.json','w') as f:
            json.dump(self.vocab_size,f)
        

        
        