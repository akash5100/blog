---
title: Transformers
tags: deeplearning
---

# Clean version of Karpathy's notebook
```py
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Dataset
bs = 32
sl = 8
Xtr = data[:0.9*len(data)] # 90% of data

def get_batch(split):
  data = Xtr if split == 'train' else val_data
  # mini batch
  idx = torch.randint(len(data)-sl, size=(bs,))
  x = torch.stack([data[i:i+sl] for i in idx])
  y = torch.stack([data[i+1:i+sl+1] for i in idx])
  return x,y

xb, yb = get_batch('train')
# xb.shape & yb.shape 
# inputs:
torch.Size([4, 8])
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
# targets:
torch.Size([4, 8])
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])

# Bigram????

# -------------------
vocab_sz = len(chars) # it's 64
emb_sz = 32
# -------------------

class Bigram(nn.Module):
  def __init__(self):
    self.embs = nn.Embeddings(vocab_sz, emb_sz) # 64 x 32
    self.l1 = nn.Linear(emb_sz, vocab_sz)       # 32 x 64

  def forward(self, idx, targets=None):
    token_emb = self.embs(idx)    # bs,1,32
    logits = self.h1(token_emb)   # bs,1,32 @ bs,32,64

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape # bs, sl, emb_sz
      targets.shape # 32, 8
      logits = logits.view(B*T,C) # 256, 32
      targets = targets.view(B*T) # 256
      loss = F.cross_entropy(logits, targets)
    return logits, loss
```