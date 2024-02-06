---
title: Modeling a simple RNN and making it stateful, what's LSTM?
categories: 
---

If we have data like:

`X` -- `Y`

`'one', '.', 'two'` -- `'.'`

`'.', 'three', '.'` -- `'four'`

where 3 words are used as input to predict 1 word from a vocabulary as an output, we can create a neural network architecture that takes three words as input and returns a prediction of the probability for each possible next word in the vocabulary.

We will use *three standard linear layers* but with *two tweaks*!

1. The first tweak is that the first linear layer will use only the first word's embedding as activations, the second layer will use the second word's embedding plus the first layer's output activation and the third layer will use the third word's embedding plus the second layer's output activations. <br>
    The key effect is that the model will take into account the information from the words that came before it. <br>
    The 'information' is actually the higher dimensionality of data points. This is created by the embedding layer. [Here](https://www.youtube.com/watch?v=wvsE8jm1GzE) is a video that explains it well in the first part.

2. The second tweak is that each of these three layers will use the same weight matrix. The goal of this specific structure is to train the model's embeddings in a way that a single set of parameters can be used consistently across all positions in the sequence (for example, embedding and learned weights for a word like "the" can be used again and again and the result doesnot depend on position of that word in the sequence). This means, the output of the network is designed not to depend on the specific position of a word in the sequence. <br>
    In other words, activation values will change as data moves through the layers, but the layer weights themselves will not change from layer to layer.

Let's take the same example from above:

![image](https://github.com/akash5100/blog/assets/53405133/7a0dd00d-64db-4b3d-afe6-bb89f4ec0fc0)
*`emb` means the embedding layer, so a word like 'one' is fed into the embedding layer.*

We can now create the language model module that.

```py
class LMModel1(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        self.ih = nn.Embedding(vocab_sz, n_hidden)
    # we need embs for every word in the vocab and n_hidden is a random num
        self.hh = nn.Linear(n_hidden, n_hidden)
        self.ho = nn.Linear(n_hidden, vocab_sz)

    def forward(self, x): # x is batch of length 64, with 3 words each
        h = F.relu(self.hh(self.ih(x[:,0])))
        h = h + self.ih(x[:,1])
        h = F.relu(self.hh(h))
        h = h + self.ih(x[:,2])
        h = F.relu(self.hh(h))
        return self.ho(h)
```
<!-- TODO: Attach an embedding of Kaggle notebook -->

`ih (input to hidden) -> hh (hidden to hidden) -> ho (hidden to output)`


*We can remove the hardcoded part*

```py
class LMModel2(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        self.ih = nn.Embedding(vocab_sz, n_hidden)
        self.hh = nn.Linear(n_hidden, n_hidden)
        self.ho = nn.Linear(n_hidden, vocab_sz)

    def forward(self, x):
        h = 0
        for i in range(3):
            h = h + (self.ih(x[:,i]))
            h = F.relu(self.hh(h))
        return self.ho(h)
```
<!-- TODO: Attach an embedding of Kaggle notebook -->

*we see a set of activations is being updated each time through the loop stored in the variable h-- this is called the **hidden state***

> Jargon: Hidden State <br> The activations that are updated at each step of a recurrent neural network.

> Jargon: Recurrent NN (Looping NN) <br> A neural network that is defined using a loop like this is called a recurrent Neural Network. <br> RNN is not a complicated new architecture but simply a refactoring of a multilayer neural network using a for loop.

*Improving our RNN*

Looking at the code, one thing seems problematic is that we are initializing our hidden state to zero for every new input sequence. Why is that problematic? 

Resetting the activations to zero for every sequence means starting with a "blank slate" for each new input sequence. It doesn't allow the model to remember information from previous sequences it has processed. Imagine you reading a story in a book, but when you turn a page you forget what you read, lol.

Another thing that can be improved in our RNN is, why only predict the 4th word after 3 words? why not predict the 2nd and 3rd words?

*First thing first, let's solve the resetting of hidden state*

We are basically throwing away the information we have about the sentences we have seen so far, this could be easily fixed by saving the state as a class variable.

`self.h = 0`, and update it each time.

But we will be creating a not very noticable, but important to deal problem. Any guesses?

So, for a word, we created embeddings and activations is stored in hidden state. For a new word, we are actually incorporating the activation of that word into the already existing hidden state. This accumulation occurs over time, and if there are 1000 tokens, the size will grow to incorporate information from 1000 tokens. But in real world, we might have 1 million tokens.

This is going to be slow and infact we wont be able to store even one mini-batch on the GPU.

The solution to this problem is to tell Pytorch that we dont want to backpropagate the derivatives through the entire implicit neural network. Instead, we will keep just last three layers of gradients. 

To remove the gradient history in Pytorch, we use `detach` method.

Here is the RNN, now stateful.
```py
class LMModel3(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        self.ih = nn.Embedding(vocab_sz, n_hidden)
        self.hh = nn.Linear(n_hidden, n_hidden)
        self.ho = nn.Linear(n_hidden, vocab_sz)
        self.h = 0

    def forward(self, x):
        for i in range(3):
            self.h = self.h + (self.ih(x[:,i]))
            self.h = F.relu(self.hh(self.h))
        out = self.ho(self.h)
        self.h = self.h.detach()
        return out
    
    def reset(self): self.h = 0 
    # Later, at the beginning of each epoch and before each validation phase 
    # this will be used. This will make sure we start with a clean state 
    # before reading those continuous chunks of text.

n_hidden = 64
vocab_sz = len(vocab)
```

This model will have the same activatiosn whatever sequence length we pick, because the hidden state will remember the last activation from the previous batch. The only thing different is the gradients will be computed at each step will consider only the sequence length instead of the whole stream. This approach is called **batchpropagation through time (BPTT)**.

Here is the gradual flow of data `64 x 1 -> 64 -- 64 -> 64 -- 30 (vocab sz)`

But this would be the shape of hidden state: `64 x 64`. This could be explained as, the output of first layer is 64, and for each element in the batch, the `bs=64`. `(bs, n_hidden)`.

> The gradients for the hidden state (self.h) will be detached after processing each sequence. This means that gradients will not be calculated or updated for the hidden state during backpropagation through time (BPTT) for subsequent sequences. However, gradients for other parameters in the model, such as the parameters of the embedding layer (self.ih) or the linear layers (self.hh and self.ho), will still be calculated and updated normally based on the loss computed at each time step.


We can use our previous [group_chunks](https://akash5100.github.io/blog/deeplearning/2024/02/05/Tokenization_for_LM.html) function to create sequence.

But wait, we can improve the our model, instead of predicting 1 word after every 3 word, what if we can predict next word after every single word?

This is simple to add in! We first need to change our data, so that the dependent variable has each of the three next words after each of our three input words.

```py
sl = 16
seqs = [ tensor(nums[i: i+sl]), tensor(nums[i+1: i+sl+1])
    for i in range(0, len(nums)-sl-1, sl)] # this range will jump 16 index

# lets see
[" ".join([vocab[o] for o in s]) for s in seqs[0]]

['one . two . three . four . five . six . seven . eight .',
 '. two . three . four . five . six . seven . eight . nine']

cut = int(len(seq) * 0.8)
train_ds = group_chunks(seqs[:cut], bs)
valid_ds = group_chunks(seqs[cut:], bs)
```

```py

class LMModel4(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        self.ih = nn.Embedding(vocab_sz, n_hidden)
        self.hh = nn.Linear(n_hidden, n_hidden)
        self.ho = nn.Linear(n_hidden, vocab_sz)
        self.h = 0

    def forward(self, x):
        outs = []
        for i in range(sl):
            self.h = self.h + (self.ih(x[:,i]))
            self.h = F.relu(self.hh(self.h))
            outs.append(self.ho(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1)

    def reset(self): self.h = 0

n_hidden = 64
vocab_sz = len(vocab)
```

the input x, to the RNN model now of length 16, instead of 3.

we are trying to predict next word after each word.

so we loop through for each word in a sequence and feed that word in the emb and pass it into the linear layer, while updating the hidden state. and it gives one output, which is of length equal to the length of vocab (later we can softmax it, which tells us a single word with highest probability).

But for now, we actually save/append that 30 shaped tensor into an array, and we do that for each word in the sequence.

so for sequence length = 16, the `outs` array will be of length 16, and each element is a tensor of size 30.

To summarize, for every word in the sequence of length 16, we made the NN predict the next word (everything is possible because how we created the dls) and we save every next word prediction and stack it. That stack of preds is helpful to calculate loss.

Actually each element of of size 64 x 30, with the help of numpy's batch processing, so under the hood, we apply those operations on the complete batch (64 in this case).

# Upcoming topics

- Multilayer RNN
- Exploding and disappearing Activations