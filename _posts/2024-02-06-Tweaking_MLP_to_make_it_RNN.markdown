---
title: Modeling a simple RNN and making it stateful, what's LSTM?
categories: deeplearning
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

The input x, to the RNN model now of length 16, instead of 3.

We are trying to predict next word after each word. So we loop through for each word in a sequence and feed that word in the embedding and pass it into the linear layer, while updating the hidden state. and it gives one output, which is of length equal to the length of vocab (later we can softmax it, which gives a single word with highest probability).

But for now, we actually save/append that 30 shaped tensor into an array, and we do that for each word in the sequence. So for sequence length = 16, the `outs` array will be of length 16, and each element is a tensor of size 30. `[16, 30] -- [sl, vocab_sz]`

To summarize, for every word in the sequence of length 16, we made the NN predict the next word (everything is possible because how we created the dls) and we save every next word prediction and stack it. That stack of preds is helpful to calculate loss.

Actually each element of size 64 x 30, with the help of numpy's batch processing, so under the hood, we apply those operations on the complete batch (64 in this case).

So the final output of model will be of shape `[bs x sl x vocab_sz]`.

```py
# understanding stack with dim=1
>>> a = torch.tensor([1,3,5])
>>> b = torch.tensor([2,4,6])
>>> torch.stack([a,b])
tensor([[1, 3, 5],
        [2, 4, 6]])
>>> torch.stack([a,b], dim=1)
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```
Let's say we have 4 data points, and it gave us output of 5 (vowels). 

![image](https://github.com/akash5100/blog/assets/53405133/8f81728c-25f7-4ca1-bb2e-9c140cf32468)

Likewise, we stack all the `64` batch and calculate the loss at one go. Since we stacked on dim=1. Here is the loss function:

```py
def loss_func(preds, targs):
    return F.cross_entropy(preds.view(-1, 5), targs.view(-1))
```

We only have one linear layer between the hidden state and the output activations in our basic RNN, so maybe we'll get better results with more.

*Creating Multilayer RNN*

In, a multilayer RNN, we pass the activations from one RNN into a second RNN.

![image](https://github.com/akash5100/blog/assets/53405133/06184635-b2d0-4d5b-a86e-d969b25ebdb0)

We can save our time and use PyTorch's RNN class, which implements exactly the same.

```py
# Args:
#  |      input_size: The number of expected features in the input `x`
#  |      hidden_size: The number of features in the hidden state `h`
#  |      num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#  |          would mean stacking two RNNs together to form a `stacked RNN`,
#  |          with the second RNN taking in outputs of the first RNN and
#  |          computing the final results. Default: 1
#         batch_first: If ``True``, then the input and output tensors are
#  |          provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#  |          Note that this does not apply to hidden or cell states. See the
#  |          Inputs/Outputs sections below for details.  Default: ``False``

class LMModel5(nn.Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        self.ih = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
        self.ho = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)

    def forward(self, x):
        res, h = self.rnn(self.ih(x), self.h)
        self.h = h.detach()
        return self.ho(res)

    def reset(self): self.h.zero_()

n_hidden = 64
vocab_sz = len(vocab)
n_layers = 2

# And if we train this:
learn = Learner(dls, LMModel5(len(vocab), 64, 2), 
                loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)
```
![image](https://github.com/akash5100/blog/assets/53405133/4c8de6ea-d643-4bd9-927f-918be1d7445d)

It disappointing than our single layer RNN, why so? The reason is that we have a deeper model, leading to exploding or vanishing activations.

*Exploding and disappearing Activations*

In Practice, creating accurate RNN model is difficult. We will get better results if we call `detach` less often and have more layers-- this will give our RNN a longer time to learn from and richer features to create. This means our model is more deep and training this kind of deep model is a key challenge. 

This is challenging because of what happens when you multiply by a matrix many times. If we multiply a number many times, eg, if you keep multiplying 1 with 2, 2, 4, 8, 16 and after 32 steps, you already at 4,294,967,296. A similar issue happens if you multiply by 0.5, you get 0.5, 0.25, 0.125 and after 32 steps, 0.00000000023. As you can see, multiplying a number even slightly higher or lower than `1` results in an explosion or disappearence of our starting number, after just few multiplications.

Because matrix multiplication is just multiplyiong number and then adding them up, exactly same thing happens-- and that's all a deep neural network is-- each extra layer is another matrix multiplication. This means that it is very easy for a deep nn to end up with extremely large or extremly small numbers.

This is a problem, because the way computer stores floating numbers, it become less and less accurate the further away the number gets from zero.

![image](https://github.com/akash5100/blog/assets/53405133/a2a5d149-1726-41da-9aa7-9797021adb5b)

This inaccuracy often leads to the gradient calculated for updating the weights end up as zero or infinity. This is referred to as the *exploding or vanishing gradients* problem. That means, in SGD the weights are either not updated at all or jump to infinity, either way that doesn't improve with training.

One option is to change the definition of a layer in a way that makes it less likely to have exploding activations. (How? idk.)

Another option is by being careful about initialization.

For RNNs, two types of layers are frequently used to avoid exploding activations: gated recurrent units (GRUs) and long short-term memory (LSTM) layers. Both of these are available in PyTorch and are drop-in replacements for the RNN layer.