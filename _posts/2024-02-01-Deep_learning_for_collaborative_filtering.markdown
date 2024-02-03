---
title: Using an Embedding in Neural Network
categories: deeplearning
---

Alright who cares about the follow up blog, [here](https://akzsh.notion.site/Tabular-Data-analysis-and-Decision-Tree-9444c1ca59d7464dbc91e7cb6cb243fc?pvs=4) is a notion notes to save me. This is the notes for tweaking Random forest on a tabular dataset to squeeze some performance, I tried this on a [Kaggle compe](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/leaderboard) and it worked out pretty badly. Score of something like 2.xx. Maybe I Will improve it when time comes. zzz

--------------------------------
<br/>

As we already initialized the embeddings of our users and products for example, We take the result of the embedding lookup and concatenate those activations together. This gives us a matrix that we can then pass through linear layers and nonlinearities in the usual way.

Since we'll be concatenating the embeddings rather than the dot product, the two embedding matrices can have different sizes i.e, different numbers of latent factors.[^1][^2]

```py
class CollabNN(nn.Module):
    def __init__(self, user_sz, item_sz, n_acts=100, y_range=(0, 5.5)):
        super(CollabNN, self).__init__()
        self.user_factors = nn.Embedding(*user_sz)
        self.item_factors = nn.Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_acts),
            nn.ReLU(),
            nn.Linear(n_acts, 1))
        self.y_range = y_range
        
    def forward(self, userid, itemid):
        embs = self.user_factors(userid), self.item_factors(itemid)
        x = self.layers(torch.cat(embs))
        return sigmoid_range(x, *self.y_range)


model = CollabNN((n_users+1, 74), (n_movies+1, 101), 50)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
loss_func = nn.MSELoss()
```
[^1]: [Kaggle Notebook](https://www.kaggle.com/code/akzsh5100/collaborative-filtering-movielens)
[^2]: [Fastbook](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb)

-----------------------------
<br/>
What do we want to do in a single epoch

```py
# one epoch

for i in dls:
    user, movie, r = i
    r = r.unsqueeze(dim=0)
    optimizer.zero_grad()
    out = model(user, movie)
    loss = loss_func(out, r)
    loss.backward() # calc grad
    optimizer.step() # update weights 
    break
```
-----------------------------
<br/>

**Understanding the above architecture.**

Let's say we decided the size of embedding matrix of users is `n_users x 74` and the embeddings size of items is `n_items x 101` (choosing 74 and 101 randomly, no special reason).

We initialized `user_factors` and `item_factors`. (The Embeddings)

Next, to input the latent factors of each user+item into a linear layer, we created the input layer of size `74+101` and `n_acts` as output. The activation for this layer is ReLU!

After that, there is another Linear layer in sequence, with `n_acts` as input and `1` as output.

This completes our model.

**Working**

In forward pass, we have `userid` and `movieid`. We get the embedding (latent factor) for that particular movie and user. So, if you assumed it would be of shape, `1 x 74` and `1 x 101`, I think you are right!

We collect those two and pass it to the Sequential layer, finally we can *sigmoid* (0 - 1) or *tanh*(-1 - 1) the logits to get activations from the final layer (as it is a score ranging between 0-1, closer to 0 means eww movie and closer to 1 means wowii movie).

I tried to create a simple diagram of the above working:

![image](https://github.com/akash5100/blog/assets/53405133/66369779-ac92-415a-8d25-54f4869ef50a)

# Sources
