---
title: Embeddings in Recommendation Systems
tags: deeplearning
---

I will write a follow up blog about Regression and Random Forest, today completes the half of the first month of 2024, I learned lots of stuffs that I am interested and participated on a Kaggle compe, predicing Energy consumption and production using solar panels, just to try what I learned and now I am one step closer to understanding the "Attention". To whoever reads this (other than me) I started writing blogs as creating notes to what I learned, example, I learned about neural net's neurons and wrote a blog on it, talking to myself, but if you read some other blog of mine or reading this, or will read after this, then-- thank you, I will continue this journey and still be writing this blog for future me like creating notes, maybe in more readable (and understandable to others) form.

-------
<br/>

#### **Collaborative Filtering**

A common problem to solve is having number of users and number of products, and you want to recommend which product the user will like or you have a new product in your database and you would like to recommend that product to you current users, but you cant recommend that to every user because someone might like it and some may hate it, you only want the first one to happen. A general solution to this problem called collaborative filtering, works like: look up what kind (genre) a user likes and find other user who have used/liked the same product and recommend the other product that those user have used/like.

For example, Netflix when we create an account asks, what kind of movie/genre you like.

#### **A Tabular Dataset -> Movie Recommendation System?**

MovieLens[^1] dataset consists of:
* 100,000 ratings (1-5) from 943 users on 1682 movies.
* Each user has rated at least 20 movies.
* Simple demographic info for the users (age, gender, occupation, zip)

There is the *User* table[^2] and a Movie (*items*) table called movie.csv. Merging those two we have:

[^1]: [MovieLens dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

[^2]: [My kaggle Notebook](https://www.kaggle.com/code/akzsh5100/collaborative-filtering-movielens?scriptVersionId=156966672&cellId=4)


```py
rating = pd.read_csv(path/'rating.csv').drop(columns=['timestamp'])
movie = pd.read_csv(path/'movie.csv', usecols=(0,1)) # for now using, movie id and title
rating = rating.merge(movie)
rating.head()
```

| userId | itemId | rating |
|----|----|----|
| 196 | 242 | 3 |
| 186 | 302 | 3 |
| 22 | 377 | 1 |
| 244 | 51 | 2 |
| 166 | 346 | 1 |

![image](https://github.com/akash5100/blog/assets/53405133/d02f4206-296c-49c1-b1d1-1937d67b9f70)

This crosstab table shows *ratings* a user gave to a movie. We would like our model to learn to fill the missing ones. Those missing cells are the place where the user have not seen that movie or not reviewed it.

**How?**

If we knew for each user to what degree they liked each category that a movie might fall into, such as genre, age, actors, etc then a simple way to fill these is multiplication of all of them and add them. For instance, lets assume these are ranging from -1 to +1, close to +1 means the user liked it, and vice versa.

![image](https://github.com/akash5100/blog/assets/53405133/29cfeda6-cbf6-49f6-bf5f-2e690c385988)

The result of the multiplication and addition is called *Score*. If we have these numbers for each users and items (movies) we can __dot product__ them and get the scores.

#### **Embeddings?**

From where do we get these numbers for each user and item, the answer is we don't, we learn them. These numbers are called *latent factors*; we refer to them as latent because, for us (humans), we don't know their meaning.

Here is how the overall image will look like:

![image](https://github.com/akash5100/blog/assets/53405133/4da135a6-1d5d-4e96-a6cf-ddf4b82d122b)

We created 2 matrix, depending on how many feature each user and movie will have (here 4, act, com, scifi, rom. So `n_user x 4` and `4 x n_movies`), initialized both with random number because we are going to learn it anyway using SGD.

#### **Speed up the calculation of scores**

We have 2 latent factor matrix, for user and item. To make predictin, we need to take the dot product, for specific user and item's latent vectors. But deep learning models don't know how to index into a matrix to lookup the vector for a specific user/movie. They only understand matrix multiplications. We can represent _looking up the index_ as multiplying by one-hot-encoded vector of that row.

```py
# User Matrix 
users = torch.tensor([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9]])
#(3 users x 3 factors)

# One-hot encoded vector for 2nd user 
one_hot = torch.tensor([0, 1, 0])

# Matrix Multiplication
users * one_hot

# Result gives latent factors for only 2nd user
tensor([[0.0, 0.0, 0.0], 
        [0.4, 0.5, 0.6],
        [0.0, 0.0, 0.0]])
```

But this is inefficient, we can think how big will be the one-hot encoded matrices for users and, so libraries like PyTorch has *Embedding* Layers that do this lookup and retrieve the vector at that index, it creates the latent factor matrices in such a way that later we dont need to multiply one-hot encoded matrix to get latent factor for specific user/item. A simple array indexing gives us the row we want.

#### **Why sharp curves of learning weights = overfitting**

We have non-linearity in labels, our universal function approxiator (neural nets) want to learn that non-linearity, So, what we learn? weights and bias. If the weights are increased more and more over time, what will happen? 

![image](https://github.com/akash5100/blog/assets/53405133/6702cbb2-bb7b-4096-b7f1-d321e0e6c721)

It will be more and more *sharp*, that's nothing but overfitting.

#### **Weight Decay or L2 Regularization**

1. Large weights in a neural network indicate that certain activations are becoming more active compared to others over time. 
2. We calculate loss to determine the disparity between our predictions and the target, and then use it to compute gradients. `preds - targs`
3. We adjust the weights to minimize the loss, with the frequency of adjustments determined by the *learning rate*.
4. Adding a large term to artificially inflate the loss can cause the gradient descent algorithm to decrease the weights more than necessary.
5. This aggressive adjustment of weights occurs because the stochastic gradient descent algorithm perceives a steep gradient in the opposite direction when the loss is artificially increased.
6. However, in practical scenarios, directly augmenting the loss would be inefficient. Instead, a large constant is added directly when calculating the gradient, effectively increasing its magnitude without inflating the loss.

> When the loss is higher, indicating poorer model performance, the gradient will typically have a larger magnitude, suggesting a steeper slope in the loss landscape. Following the direction opposite to the gradient with appropriate step size allows gradient descent to efficiently navigate the landscape and converge to a better set of model parameters, thereby minimizing the loss.

> Whether a weight becomes smaller or larger depends on the sign of the gradient for that weight and the direction opposite to it that is followed during optimization.

We calculate the gradient, the step function makes weights more less that it should be. This is called _Weight Decay_.

In practice, it would be very inefficient, we can do this _increasing of loss_ task directly while calculating gradient.


> d/dp p^2 = 2p^(2-1) = 2p

Under-the-hood, this is the grad calculation, right?:

`params.grad += 2 * params`

We can add a constant `wd` Weight Decay, so big that it will make it twice as big, so we can skip the inefficient __sum of weights whole squared__ part.

`params.grad += 2 * params * wd` <- Weight decay constant

#### **Movie Recommendation System with Embeddings (MovieLens Dataset)**

Follow kaggle notebook for more, but I will just create a Embedding layers and our Recommendation system's model for scratch here:

What the embedding is, guessed?-- random numbers for each users and for each movies. And they are learnable. Which means `return_grad=True`.

```py
def create_embs(size):
    return nn.Parameters(torch.zeros(*size).normal_(0, 0.01))
```

torch.tensor.normal_?[^3]

[^3]: [normal_](https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html)

```py
class DotProductModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors, y_range=(0, 5.5)):
        self.user_factors = create_embs([n_users, n_factors])
        self.item_factors = create_embs([n_items, n_factors])
        self.user_bias = create_embs([n_users])
        self.item_bias = create_embs([n_items])
        self.y_range = y_range

    def forward(self, x): # assuming x is [userid, itemid]
        # we need dot product of user*item matrix
        users = self.user_factors[:, 0] # every row's userid
        items = self.item_factors[:, 1] # every row's itemid
        res = (users*items).sum(dim=1) # keep the dim to 1
        # add bias
        res += self.user_bias[x[:,0]] + self.item_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range) # unpack y_range and pass
```


#### **Direction and Distance of embeddings**

The embedding layers are hard to understand directly, but _Principal Component Analysis (PCA)_ is used to get the underlying directions in matrix.

--------
<br/>

**Cold Start Problem / Bootstrapping?**

The biggest challenge with collaborative filtering in practice is, when you have no data for the user, how would you recommend it? Or when you have a complete new product in you database, whom would you recommend it?

Taking the average just works, taking the average for the sci-fi may high compared to average of action factor, __It would probably be better to pick a particular user to represent average taste lol.__

Or, like what Netflix, amazon prime do for new users? (create form, like what genre you like, what actors you like etc)

#### **Otakus are kinda poison to our embeddings**

Imagine, a small number of users ended up setting up recommendation for the complete database. For instance, people who watch anime, tends to only review anime's and this result in the overall recommendation's direction to incline toward anime's. This result in getting some anime's in _Top ten movies_. 

This is a database problem, this can be solved by hiring a good database maintainer, no?

In the end, this is coming back to _how to avoid disaster when rolling out any kind of machine learning system_. It’s all about ensuring that there are __humans__ in the loop; that there is careful monitoring, and a gradual and thoughtful rollout.

Yo, what we saw above is just a simple dot product, there are 2 ways that I learned:

1. Get the latent factors (embeddings) and predict scores with simple Dot products. when data is simple and big I assume, for more speed and efficiency?
2. Other is to get the latent factors and concat them (movies_embs + user_embs) and pass them to neural net layers. --The Deep Learning Approach

Look below the image of Google play's rec sys.


![image](https://github.com/akash5100/blog/assets/53405133/4ec0baa6-22f1-4fe8-87c7-9f08e4e65ec5)


**Entity Embedding Paper and Google Play's Recommendation system**

Google Play's recommendation system paper[^4]

[^4]: [Wide and Deep Learning for Recommender System](https://arxiv.org/abs/1606.07792)

![image](https://github.com/akash5100/blog/assets/53405133/eab05bc6-9895-46d5-9834-eadf53437db0)

This entity embedding specifically refers to the second part, when we have embeddings for categorical variables, we can concatenate them (the embedding) with RAW categorical data, (either one hot encoded or ordinal) and use that result to feed into Neural network.


> this below is from 'Practical Deep Learning for Coders book' worth sharing

![image](https://github.com/akash5100/blog/assets/53405133/e7bd8588-0954-4649-a7c6-fc6c004c2de6)

**Other thing that I learned, that want to remember is (might be wrong on this)**

Big orgs doesn't create better architectures (their researchers do), How the researchers do? on addition to working on researches, they stay up-to-date with research on that specific domain they are working, for example NLP, Computer Vision etc. When they like some new discoveries, they keep that. On top level, this could be more easily understood if you think of this example: there are startups like Figma which provides better features from the already available products like Adobe, and later Adobe buys it. The same thing happens with architectures - maybe they see a breakthrough research, if open source research, they test it and if they get success they spent their resources in that, who knows.

**Sources**