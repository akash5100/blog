---
title: Self-Supervised learning and Transfer learning (feat. LM)
categories: deeplearning, transfer learning
---

A language model is a model that is trained to guess the next word in a text (having read the ones before). This kind of task is called self-supervised learning.

> **Jargon: self supervised learning** <br /> Training a model using labels that are embedded in the independent variable, rather than requiring external labels.

Self supervised learning is not usually used for the model that is trained directly, but instead is used for pretraining a model used for transfer learning. Self supervised learning is used to train a base model, and that base model is used to train different model for specific task like text classification![^2]

> Wherever possible, you should aim to start your neural network training with a pre-trained model, and fine tune it. You really don’t want to be starting with random weights, because that’s means that you’re starting with a model that doesn’t know how to do anything at all! With pretraining, you can use 1000x less data than starting from scratch.[^1]

Here is an example, say our goal is to train text classifier on IMDb's reviews, either positive or negative.

we can do,

`Wikipedia's pretrained model -> trained on IMDb classifier`

But we can achieve better resilt by,

`Wikipedia's pretrained model -> Finetune on IMDb corpus -> trained on IMDb classifier`

We can finetune on IMDb corpus by getting all the text file and training the model on that big text chunk. This is known as Universal Language Model Fine-Tuning (**ULMFiT**)[^4]

-----------
<br/> **Pretext & Downstream tasks (in transfer learning)**

The task that pretrained model performs is called pretext task and what the model (that we want) performs after fintuning is called downstream task.

The most important question that needs to be answered in order to use self-supervised learning in computer vision is: *what pretext task should you use?* It turns out that there are many you can choose from.

Choosing a pretext task

The task that the base model is going to perform (pretext task) should be *something that is useful/related to the task* that the final model after finetuning (downstream task) is going to perform. 

The relationship between pretext and downstream tasks is that the pretext task is designed to encourage the network to *learn useful features* for the downstream task, and the downstream task is used to *evaluate the quality* of the learned features.

Take an example: *autoencoder*-- This is a model which can take an input image, converted into a greatly reduced form (using a bottleneck layer), and then convert it back into something as close as possible to the original image.

This model is using *compression* as a pretext task.

However, solving this task requires not just regenerating the original image content, but also regenerating any noise in the original image. Therefore, if your downstream task is something where you want to generate higher quality images, then this would be a poor choice of pretext task.

You should also ensure that the pretext task is something that a human could do. 

For instance, you might use as a pretext task the problem of *generating a future frame of a video*. But if the frame you try to generate is *too far* in the future then it may be part of a completely different scene, such that no model could hope to automatically generate it.

-----------------
<br/> **how would you approach the task of applying a neural network to a language modeling problem?**

The data is *texts*, we want to teach our neural network to predict next word. How can we do that? I think it all depends on what we feed to neural net, the structure of data and labels. We can somehow convert the text to numbers because thats what computer understands, We can then preprocess the data in such a way that the next word is the label. How? Idk. but if we do that we might be able to predict next word. Maybe the output will consist of list of probabilities of words like we did in multi-class prediction (Crossentropy and softmax).

Here is the answer:

1. make list of all possible levels of that categorical variable-- creating `vocab`
2. In the created vocab, replace the words with their index.
3. creating embedding matrix for each item in the vocab.
4. using this embedding matrix as first layer in the NN. (we did the same thing in Recommendation system!) <br/>
    A dedicated embedding matrix can take input to the index of vocab (created in step 2), this is faster and much efficient approach than normal one-hot encoded indexing the vocab.[^3]


![image](https://github.com/akash5100/blog/assets/53405133/be06002d-a485-410c-912b-e3f5c73feb62)

*For now, understanding the architecture of this model doesn't matter, this is just for understanding how we used embeddings! 30 is the vocab size. and we predicted next word after that 3 word sequence.*

The main steps can be named as:
1. Tokenization-- Converting Text into list of words (the vocab)
2. Numericalization-- Converting each word in vocab to number, by replacing them with their indices (simple!)
3. Language Model Data Creation (X&Y)-- Hmmm
4. Language Model Creation-- Training an Architecture like LSTM on DataLoaders we created, yo what? (ignore)

Next, I would be creating notes on Tokenization and Numericalization.


**Sources**

[^1]: [Fastai blog post](https://www.fast.ai/posts/2020-01-13-self_supervised.html)
[^2]: [Jurgen introduced it way before](https://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf)
[^3]: [Fastbook course](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)
[^4]: [ULMFiT](https://arxiv.org/abs/1801.06146)