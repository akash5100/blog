---
title: Creating simple RAG
tags: deeplearning
---

> Life update, last week I joined [Plane.so](https://plane.so) as AI engineer let's see if I can make it up to AI researcher!

This is the [paper](https://arxiv.org/abs/2005.11401) which introduced RAG. The problem with LLMs is that they "hallucinate" and have fixed world knowledge, bleh... we already know that RAG architecture tries to address it. My main motive to create RAG is to understand vector embeddings more in-depth and how we create it, search it and what happens next if we find those retieved documents?

## RAG Model Architecture
This paper introduced, this simple architecture.
- Get a encoder model, which encodes anything into vectors
- Use this encoder to encode Documents (z)
- Use the same encoder to encode the query (x)
- find document vectors `z` having similar to query vector `x`.
- concat `x` and `z` vectors and feed them to a generator `G` (generates output)
- In this paper, the author used BERT base as encoder for query and documents
- and used BART as generator.

Starting to understand how this "encoder" works.

### VectorDB
But first, let's assume we don't know how the embeddings are created :), but [we know that embeddings](https://akash5100.github.io/blog/2024/01/15/embeddings.html#embeddings) are high-dimensional representation of words/sentences/images ~ our data. For now let's say we have a model D, which embeds your data into high dimensional vectors. i.e.,:
```py
data -> D -> vectors
# we use this encoder to create embeddings for our documents
# we have docs, z
z  -> D -> Zi # vectors for z
z -> D(z) -> Zi
```
We got a query (x) and we want to our LLM to response something related to documents (z) and give output (y).
```py
# we have query, x
x -> Q -> Xi # Q is the query encoder 
x -> Q(x) -> Xi # 
```

The query encoder Q and the document encoder D are based on same model. Now we have query vector and N document vectors.

### Similarity Search
Given a query vector and N candidate vectors, find the most similar one.

1. Dot product (Inner product)
  The inner product (or dot product) of two vectors measures the magnitude of their overlap. For vectors q (query vector) and v (candidate vector), the inner product is defined as `q.v = sum(qi * vi)` (dot product, at i'th index).
  - if the result is high, it means vectors are similar in terms of their magnitude and direction.
  - The candidate vector with high values are more similar to the query
  - Suitable for scenarios where the magnitude of the vectors is important

2. Euclidian distance
  Learned in school right? The distance between two points, in a geometric plane. It measures the straight-line distance between two points in a multidimensional space.
    ```py
    a = [1,2,3]
    b = [4,5,6]
    x = sum((a[0]-b[0])**2, (a[1]-b[1])**2, (a[2]-b[2])**2)
    ed = x**0.5
    ```

3. Cosine Similarity
  Calculates the cosine of the angle between two vectors, which indicates their directional similarity regardless of their magnitude.
  Given A and B vectors
    ```py
    ma = sum(x**2 for x in A) # magnitude of a
    mb = sum(x**2 for x in B) # magnitude of b
    dp = A.B #dot-product
    cosine_similarity = dp / ma * mb
    ```
  - Cosine similarity is close to 1, vectors are pointing in same direction


- Dot product gives a magnitude (MAGNITUDE)
- Euclidian distance gives a straight-line difference (DISTANCE)
- cosine similarity purely focuses on the angle between them and thus their directional similarity (DIRECTION)

So for a RAG on wiki pages, what should I use?
- In a high-dimentional spaces, magnitude can vary
- I can use cosine similarity, because it
  1. It measures the direction and orientation
  2. CS normalizes the vectors too!
- ED is good but it doesn't normalizes vectors, in a high-dimensional embedding spaces, bleh

### Some details on encoder according to the paper
In this paper, the author used BERT base as an encoder and maximum inner product search (MIPS) to search for similar vectors, because it is sub-linear. (other is MCSS - Maximum cosine similarity search)

After retrieving content when generating the BART model: we combine input vector X with Z by simply concatenating them. 

> This is why they choose BART: BART was pre-trained using a denoising objective and a variety of different noising functions. It has achieved state-of-the-art (SOTA) results on a diverse set of generation tasks and outperforms comparable-sized T5 models.  
> This again raises question: If BERT outperformed GPT-2 of same parameter size and BART outperformed BERT, T5 and GPT-2 of same size, can a encoder-decoder model similar to BART of GPT-4 parameter size will outperform GPT-4?

**Training**  
Updating the document encoder D while training showed no significant improvement + its was slow, so they (authors) decided to freeze the document encoder and only finetuned the query encoder Q and the generator G, which is the BART model.
### Final small detail before we create our own embedder:
They introduced 2 architectures:
- RAG sequence
- RAG token

#### 1. RAG-Sequence
The difference is simple, the sequence model retrieves bunch of docs and generate output from generation individually from the generator G. Say we retrieved n documents, then we will generate n outputs, and finally we reconsider all the n outputs to generate a single output. For this reconsidering all outputs from the previous generations, we use BEAM SEARCH. damn, I need to study what is beam seach (self remainder).

#### 2. RAG-Token
This one is simple, after we retrieved `n` document we use every documents to generate a single output and each token generated can be based on any of the `n` document.

But [this](https://arxiv.org/pdf/2310.03214) architecture is used now a day (i think) because perplexity uses it :)

### Back to encoder
Before this paper, another paper tried to create RAG, they used something called Dense Passage Retrieval (DRP). And DRP is based on BERT. So, they somehow used BERT (LLM) to create encoder which can embed data? what?

I found out that there is a leaderboard for data embedders called MTEB (Massive Text Embedding Benchmark), the models in this leaderboard are named as Qwen2-instruct, Mistral, Meta-llama, these are LLMs, so we can use LLMs to create embeddings, or rather (just assumtions ahead) instead of predicting the next tokens we can somehow use the hidden layers of the LLMs as Embeddings? or maybe make them give the embeddings in an unsupervised way?

It turns out, for creating embeddings from scratch: the skip-gram model (and word2vec in general) is one way to create word embeddings from scratch. This process is often called "training word embeddings" or "learning word representations." Its unsupervised training as expected.
  
**Bag-of-words Model**  
In bag of words, as we know given a context predict the target. BoW dont consider the context in which the words appear, it just considers the frequency.

**Skip-Gram model**  
Skip gram is opposite of BoW, given a target predict the context, example for a given sentence: "Hello how are you bob?", if the target is "bob" we might have dataset like (bob, hello), (bob, you) etc.

Word2Vec uses this two approaches for generating word embeddings. Its word level, for our use case, which is creating embeddings for documents like articles/blog, we might need to go beyond word-level, something like sentence, passage or even document-level embeddings.

Modern techniques do use pre-trained LLMs such as BERT and fine-tune them for generating embeddings or they use average of hidden states to create embeddings.

For our problem, I found this paper [S-BERT / Sentence BERT](https://arxiv.org/pdf/1908.10084), sounds promising for our understanding, because it used BERT to create encoder.

### Sentence-BERT
> todo

## Roadmap to MinRAG:
From what I understand, I need a document, query encoder and a generator
- Use S-BERT (small) as encoder (probably)
- Use GPT-2 (small) as generator
- setup encoder
  - some how encode few wiki pages and store it?
  - encode query
  - use MCSS and get top-k vectors
- concat and generate
