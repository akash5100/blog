---
title: 'Case Study: Transformer-Based architecture development'
tags: deeplearning
comments: true
---

> work in progress. publishing this but will be updated every few days, as I learn new stuffs. learning > blog

I am learning how the transformer-based architectures got evolved from 2017 ("Attention is all you need") till today (May 2024). So, I thought why not write a case-study like [this](https://cs231n.github.io/convolutional-networks/#case-studies) (by Andrej Karpathy).

#### **Timeline**
- [GPT (June 2018)](#gpt-june-2018)
- [BERT (October 2018)](#bert-october-2018)
- [Transformer-XL (January 2019) and XLNet (June 2019)](#transformer-xl-january-2019-and-xlnet-june-2019)
- [GPT-2 (February 2019)](#gpt-2-february-2019)


#### How can unsupervised learning work?
[Why it works? Generalization?](https://www.youtube.com/watch?v=AKMuA_TVz3A), hypothetically...

### GPT (June 2018)

`TODO`
https://openai.com/index/language-unsupervised/

### BERT (October 2018)

Google introduced [BERT](https://arxiv.org/abs/1810.04805), which stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. It is another language model based on Transformer architecture but unlike recent language models BERT Is designed To pretrain deep bidirectional representation from unlabeled text. As a result, the pre-trained BERT model can be fine tuned with just one additional output layer to create state of the art models for a wide range of tasks such as question-answering, system language inference. Without any substantial task-specific architecture modification.

Language model pre-training has been shown to be effective in the recent years. There are two existing strategies for applying pre-trained language representation to down-stream tasks:
1. **feature-based**: Uses task-specific architecture, example adding a layer at the end of the trained model and train them for downstream task.
2. **fine-tuning based**: basically transfer learning? (like used in GPT). The trained weights gets updated as well finetuned on a task specific dataset.

The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations. But the authors of paper argues:

> Current technique limit the potential of pre-trained representations. Especially for fine tuning the issue stems from the unidirectional nature of standard language model like openai's GPT. Which only allows token to attend from past. This restriction hampers performance and task requiring bidirectional context, such as question answering.

**Architecture.** The architecture was based on original Transformer based on Vaswani et al. (2017). BERT-base parameters equals to GPT-base for comparision purposes. But BERT uses bidirection self-attention. While the GPT uses constrained self-attention where every token can only attend to context to its left, BERT can see in both direction. This means BERT uses the Encoder-only part of the Transformer.

**Input/Output representations** BERT can handle `TODO`, `[CLS]`, `[SEP]`, `[MASK]`
zzz.

**PreTraining BERT** Implementing this (encoder-only model) however was challenging. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional would allow each word to indirectly attent itself (they called it "see itself") and with this instead of learning important features, the model would cheat and wont learn anything at all. In order to fix this, BERT is trained using two unsupervised task.

- **TASK #1: Masked Sequence**: they masked some percentange of the input tokens at random, and then predict those tokens. This procedure is called **"masked LM (LML)"**. Introduced back in 1952 (Taylor, Cloze task in the literature).
- **TASK #2: Next Sequence Prediction**: `TODO`



### Transformer-XL (September 2018) and XLNet (June 2019)

**TL;DR:** In the next section, you'll learn that GPT-2 has a summarization problem - it struggles with long-term dependencies, leading to hallucinations like "color/log/hat/car" etc. While I couldn't find a specific paper addressing this issue, it's related to the limited attention scope within the context window. TransformerXL tackles this problem by increasing long-term dependencies without expanding the context window. Additionally, it introduces a new positional encoding technique called **Relative Positional Encoding**, which replaces the absolute positional encoding used in the original Transformer. This approach has been adopted in later models like OpenAI's "*[Fill In the Middle](https://arxiv.org/pdf/2207.14255)*" (2022).

<hr>
<figure>
  <img src="{{site.baseurl}}/assets/Case_Study:_Transformer_based_architecture_development/xl-vanilla.png" alt='xl-vanilla' style="max-width: 100%; height: auto;">
  <figcaption>Illustration of the vanilla transformer with a segment length 4 ~from the Transformer-XL paper</figcaption>
</figure>
<hr>

In the vanilla Transformer, attention doesn't flow between context windows. Consider a context length (sequence length) of 64, where 64 tokens constitute a single segment during training. In this scenario, the Transformer is unable to attend to long-term dependencies, as it's limited by the context length of 64. The next segment will have no information about the previous sentence. As a result, 

The fixed context length means the model cannot capture longer-term dependencies beyond the predefined context length. Moreover, fixed-length segments are created by selecting a consecutive chunk of symbols without respecting sentence or semantic boundaries. This leads to a lack of necessary contextual information, making it difficult for the model to accurately predict the first few symbols. Consequently, optimization is inefficient, and performance suffers. They refer to this issue as **context fragmentation**. 

And during inference, the vanilla model consumes a segment of the same length as in training, but only makes one prediction at the last position and in the next step, the segment is shifted to the right by only one position. We lost the context detail again for the next prediction.

TransformerXL used caching of Key and Value attention computation to speed up inference. I think originally introduced in "*Edouard Grave, et al. 2016 Improving neural language models with a continuous cache*".

<hr>
<figure>
  <img src="{{site.baseurl}}/assets/Case_Study:_Transformer_based_architecture_development/xl-model.png" alt='xl-model' style="max-width: 100%; height: auto;">
  <figcaption>Illustration of the Transformer-XL with a segment length 4 ~from the Transformer-XL paper</figcaption>
</figure>
<hr>

1. **Introducing Segment-Level recurrence with state reuse.**  
  To address the limitation of using fixed-length context, they propose to introduce a recurrence mechanism to the Transformer architecture. During training, the hidden state sequence computed for the previous segment is fixed and cached to reuse as an extended context when the model processes the next new segment is fixed and cached to be reused as an extended context when the model process the next segment. If we assume two consecutive segments of length `L`, then  
    
    `S_t` = `[x_t,1 , ... , x_t,L]` and  
    `S_t+1` = `[x_t+1,1, ... , x_t+1,L]`.  
    
    Say the hidden state/calculated attention "memory" of the first segment is given by:
  `h_t^n` where `n` is the n-th layer hidden state. Then we can use this to calculate the hidden state of next segment.  
    
    `h^~` = `[ SG(h_t^n) ◦ h_t+1^n-1 ]`  
      
    Where `SG` is "Stop gradient", similar to `torch.no_grad`, because we dont want to calculate the gradient for previous context, we store the cache as a constant with no backward graph.  
      
    `◦` indicates concat within the context dimension, say the input is of shape, `B,T,C` and after concatination of two segments, it would be `B,2*T,C`. (This notation is from my old blogs, where B is the batch size, T is the sequence length/context length, C is the embedding size).  
      
    We use `h^~` to calculate `Key` and `Value` representations. Not the `Query` of course.  
      
    `K_t+1^n` = `h^~         * W_k^t`  
    `V_t+1^n` = `h^~         * W_v^t`  
    `Q_t+1^n` = `h_t+1^n-1   * W_q^t`  
    
    Why not cache Q? That's a mystery. or [find out here](https://akash5100.github.io/blog/2024/05/06/Key-Value_caching_for_fast_inference.html).

  
2. **Relative positional encodings.**  
  With the above achievement (reusing the previous context attention) we just created another problem. Notice, the positional encoding of current segment's first token is equal to the previous segment's first token.  
    
    To fix this. its a `todo`  

<hr>
<figure>
  <img src="{{site.baseurl}}/assets/Case_Study:_Transformer_based_architecture_development/xl-incoherence.png" alt='xl-incoherence' style="max-width: 100%; height: auto;">
  <figcaption>makes model confuse? ~from https://vimeo.com/384795188</figcaption>
</figure>
<hr>

### GPT-2 (February 2019)
`TODO-- easy already read the paper`

### Sparse Transformers (April 2019)
The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, is a widely used neural network model for natural language processing tasks. However, its computational complexity and memory requirements grow quadratically with the sequence length, which can become a limitation for long sequences.

- Given sequence of length `n`.
  - number of computations required to process attention weights: `n x n = n^2`
  - Memory required to store the intermediate results and attention weights grows quadratically


### Reformer
### Linformer
### Vision Transformer & Image transformer (Niki paramr)
### 


### RoBERTa (July 2019)
`TODO`

### DistilBERT (March 2020)
`TODO`