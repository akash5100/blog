---
title: Global Attention
tags: deeplearning
---

To understand how global attention works quickly, take a long sentence, tokenize it, and then compare how the original attention mechanism (using Key-Value-Query, or KQV) works on the tokens versus how it operates with global tokens derived from blocks of those tokens. 

### Step 1: Original Sentence and Tokenization

**Sentence**:  
*"The quick brown fox jumps over the lazy dog near the riverbank in the sunny park."*

**Tokenization**:  
- Tokens: `["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "near", "the", "riverbank", "in", "the", "sunny", "park"]`

### Step 2: KQV Attention Mechanism

#### KQV Calculation

1. **Input Tokens**: 
   - `["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "near", "the", "riverbank", "in", "the", "sunny", "park"]`

2. **Key, Query, Value Calculation**:
   - Each token is transformed into keys (K), queries (Q), and values (V) using linear transformations.
   - For simplicity, assume each token has a vector representation (embedding).

3. **Attention Scores**:
   - Compute attention scores for each token against every other token:
   - For example, the attention score for "fox" attending to "jumps" could be calculated as:
    $$
    \text{score}(\text{fox}, \text{jumps}) = Q_{\text{fox}} \cdot K_{\text{jumps}}^T
    $$

4. **Softmax and Attention Weights**:
   - Apply softmax to the scores to get attention weights, which determine how much focus each token gives to others.

5. **Weighted Sum**:
   - The output for each token is a weighted sum of the values:
   $$
   \text{output}_{\text{fox}} = \sum_{j} \text{softmax}(\text{score}(\text{fox}, \text{token}_j)) \cdot V_{\text{token}_j}
   $$

### Visualization of Attention (KQV)

- Each token attends to all other tokens, resulting in a dense attention matrix. For instance, "fox" might attend strongly to "jumps" and "lazy," while attending less to "the" or "riverbank."

### Step 3: Breaking into Blocks and Global Tokens

#### Block Creation

Let’s break the tokens into blocks of size 5:

- **Blocks**:
  - Block 1: `["The", "quick", "brown", "fox", "jumps"]`
  - Block 2: `["over", "the", "lazy", "dog", "near"]`
  - Block 3: `["the", "riverbank", "in", "the", "sunny", "park"]`

#### Global Token Calculation

1. **Global Token for Each Block**:
   - For Block 1: Average the embeddings of `["The", "quick", "brown", "fox", "jumps"]`
   - For Block 2: Average the embeddings of `["over", "the", "lazy", "dog", "near"]`
   - For Block 3: Average the embeddings of `["the", "riverbank", "in", "the", "sunny", "park"]`

### Visualization of Global Attention

Now, each block has a global token that summarizes its context:

- **Global Tokens**:
  - Global Token 1: Represents the context of Block 1 (e.g., action of jumping).
  - Global Token 2: Represents the context of Block 2 (e.g., nearby elements).
  - Global Token 3: Represents the context of Block 3 (e.g., the environment).

### Attention Mechanism with Global Tokens

1. **Attention Scores**:
   - Each token in a block can attend to both its local tokens and the global token of its block.
   - For example, "fox" in Block 1 can attend to "jumps" and the Global Token 1.

2. **Weighted Sum**:
   - The output for "fox" now considers both its local context and the global context:
   $$
   \text{output}_{\text{fox}} = \text{softmax}(\text{score}(\text{fox}, \text{jumps})) \cdot V_{\text{jumps}} + \text{softmax}(\text{score}(\text{fox}, \text{Global Token 1})) \cdot V_{\text{Global Token 1}}
   $$

### Comparison of Attention Mechanisms

- **Original KQV Attention**:
  - Each token attends to every other token, creating a dense attention matrix.
  - This can capture fine-grained relationships but may be computationally expensive for long sequences.

- **Global Token Attention**:
  - Each token attends to local tokens and a global token, reducing the complexity.
  - The global token provides a broader context, allowing the model to understand relationships across blocks without needing to attend to every individual token.

### Conclusion

Global attention captures the attention of tokens in sequence as well as the attention of blocks of tokens. this makes the learning better.


### Source

[LongT5 Paper](https://huggingface.co/papers/2112.07916)