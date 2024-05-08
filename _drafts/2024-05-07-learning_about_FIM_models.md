---
title: learning about FIM models
tags: deeplearning
---
```intro```

### Text data 

### what they did for evaluation? AR and Infilling eval

### FIM Training and Inference
  #### Sentinel Tokens
    - Best of N generation goes here?
  #### FIM rate?
  #### Document level FIM
    - PSM mode
  #### SPM mode
  #### Context level FIM
    - Default: Docs -> special tokens -> merge -> chunk
    -        : Docs -> FIM -> special tokens -> merge -> chunk (to fit context length)
      - bad for FIM-- prefix and suffix chunked my have no context at all!
    - Fix
      -      : Docs -> special tokens -> merge -> chunk -> FIM
      - Apply FIM after the chunking step.
      - Split the chunked context into individual documents based on the <EOT> boundary token.
      - Select some documents (with a probability given by the FIM rate) and transform them into FIM examples.
      - Join the FIM examples back together with <EOT> boundary tokens.
      - Trim the resulting context slice to the model's context length.
    -------------------------
    Here are examples of both the issue and the solution:
    Issue: Fragmented FIM data
    
    Initial document: "The quick brown fox jumps over the lazy dog. The sun is shining brightly today."
    Joined with <EOT>: "The quick brown fox jumps over the lazy dog. <EOT> The sun is shining brightly today."
    Chunked to context length (e.g., 20 tokens): "The quick brown fox jumps over <EOT> The sun is shining"
    FIM applied: "The quick brown fox [MASK] over <EOT> The sun is shining" (prefix cut off)
    
    Solution: Applying FIM after chunking
    
    Initial document: "The quick brown fox jumps over the lazy dog. The sun is shining brightly today."
    Chunked to context length (e.g., 20 tokens): "The quick brown fox jumps over the lazy dog. <EOT> The sun"
    Split on <EOT>: ["The quick brown fox jumps over the lazy dog.", "The sun"]
    Apply FIM (with probability): ["The quick brown fox jumps over the [MASK].", "The sun"] (only one document transformed)
    Joined with <EOT>: "The quick brown fox jumps over the [MASK]. <EOT> The sun"
    Trimmed to context length: "The quick brown fox jumps over the [MASK]. <EOT> The sun" (complete FIM example)
    In the solution example, the FIM transformation is applied after chunking, ensuring that the prefix and suffix are not cut off, and the resulting FIM example is complete and meaningful.


### Architecture upgrades
   