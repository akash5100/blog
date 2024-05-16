---
title: 'Gradient-Checkpointing, a hack for training deep NN'
tags: deeplearning
---

You got a very deep neural network to train, lets say wide 128-layers. Does it fits in memory? Yes (barely). The activations and gradients in forward and backward pass respectively takes a lot of memory. But you want to train more deep NN. Why? because, we build the compute (stack more layer) ~= win. ([scaling hypothesis](https://gwern.net/scaling-hypothesis#scaling-hypothesis)) / Blessing of scaling.

See the original gradient checkpointing implementation.[^1]

Here is a visualization of vanilla training:

<hr>
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Deep_network_Training_Hack_Gradient_Checkpointing/output.gif" alt='vanilla training' style="max-width: 100%; height: auto; align: center">
  <figcaption>top left is "input", and the bottom right most is the "loss". Upper layer is the forward pass i.e, attentions and lower layer is the backward pass, i.e, gradients  
  The purple shaded circles indicate which of the nodes need to be held in memory at any given time.
</figcaption>
</figure>
<hr>

However, if the cost of computation < cost of memory, we have limited memory and we are willing to recalculate those nodes (attentions and gradients) then we can save a lot of memory. We can simply recompute them when we need in the backward pass. We can always recompute the activations by running the same input data through a forward pass. See below:

<hr>
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Deep_network_Training_Hack_Gradient_Checkpointing/output_poor.gif" alt='recompute forward pass for each backward pass' style="max-width: 100%; height: auto; align: center">
  <figcaption>We recompute the activation when we need during the backward pass.</figcaption>
</figure>
<hr>

There is a lot of compute waste. For `N-layers` there would be `N` extra forward pass. Gradient checkpointing[^1] is something in between this two methods, where we recompute the forward pass but not too often.

There are checkpoints nodes in the memory during the forward pass, while the remaining nodes are recomputed at most once. After being recomputed, the non-checkpoint nodes are kept in memory until they are no longer required[^1].

<hr>
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Deep_network_Training_Hack_Gradient_Checkpointing/output2.gif" alt='gradient checkpointing' style="max-width: 100%; height: auto; align: center">
  <figcaption>In the first forward pass, we set checkpoints. In the backward pass we recalculate from the last checkpoint. This method trades off compute and memory </figcaption>
</figure>
<hr>

**Optimal checkpointing selection**  
Marking every \\(\sqrt{n}\\) -th node as a checkpoint optimizes memory usage, scaling with the square root of the number of layers.

### Sources
[^1]: [Gradient Checkpointing](https://github.com/cybertronai/gradient-checkpointing)