---
title: Convolutional Neural Networks (CNNs / ConvNets)
tags: deeplearning
---

Convolutional Neural Networks are very similar to ordinary neural networks. They are made up of neurons that have learnable weights and bias. Each neuron receives some input, performs a dot product and optionally follow it with a nonlinearity. The whole network still expresses a single differentiable score function from the raw image pixel on one end to class scores at the other and they still have a lost function example softmax. On the last fully connected layer and all the tips/tricks that have been developed for learning regular neural networks still applies.

So what changes? ConvNet architecture make the explicit assumption that the inputs are image which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameter in the network.

### Architecture Overview

**Recall**: Regular neural nets. Neural nets takes an input (a single vector) and transform it through a series of hidden layers. Each neuron is fully connected to all neurons in the previous layer. Where neurons in a single layer function completely independently. Do not share any connection. At the end it has an output layer which represents the class scores.

**Regular Neural Nets don’t scale well to full images**. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32x32x3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. 200x200x3, would lead to neurons that have 200x200x3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

**3D volumes of neurons**. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. (Note that the word depth here refers to the third dimension of an activation volume, not to the depth of a full Neural Network, which can refer to the total number of layers in a network.) For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (~RGB) (width, height, depth respectively). The neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension. Here is a visualization:

<hr>
<br>
<figure>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/neural_net2.jpeg" alt='neural net' height=200px>
  <hr>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn.jpeg" alt='cnn' height=200px>
  <figcaption>Top: A regular 3-layer Neural Network. Bottom: A ConvNet arranges its neurons in three dimensions (height, width, depth) as visualized in one of the layers. Every layer of ConvNet transforms the 3D output volume of neuron activations. In this example the red block is the image, so its dimensions would be, height (28), width (28) and depth (3, because 3 channels ~RGB) </figcaption>
</figure>
<hr>
<br>

> ConvNets are made up of layers. Every layer has a simple API: it Transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not has parameters.

### Layers used to build ConvNets

In ConvNet architecture, three types of layers are used to construct the network: **Convolutional layer**, **Pooling layer** and **Fully connected layer** (exactly as seen in regular Neural Networks). These layers are arranged in a coordinated manner to form the architecture of the ConvNet.

a simple ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

- The **input** image is of shape [32x32x3], 32 height, 32 width and 3 color channels.
- **CONV** layer transforms a 2D input into 3D activations, typically reducing the spatial dimensions, but it can also, if desired, increase the spatial dimensions. For instance, it may produce activations like [32x32x10] if 10 filters are used.
- **Non-linearity** layer will apply an elementwise activation function, such as the ReLU (**max 0,x**). This leaves the size of activations unchanged ([32x32x10]).
- **POOL** layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x10]
- **FC** (i.e, fully connected) layer will compute the class scores, resulting in the volume size [1x1x10], where each of the 10 numbers corresponds to a class score, such as among the 10 categories of CIFAR-10. This is similar to ordinary neural net.

This way ConvNets transform the original image layer by layer from the original pixel values to the final class scores. The CONV/FC layers perform transformation that are function of not only the activation in the input, but also the parameter (the weight and bias of the neurons). On the other hand, RELU/POOL layer will implement a fixed function. The parameter in the CONV/FC will be trained with gradient descent.

<hr>

<figure style="padding: 12px;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/convnet.jpeg" alt='convnet'>
  <figcaption>The activations of an example ConvNet architecture. The initial volume stores the raw image pixels (left) and the last volume stores the class scores (right). The middle shows visulizations of activations, since it is difficult to visualize 3D, here the representation is in 2D.</figcaption>
</figure>

<hr>
<br>
*Details on each of the layer*:

### Convolutional Layer

CONV Layer is the core building block of CNNs that does the most computational heavy lifting.

**Overview and intuition without brain stuff**. The CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5px width and height, and 3 because images have depth 3, the color channels). During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.

**The brain view**. As a brain/neuron analogy, every entry in the 3D output volume can also be interpreted as an output of a neuron that looks at only a small region in the input and shares parameters with all neurons to the left and right spatially (since these numbers all result from applying the same filter).

*Details of neuron connectivities, their arrangement in space, and their parameter sharing scheme*:

**Local Connectivity**. When dealing with high-dimensional inputs such as images, as we discussed above it is impractical to connect neurons to all neurons in the previous volume. Instead, we connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently this is the filter size). *The extent of the connectivity along the depth axis is always equal to the depth of the input volume.* It's worth noting once more that there's a difference in how we handle the width, height, and depth dimensions: The connections are limited to nearby areas in 2D space (width and height), but they cover the entire depth of the input volume.

*Example 1*. For example, let's consider a scenario where the input volume has dimensions [32x32x3] (W, H, D). If we have a receptive field (filter size) of 5x5, then each neuron in the convolutional layer would have weights associated with a [5x5x3] region in the input volume *for a total of 5*5*3 = 75 weights (and +1 bias parameter)*. Here, the "3" corresponds to the D of the input volume, representing the three color channels.

*Example 2*. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of [3x3x20] 3x3x20 = 180 connections to the input volume. Notice that, again, the connections are limited to nearby areas in 2D space (e.g. 3x3), but they full cover the entire depth of the input volume (20).

<hr>

<figure style="padding: 12px;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/depth.png" alt='depth'>
  <figcaption>Notice, INPUT: 32x32x<b>3</b> -> FILTER: 6 x 5x5x<b>3</b> -> ACTIVATION MAP: 28x28x<b>6</b> -> FILTER: 10 x 5x5x<b>6</b> -> ACTIVATION MAP: 24x24x<b>10</b> </figcaption>

</figure>

<hr>
<br>

**Spatial arrangement**. How many neurons are there in the output volume or how they are arranged? Three hyperparameters control the size of the output volume: the **depth**, **stride** and **zero-padding**. We discuss these next:

1. First, the **depth** of the output volume is a hyperparameter. It corresponds to the number of filters you would like to use to process the image. Each filter learns something different. For instance, one filter might detect vertical edges, while another might detect blobs of color. The depth dimension represents these different filters, each learning to detect a different feature. We can think of a group of neurons, all focusing on the same area of the input (some people also prefer the term fibre).

2. Second, we must specify the **stride** with which we slide the filter. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.

3. As we will soon see, sometimes it will be convenient to pad the input volume with zeros around the border. The size of this **zero-padding** is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).


We can compute the spatial size of the output volume as a function of the input volume size (**W**), the receptive field size of the Conv Layer neurons (**F**), the stride with which they are applied (**S**), and the amount of zero padding used (**P**) on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by 
<div>
\[ (W−F+2P)/S+1 \]
</div>

For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.

*Use of zero-padding.*
Let's say in the above example, the image size is 32x32 if we pad the image with 2 pixels at the end that would make it, 36x36. Now if the apply the same filter (5x5) the output would be:

<div>
\[ (W - F + 2P)/S + 1 = 36 - 5 + 1 = 32 \]
</div>

It preserved the input shape, when the stride is **S = 1** ensures that the input volume and output volume will have the same size spatially. It is very common to use zero-padding in this way.

*Constraints on strides.*
Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size **W = 10**, no zero-padding is used **P = 0**, and the filter size is **F = 3**, then it would be impossible to use stride **S = 2**, since

<div>
\[ (W - F + 2P)/S + 1 = 36 - 5 + 1 = 32 \]
</div>

i.e. not an integer, indicating that the neurons don’t "fit" neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library could throw an exception or zero-pad the rest to make it fit, or crop the input to make it fit, or something. As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions "work out" can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.

*Real world example*
`TODO`

**Parameter Sharing**
`TODO`

**Backpropogation**
`TODO`

**Dilated convolutions**
`TODO`

> Topic for later, **Causal Convs** and **Causal Dilated Convs** (When generating sequence with RNN and LSTM!)

### Pooling

`TODO`

### Normalization Layer
Many types of normalization layers have been proposed for use in ConvNet architectures, sometimes with the intentions of implementing inhibition schemes observed in the biological brain. However, these layers have since fallen out of favor because in practice their contribution has been shown to be minimal, if any. For various types of normalizations, see the discussion in Alex Krizhevsky’s (cuda-convnet library API)[https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki#Local_response_normalization_layer_(same_map)].

### Fully-Connected (FC) Layer

### Converting FC layers to Conv layers

### ConvNet Architecture

### Case studies

#### LeNet
#### AlexNet
#### ZFNet
#### VGGNet
#### GoogLeNet
#### ResNet

#### Computational Considerations
`TODO`

### R-CNN
`TODO`

### Fast-CNN
`TODO`

### Faster-CNN
`TODO`

### Visualizing CNN
`TODO`

### DeepDream
`TODO`

### Transfer Style of image
`TODO`

### Chaos-- Fooling CNN using Adverserial modified images
`TODO`

### Sources