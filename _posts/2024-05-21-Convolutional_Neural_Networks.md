---
title: CNNs
tags: deeplearning
---

This is a quick skim notes for [CS231n Introduction to CNN lecture 7](https://www.youtube.com/watch?v=LxfUGhug-iQ&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=8), I used slides from this lecture to create notes and this is NOT an attempt to replicate [notes by cs231n](https://cs231n.github.io/convolutional-networks/), its already the best notes on CNNs out there.

CNNs are similar to ordinary neural networks, they have trainable weights and bias, receives input, bunch of trainable layers followed with non-linearity. Each layer is completely differentiable, means they can learn. At the end an output layer predicting classes or so.

So what changes? ConvNet architecture make the explicit assumption that the inputs are image which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameter in the network.

**Why can't regular NNs don't scale to images?**  

Consider an image from CIFAR-10 dataset, size 32x32x3, input neurons would be 32x32x3 = 3072 weights. Good? Now consider an image 200x200x3 image, 120,000 weights in the first layer, moreover we would almost certainly want to have several such neurons, so the parameters would add up quickly. Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

**3D volumes of neurons**  

CNN's take the advantage of the fact that the input are images and use the architecture in a more sensible way. In particular, unlike the regular neural networks the layer of convolution net have neurons arranged in three dimensions: Height, width and depth. The depth here is not the depth of the whole neural net. But instead, it's the depth of trainable weights. A good visualization:

<hr>
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/neural_net2.jpeg" alt='neural net' style="max-width: 70%; height: auto; align: center">
  <hr>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn.jpeg" alt='cnn' style="max-width: 100%; height: auto; align: center">
  <figcaption>Top: A regular 3-layer Neural Network. Bottom: A ConvNet arranges its neurons in three dimensions (height, width, depth) as visualized in one of the layers. Every layer of ConvNet transforms the 3D output volume of neuron activations. In this example the red block is the image, so its dimensions would be, height (28), width (28) and depth (3, because 3 channels ~RGB) </figcaption>
</figure>
<hr>

## Convolve
*Filters*, also called *kernels*, are small (typically 3x3 or 5x5) used to slide over the image spatially, computing dot products. The filters also have depth, eg. 3x3x96 here 96 is depth of the kernel. The depth of kernel should be equal to the depth of the input. If the input is 32x32x3 image, where 3 is ~RGB. then kernel would have depth 3 (3x3x**3** or 5x5x**3**).

> in the above case, **spatial** dimension is the 32x32 (height and width of input).

<hr>
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn1.png" alt='cnn1' style="max-width: 100%; height: auto; align: center">
  <figcaption>Kernels slide over the image, to convolve/feature extract.
  </figcaption>

  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn2.png" alt='cnn2' style="max-width: 100%; height: auto; align: center">
  <figcaption>The dimension of the activation map (right) depends on input (left) size & depth and the kernel size & depth. Notice the reduction in size of the output image.
  </figcaption>

  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn3.png" alt='cnn3' style="max-width: 100%; height: auto; align: center">
  <figcaption>let's say, there are 2 input channel (shown in the image 'consider a second green filter', blue and green) the kernel will also be slide through all location of the 2nd channel. Creates another output activations. Each kernel create a output activation for each channel.
  </figcaption>

  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn4.png" alt='cnn4' style="max-width: 100%; height: auto; align: center">
  <figcaption>Now, if there are 6, 5x5 kernel. we will get 6 seperate activation maps.  
  6x28x28 in this case.
  </figcaption>

  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/cnn5.png" alt='cnn5' style="max-width: 100%; height: auto; align: center">
  <figcaption>Stack this up, plus non-linearlity (ReLU) on the activation maps. </figcaption>

  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/math.png" alt='math' style="max-width: 100%; height: auto; align: center">
  <figcaption>an overview of input to the conv layers and its output. each convs are followed by non-linearity (for example, ReLU), just stack them and we have a architecture?</figcaption>
</figure>
<hr>

**An overview of the ConvNet architecture**  
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/convnet.jpeg" alt='convnet' style="max-width: 100%; height: auto; align: center">
  <figcaption>The activations of an example ConvNet architecture. The initial volume stores the raw image pixels (left) and the last volume stores the class scores (right). The middle shows visulizations of activations, since it is difficult to visualize 3D, here the representation is in 2D.</figcaption>
</figure>

**How do we know, the size of the output after convolution?**  
`(N-F)/S + 1`, where `N` is the number if input dimension, `F` is the kernel dimension. `S` is stride.

**Oh yes, what is Stride?**  

Kernels slide through the input layer, **Stride** refers to number of pixels the filter mover over the input image during convolution ([see some gifs](https://github.com/jerpint/cnn-cheatsheet?tab=readme-ov-file#strided-1-and-stride-2))

**Padding**  

In practice, it is common to pad the input image with 0s around the edges.

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/pad.png" alt='pad' style="max-width: 100%; height: auto; align: center">
  <figcaption>Good effect, we can control the spatial size of the output. Example, to retain the original size of input image, we can use (F-1)/2 padding. where F is the filter dim. </figcaption>
  <br>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/pad2.png" alt='pad2' style="max-width: 100%; height: auto; align: center">
  <figcaption>You see.</figcaption>
</figure>

**math time, calculate how many trainable params**  

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/params.png" alt='params' style="max-width: 100%; height: auto; align: center">
  <figcaption>kernels are trainable</figcaption>
  <hr>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/common.png" alt='params' style="max-width: 100%; height: auto; align: center">
  <figcaption>some common settings for Convolutions</figcaption>
</figure>


**we can also have 1x1 Kernels**  

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/1x1.png" alt='1x1' style="max-width: 100%; height: auto; align: center">
  <figcaption>1x1 kernels are used for dimensionality reduction without altering spatial dimensions, 64 -> 32 (halved but the spatial dimension remains same)</figcaption> 
</figure>

### Summary 1: Convolution Layer
- Accepts input of shape **W, H, D**
- Requires hyperparameters
  - Number of Kernels-- **N**
  - Kernel dim-- **F**
  - Stride-- **S**
  - Padding-- **P**
<hr>

**The brain/neuron view of CONV layer**  

Recall how the activation map depth is equal to the number of kernels.

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/neuron.png" alt='neuron' style="max-width: 100%; height: auto; align: center">
  <figcaption>a neuron view of convolution</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/neuron2.png" alt='neuron2' style="max-width: 100%; height: auto; align: center" style="max-width: 100%; height: auto; align: center">
  <figcaption>the output of a patch from img and element wise multiplication of kernel is the size of kernel itself, then we calculate the sum of it. ~Dot product. This creates a single activation in the activation layer</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/neuron3.png" alt='neuron3' style="max-width: 100%; height: auto; align: center" style="max-width: 100%; height: auto; align: center">
  <figcaption>Create layers spatially, they are not interconnected. It similar to channels stacked.</figcaption>
</figure>

**Pooling Layer**  

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/pool.png" alt='pool' style="max-width: 100%; height: auto; align: center">
  <figcaption>This throws away some spatial information, but dont worry, not all information is equally valuable. Hypothetically, Pooling layer discards some less important details.</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/pool2.png" alt='pool2' style="max-width: 100%; height: auto; align: center">
  <figcaption>Max pooling is generally better than average pooling because it preserves the most prominent features and introduces non-linearity, making it more effective in capturing key patterns and reducing overfitting.</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/pool3.png" alt='pool3' style="max-width: 100%; height: auto; align: center">
</figure>
<hr>

### Case Study

#### **LeNet-5**
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/lenet5.png" alt='lenet5' style="max-width: 100%; height: auto; align: center">
  <figcaption>Architecture: [Conv-pool-Conv-pool-conv-FC]</figcaption>
</figure>

<hr>

#### **AlexNet**
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/alex.png" alt='alexnet' style="max-width: 100%; height: auto; align: center">
  <figcaption>Increased the kernel depth, from 6 to 96. Applied kernel of size 11x11 at stride of 4.</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/alex2.png" alt='alexnet2' style="max-width: 100%; height: auto; align: center">
  <figcaption>AlexNet was the first paper to use ReLU. This made ReLU popular. There are ReLU after each convolution layers and FC layers. We don't use the NORM layers used in AlexNet now, because it doesn't actually gives any improvement.</figcaption>
</figure>

Architecture of AlexNet: `C = Conv`, `P = Pool`, `O = Output layer`  
`[C-P-C-P-C-C-C-P-FC-FC-O-Softmax]`

some details:  
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/alex3.png" alt='alexnet3' style="max-width: 100%; height: auto; align: center">
  <figcaption>LR is reduced by 10, when Val accuracy plateaus</figcaption>
</figure>

<hr>

#### **ZFNet**
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/ZFnet.png" alt='ZFnet' style="max-width: 100%; height: auto; align: center">
  <figcaption>Similar architecture to AlexNet but increased number of kernels (N)
  </figcaption>
</figure>

<hr>

#### **VGGNet**
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/vgg.png" alt='vgg' style="max-width: 100%; height: auto; align: center">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/vgg2.png" alt='vgg2' style="max-width: 100%; height: auto; align: center">
  <figcaption>A rough diagram that I made, shows there are 13 Conv Layers used in VGG net (there are also pooling)</figcaption>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/vgg3.png" alt='vgg3' style="max-width: 100%; height: auto; align: center">
  <figcaption>~by Andrej Karpathy</figcaption>
</figure>

Memory footprint of VGGNet shows,
- Most memory is in early CONV layer (3M)
- Most Parameters are in late FC layers (10M)
- 93MB/image in forward pass alone

**Average Pooling Layer replaces FC layer in the end**  
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/avgpool.png" alt='avgpool' style="max-width: 100%; height: auto; align: center">
</figure>
The example is from the FC layer of VGG net, the last POOL2 output is of shape: 7x7x512, where 7x7 are spatial dim and 512 is depth/volume. each 7x7 is averaged pool to 1. reducing the size quite a lot.

<hr>

#### **GoogLeNet** (introduced inception)
<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/google.png" alt='google' style="max-width: 100%; height: auto; align: center">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/google2.png" alt='google2' style="max-width: 100%; height: auto; align: center">
  <figcaption>This paper introduced Inception module, remove last FC layer with AveragePool layer to reduce parameters.</figcaption>
</figure>

<hr>

#### **ResNet**

In 2015, ResNet won 1st place in many competition at the same time. GoogleNet was 22 layers, introduced resnet was 152 layers. 

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/meme.png" alt='meme' style="max-width: 50%; height: auto; align: center">
  <figcaption><br>So increasing layers == win?</figcaption><br>
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/resnet.png" alt='resnet' style="max-width: 100%; height: auto; align: center">
  <figcaption>Well, yes thats what scaling hypothesis says. But increasing layers is challenging. Problems with vanilla nets, increasing the number of layers converges in training (left img, dashed lines) but the validation error of 56 layer > 20 layer net. Which makes no sense.</figcaption>
</figure>

ResNet took, 2-3 weeks of training on 8 GPU, but at runtime, its faster than VGGNet, even though it has 8x more layers.

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/resnet2.png" alt='resnet2' style="max-width: 100%; height: auto; align: center">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/resnet3.png" alt='resnet3' style="max-width: 100%; height: auto; align: center">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/resnet4.png" alt='resnet4' style="max-width: 100%; height: auto; align: center">
  <figcaption>Highway network is also a good paper by Jürgen Schmidhuber. Introduced just before ResNet. The Skip connection in Highway network had trainable weights. where as ResNet Skip connections is literally vanilla addition of input to output.</figcaption>
</figure>

<hr>

#### **AlphaGo** (policy network?)

Playing Go (harder than chess) using CNNs? The 48 rules of playing Go embedding into the image's channel?

<figure style="text-align: center;">
  <img src="{{site.baseurl}}/assets/Convolutional_Neural_Networks/go.png" alt='go' style="max-width: 100%; height: auto; align: center">
</figure>

### Summary 2: Trends
- ConvNets Stack Conv, pool, FC layers
- Smaller filters and deep architecture
- get rid of Pool/FC layers (just convs) 
  - Treads toward stride-convs only layer, where you try to reduce image dimension (not depth) using convs instead of using pooling layer.
  - ResNet/GoogleNet challenge this paradigm


<!--
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
`TODO` -->

### Sources

- [Andrej Karpathy lecture on CNNs](https://youtu.be/LxfUGhug-iQ?si=Sq-eXEBLHASwBR_i)