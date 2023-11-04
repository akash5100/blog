---
title: Algorithm behind universal function approximator
categories: deeplearning
---

A artificial neural network can learn (almost) anything, and so its called a universal function approximator. To understand how it works, we need to know function.

Function, let's say f(x) is just a system of inputs and outputs, a number in, a number out.

x -> f(x) -> y

We give a input x, and it outputs y. We can plot all the functions on a graph, where it gives an output for an input. What is important is, if you know a function you can always calculate the output (y) for a given input (x).

But let's say, we dont know what the function is, but we know some of the inputs and outputs. Is there a way we can reverse engineer that to produce that function?

We can still capture some x's and y's, making predictions. What we need to do is function approximation and more generally a function approximator.

f(x) ≈ function approximator ≈ N(x)

That is what a **neural network** is.

Neural networks are made up of neurons and a neuron itself is just a function, it can take any number of input and gives one output. Each inputs are multiplied together by a weight and added together along with a bias. The weights and bias makes up the parameters of the neuron, and values of weights and bias can change as the network learn.

```py
N(x1, x2, ...,xn) = w1*x1 + w2*x2 + ... wn*xn + b
# or simply
N(x) = w*x + b
```

Neuron is building block and it can be combined with other neuron to form a more complicated function, one built from lots of linear functions.

There is one big problem, linear function can combine to give a linear function. We need to make something more than just a line, we need non linearity.

<hr />
<br />
Let's use ReLU, we use it as our activation function. means we simply apply it to our previous naive neuron.

```py
ReLU(x) = max(x,0)

N(x) = max((w*x + b), 0)
```

How do we find the weights and biases automatically?
<br>
The most common algorithm for this is called [Backpropogation](https://en.wikipedia.org/wiki/Backpropagation).

The time I wrote this blog, I was coding a neural net(from scratch) to train [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, a dataset of handdrawn numbers (0-9) so that the neural net can classify any input.

To train images, let's say 3x3 pixel image (just an example)

![image](https://github.com/akash5100/blog/assets/53405133/0501fc39-a9c7-4394-ae32-8dbd82963d6c)

We [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) the matrix (image represented on 2D array)

![image](https://github.com/akash5100/blog/assets/53405133/7ba14fba-1370-446f-b893-4df9fc902864)

and the same thing, let's represent the pixels with 0-9.

![image](https://github.com/akash5100/blog/assets/53405133/fba41914-233b-4dbf-9202-026bf25347b8)

This is the dot product of Input array and weights with a bias.

To take the dot product multiply each input by each weight and then add them all up.

![image](https://github.com/akash5100/blog/assets/53405133/dcd744b4-c278-44fb-af72-1009707d6f9b)

![image](https://github.com/akash5100/blog/assets/53405133/38830160-b362-41ad-a3d0-327bcd8e69c7)

This dot product is passed into an activation function, in this case a ReLU.

```
max(-0.26, 0) = 0
```


We feed the original inputs to a layer of neurons, each with their own weight and with their own learned value. The values of the weights and biases are calculated through the training process. We give the network input, and it produces an output. We compare the output with the actual output. This comparison is called the loss, which quantifies the difference between our prediction and the actual value.

![image](https://github.com/akash5100/blog/assets/53405133/71d57e06-ba9f-4813-ba2f-948387d097a6)

We now have to find out, what changes we can make to weights and biases of these neural net to reduce the loss. As we dont know what the function is in the neuron, we need to make changes to weights to see that if that change makes the loss go up or down. (Bruh! millions of neuron and for each neuron how many calculations???)

Calculating Gradient
---
The one magic step is the bit where we calculate the gradients. We use calculus as a performance optimization; it allows us to more quickly calculate whether our loss will go up or down when we adjust our parameters up or down. In other words, the gradients will tell us how much we have to change  each weight to make our neural net better. Derivative of a function tells you how much a change in its parameters will change its result. The key point about a derivative is this: for any function, such as the quadratic function, we can calculate its derivative. The derivative is another function. It calculates the change, rather than the value. For instance, the derivative of the quadratic function at the value 3 tells us how  rapidly the function changes at the value 3. More specifically, you may recall that  gradient is defined as rise/run; that is, the change in the value of the function, divided  by the change in the value of the parameter. 

When we know how our function will change, we know what we need to do to make it smaller. This is the key to machine  learning: having a way to change the parameters of a function to make it smaller. Calculus  provides us with a computational shortcut, the derivative, which lets us directly  calculate the gradients of our functions.

> Life would probably be easier if backpropagation (backward pass) was just called calculate_gradient, but deep learning folks really do like to add jargon everywhere they can

Now that we know what changes to make to weights, we do that and repeat the step.

> btw, I forgot to write earlier -- we initialize the parameters to random values :)

Here are the steps:

![image](https://github.com/akash5100/blog/assets/53405133/587b4f48-b318-4e43-bff1-e96f028d0f61)

adding some flavor to Gradient Decent makes it Stochastic gradient descent (SGD).

ChatGPT: Stochastic Gradient Descent (SGD) is like spice in the recipe of Gradient Descent. In standard Gradient Descent, you calculate the average gradient using the entire dataset, which can be computationally expensive. In SGD, you spice things up by randomly selecting a single data point or a small batch of data points for each iteration. This randomness introduces noise, but it speeds up the process, making it more like a spicy, fast-paced version of Gradient Descent.


<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/2zYzyRzz6pRmhPzyfMEC8s?utm_source=generator" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>