---
title: Understanding loss.backward()
tags: deeplearning
---

Relying on any library is not good; therefore, this time, we remove this black box and understand what's happening inside. Writing a backward pass manually would be helpful. Backpropagation is not something that works magically or automatically if you are hoping to debug. In other words, it is easy to fall into the trap of abstracting away the learning process, believing that you can simply stack arbitrary layers together and backprop will magically make them work on your data. So let's look at a few explicit examples where this is not the case in quite unintuitive ways.


<!-- TOC -->
<!-- INSERT -->

### Simple expressions and interpretation of the gradient


Let's start with simple expressions to develop the notation and conventions for more complex ones.[^1][^2][^3]
<div>
Consider a straightforward multiplication function of two numbers \( f(x,y) = xy \). It's a matter of simple calculus to derive the partial derivative for either input:

\[ \frac{\partial f}{\partial x} = y \quad \text{and} \quad \frac{\partial f}{\partial y} = x \]

<b>Interpretation:</b> Derivatives indicate the rate of change of a function with respect to that variable around an infinitesimally small region near a particular point:

\[ \frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \]

A technical note is that the division sign on the left-hand side is not a division; it indicates that the operator \( \frac{d}{dx} \) is being applied to the function \( f \), returning a different function (the derivative). When \( h \) is very small, the function is well-approximated by a straight line, and the derivative is its slope. In other words, the derivative on each variable tells you the sensitivity of the whole expression to its value. For example, if \( x = 4, y = -3 \), then \( f(x,y) = -12 \) and the derivative on \( x \), \( \frac{\partial f}{\partial x} = -3 \), indicates that increasing the value of \( x \) by a tiny amount would decrease the whole expression by three times that amount. Similarly, since \( \frac{\partial f}{\partial y} = 4 \), increasing \( y \) by a small amount would increase the output of the function by four times that amount.
<br>
<br>
<b>The Gradient:</b> The gradient \( \nabla f \) is the vector of partial derivatives, so we have \( \nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x] \). Though technically a vector, we'll often use terms like "the gradient on \( x \)" instead of the technically correct phrase "the partial derivative on \( x \)" for simplicity.

We can also derive the derivatives for the addition operation:

\[ f(x,y) = x+y \quad \rightarrow \quad \frac{\partial f}{\partial x} = 1 \quad \text{and} \quad \frac{\partial f}{\partial y} = 1 \]

That is, the derivative on both \( x \) and \( y \) is one, regardless of their values. This makes sense since increasing either \( x \) or \( y \) would increase the output of \( f \), and the rate of increase would be independent of their actual values.

Lastly, let's consider the max operation:

\[ f(x,y) = \max(x,y) \quad \rightarrow \quad \frac{\partial f}{\partial x} = \begin{cases} 1 & \text{if } x \geq y \\ 0 & \text{otherwise} \end{cases} \quad \text{and} \quad \frac{\partial f}{\partial y} = \begin{cases} 0 & \text{if } x \geq y \\ 1 & \text{otherwise} \end{cases} \]

Here, the (sub)gradient is 1 on the input that was larger and 0 on the other input. Intuitively, if \( x = 4 \) and \( y = 2 \), then the max is 4, and the function is not sensitive to the setting of \( y \). That is, if we were to increase it by a tiny amount \( h \), the function would keep outputting 4, and therefore the gradient is zero—there's no effect. Of course, if we were to change \( y \) by a large amount (e.g., larger than 2), then the value of \( f \) would change, but the derivatives tell us nothing about the effect of such large changes on the inputs of a function; they are only informative for tiny, infinitesimally small changes, as indicated by the \( \lim_{h \to 0} \) in their definition.
</div>
<br>


### Compound Expressions with Chain Rule

<div>
Let's now consider more complicated expressions that involve multiple composed functions, such as \( f(x,y,z) = (x+y)z \). While this expression is simple enough to differentiate directly, we'll take a particular approach that will be helpful for understanding the intuition behind backpropagation. Specifically, note that this expression can be broken down into two expressions: \( q = x+y \) and \( f = qz \). Moreover, we know how to compute the derivatives of both expressions separately, as seen in the previous section. \( f \) is just the multiplication of \( q \) and \( z \), so \( \frac{\partial f}{\partial q} = z \), \( \frac{\partial f}{\partial z} = q \), and \( q \) is the addition of \( x \) and \( y \) so \( \frac{\partial q}{\partial x} = 1 \), \( \frac{\partial q}{\partial y} = 1 \).

However, we don’t necessarily care about the gradient on the intermediate value \( q \)—the value of \( \frac{\partial f}{\partial q} \) is not useful. Instead, we are ultimately interested in the gradient of \( f \) with respect to its inputs \( x, y, z \). The chain rule tells us that the correct way to "chain" these gradient expressions together is through multiplication. For example, \( \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x} \). In practice, this is simply a multiplication of the two numbers that hold the two gradients. Let's see this with an example:

```python
# Set some inputs
x = -2; y = 5; z = -4

# Perform the forward pass
q = x + y  # q becomes 3
f = q * z  # f becomes -12

# Perform the backward pass (backpropagation) in reverse order:
# First backprop through f = q * z
dfdz = q  # df/dz = q, so gradient on z becomes 3
dfdq = z  # df/dq = z, so gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
# Now backprop through q = x + y
dfdx = dfdq * dqdx  # The multiplication here is the chain rule!
dfdy = dfdq * dqdy  
```

We are left with the gradient in the variables \([ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} ]\), which tell us the sensitivity of the variables \( x, y, z \) on \( f \). This is the simplest example of backpropagation. Going forward, we will use a more concise notation that omits the \( df \) prefix. For example, we will simply write \( dq \) instead of \( \frac{\partial f}{\partial q} \), and always assume that the gradient is computed on the final output.

</div>


Let's take a fully connected layer with sigmoid non-linearity computes (using raw numpy):

```py
h = 1/(1 + np.exp(-np.dot(W, x))) # forward pass
dx = np.dot(x.T, h * (1-h))       # backward pass: local gradient for x
dW = np.dot(np.diag(dz), x.T)     # backward pass: local gradient for W
```

*Applying quotient rule*. **d/dx(sigmoid(x))** and then we get, **h = sigmoid(x)**, replacing that, we get **sigmoid(x) * (1 - sigmoid(x)) = h(1-h)**. If your weight matrix **W** is initialized too large, the output of the matrix multiply could have a very large range (e.g. numbers between -400 and 400), which will make all outputs in the vector *h* almost binary: either 1 or 0. But if that is the case, **h(1-h)**, which is local gradient of the sigmoid non-linearity, will in both cases become zero (“vanish”), making the gradient for both x and W be zero. The rest of the backward pass will come out all zero from this point on due to multiplication in the chain rule.

### Forward pass

`TODO`

### Backward pass

`TODO`

### Manual backward pass

`TODO`

### Slightly better

`TODO`

### Gradient checking

`TODO`

### Sources


[^1]: Inspired by [CS231n notes](https://cs231n.github.io/optimization-2/)
[^2]: This blog is written as notes to this [youtube lecture](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5).
[^3]: CS231n Lecture on backprop can be found [here](https://www.youtube.com/@andrejkarpathy4906).