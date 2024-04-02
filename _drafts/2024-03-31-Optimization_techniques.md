---
title: Optimization techniques
tags: deeplearning
permalink: /:slug/
---

in practice we barely use SDG

problem with SGD

```py
vs = zeros_like(p) for p in parameters
for v, p in zip(vs, parameters)
  v.data = mu * v + lr * dx
x -= v
```

usually we do single mu, but sometime people decay it over time


```py
vs = zeros_like(p) for p in parameters
for v, p in zip(vs, parameters)
  v.data = mu * v + lr * dx
x -= v
```

When we update the velocity \( V \) using the equation \( V = \text{momentum} \times V + \text{lr} \times \text{grad} \), we are essentially accumulating the gradients over time with momentum. Here's a breakdown:

- \(\text{momentum} \times V\): This term represents the momentum from the previous step. It is a fraction (determined by the momentum hyperparameter) of the previous velocity, which serves to maintain directionality and smooth out the updates.
- \(\text{lr} \times \text{grad}\): This term represents the current gradient scaled by the learning rate. It is the immediate influence on the velocity based on the current gradient.

By summing these two terms, we are accumulating the effect of past gradients (with momentum) while also considering the current gradient to update the velocity. This accumulation helps to smooth out the updates and navigate through the optimization landscape more efficiently, especially in regions with high curvature or noisy gradients.

NESTEROV MOMENTUM

```py
for v, p in zip(vs, parameters):
    p_temp = p.data - lr * v
    grad_temp = calculate_gradient_at(p_temp)
    v.data = momentum * v + lr * grad_temp
```

