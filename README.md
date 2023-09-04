# SloppyGrad 🤪 : Because Perfection is Overrated

Welcome to the wild and wacky world of SloppyGrad, my attempt to learn neural networks but with sprinkles of chaos, a pinch of instability, and lots of crashes.

## What is SloppyGrad?
SloppyGrad takes inspiration from micrograd and tinygrad, and it's an autograd engine. The primary goal of SloppyGrad is to contribute to tinygrad. After all, the best way to truly understand something is to try and recreate it.

## Getting Started
To get started with SloppyGrad, follow these simple steps:

### Installation:

To install SloppyGrad, you can use pip:

```bash
git clone https://github.com/realabdu/sloppygrad.git
cd sloppygrad
pip install -e .
```
Simple usage example:

!sloppygrad supports 0-dimension Tensors
```python
from sloppygrad.tensor import Tensor
from sloppygrad.nn import MLP


# create a imple multilayer perceptron
# with 3 input and two hidden layer for 4 neurons each
mlp = MLP(3,[4,4,1])

# this mlp has a 41 parameters  that's weights,biases
print(f"MLP has {len(mlp.parameters())} parameters")

# initiate the data exmaples
xs = [
    [2.0,3.0,-1.0],
    [3.5,-1.0,0.5],
    [2.5,1.0,1.5],
    [2.0,-3.0,-1.0]
    ]
# and the expected output with respect to each array
ys = [1.0,-1.0,-1.0,1.0]
# that's if the input is [2,3,-1] the input should be 1


#init loss function
def loss_func(y_preds) -> Tensor:
    return sum((p - y)**2 for y,p in zip(ys,y_preds))


# init hpyer parameters
epochs = 70
lr = 0.01

for step in range(epochs):
    #forward pass
    ypreds = [mlp(x) for x in xs]
    # get the loss
    loss = loss_func(ypreds)

    #backward pass
    loss.backward()

    #optimization loop
    for p in mlp.parameters():
        # say p.grad is -0.8 and lr is 0.01,you would add to minus maknig the model worse so you need to inverse it
        p.data += -lr * p.grad

    for p in mlp.parameters():
        p.grad = 0.0
    print(f"step :{step+1}, loss {loss.data}")

# test out the model
ypred = [mlp(x) for x in xs]
print(ypred)
```

TODO:
- support multi-dimension tensors


Explore the unpredictable world of SloppyGrad, and let's see where chaos and creativity take us! 🚀





