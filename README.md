# micronet

## Overview

Micronet is a tiny, lightweight python framework for creating and training feed-forward neural networks. Micronet works with custom activation and loss functions, but comes with the most common already provided. 
There is no support for hardware acceleration, but micronet makes use of numpy for efficient matrix operations.

## Example Usage
Micronet provides a simple API inspired by the Keras Sequential API found in Tensorflow.

```python
import micronet
from micronet.activations import relu, softmax

model = MLP([
    Layer(2, 8, relu),
    Layer(8, 8, relu),
    Layer(8, 2, softmax)
])

```

And training the model is as simple as a single line of code.

```python
from micronet.losses import cross_entropy

model.train(xs, ys, loss=cross_entropy, epochs=20, batch_size=32, learning_rate=0.01)
```

## License

MIT