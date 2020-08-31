## Model-less Performance Prediction for Neural Architecture Search Acceleration

This repository contains experimentation code for all experiments done in <em>Model-less Performance Prediction for Neural Architecture Search Acceleration</em>.

Additionally, this repository contains a Tensorflow 2.1 implementation of a [NASNet](https://arxiv.org/abs/1707.07012)-like network, with the following properties beneficial to anyone experimenting with NAS:
- Mutation support, with inheritance of weights from parent for all non-mutated ops
- A detailed and object-oriented framework for interacting with every level of abstraction of components within the network (from model, to cell, to group/block, to operation)
- Saving and loading routines for models and model architectures, including an embedding scheme and serialization/deserialization of model hyperparameters and metrics
- A handful of model visualization and analysis tools