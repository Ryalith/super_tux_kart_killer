import torch
import torch.nn as nn
import numpy as np
from typing import List


def build_mlp(dims, activation=nn.ReLU(), dropout=0.0):
    """
    Builds an MLP network based on the provided dimensions and activation function.

    Args:
        dims (list): List of integers, where each integer represents the size of a layer.
        activation (nn.Module): The activation function to apply after each layer. Default is nn.ReLU().
        dropout (float): Dropout rate to apply after each layer. Default is 0.0 (no dropout).

    Returns:
        nn.Module: The constructed MLP model.
    """
    layers = []

    # Loop through the layers, connecting each to the next one
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))  # Linear layer
        layers.append(activation)  # Activation function

        if dropout > 0:  # If dropout is specified, add it
            layers.append(nn.Dropout(dropout))

    # Return the model as a nn.Sequential object
    return nn.Sequential(*layers)


def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Function used for orthogonal initialization of the layers. Taken from here
    in the cleanRL library:
    https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_ortho_mlp(sizes: List[int], activation, output_activation=nn.Identity()):
    r"""Helper function to build a multi-layer perceptron

    function from $\mathbb R^n$ to $\mathbb R^p$ with orthogonal initialization

    :param sizes: the number of neurons at each layer
    :param activation: a PyTorch activation function (after each
        layer but the last)
    :param output_activation: a PyTorch activation function (last
        layer)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), act]
    return nn.Sequential(*layers)


def stk_state_dict_to_tensor(state) -> torch.Tensor:
    raise NotImplementedError()


def stk_action_dict_to_tensor(action) -> torch.Tensor:
    raise NotImplementedError()


def stk_state_tensor_to_dict(state) -> dict:
    raise NotImplementedError()


def stk_action_tensor_to_dict(action) -> dict:
    raise NotImplementedError()
