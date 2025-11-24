import torch.nn as nn


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
