"""
This module provides utility functions and classes.
One of the utilities included is a class for adding Gaussian noise to tensors
which is required for the poisoned data.

Classes:
    AddGaussianNoise: A callable class that adds Gaussian noise to a given tensor.
"""
import torch

class AddGaussianNoise(object):
    """
    A transformation that adds Gaussian noise to a tensor.

    Attributes:
        mean (float): The mean of the Gaussian noise to be added.
        std (float): The standard deviation of the Gaussian noise to be added.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
