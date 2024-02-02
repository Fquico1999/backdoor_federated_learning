"""
This module provides utility functions and classes.
One of the utilities included is a class for adding Gaussian noise to tensors
which is required for the poisoned data.

Classes:
    AddGaussianNoise: A callable class that adds Gaussian noise to a given tensor.
    PoisonedDataset: A custom dataset class to mix backdoor and clean data in proper amounts.
"""
import torch
from torch.utils.data import Dataset

class AddGaussianNoise():
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

class PoisonedDataset(Dataset):
    """
    Class that combines a clean and poisoned dataset and allows for mixed batches,
    determined by mixing parameter poison_per_batch.
    """
    def __init__(self, clean_dataset, poison_dataset, clean_indices, poison_indices, poison_per_batch):
        """
        Initializes the mixed dataset with specified clean and poison subsets and mixing parameters.

        Args:
            clean_dataset (Dataset): The clean dataset with its original transforms.
            poison_dataset (Dataset): The poisoned dataset with its specific transforms.
            clean_indices (list): Indices of clean samples in the clean dataset.
            poison_indices (list): Indices of poison samples in the poison dataset.
            poison_per_batch (int): Number of poison samples to include in each batch.
        """
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.clean_indices = clean_indices
        self.poison_indices = poison_indices
        self.poison_per_batch = poison_per_batch
        self.clean_per_batch = len(clean_indices) // (len(clean_indices) // poison_per_batch)

    def __getitem__(self, index):
        """
        Retrieves an item from the mixed dataset.
        The mixing is done on-the-fly based on the specified index.

        Args:
            index (int): The index of the item to fetch.
        """
        if index % (self.clean_per_batch + self.poison_per_batch) < self.clean_per_batch:
            # Fetch from clean dataset
            clean_index = self.clean_indices[index % len(self.clean_indices)]
            return self.clean_dataset[clean_index]

        # Fetch from poison dataset
        poison_index = self.poison_indices[index % len(self.poison_indices)]
        return self.poison_dataset[poison_index]

    def __len__(self):
        """
        Returns the total length of the mixed dataset,
        calculated based on the mixing ratio and the sizes of the clean and poison subsets.

        Returns:
            int: Total number of samples in the mixed dataset.
        """
        return len(self.clean_indices) + len(self.poison_indices)
