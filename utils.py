"""
This module provides utility functions and classes.
One of the utilities included is a class for adding Gaussian noise to tensors
which is required for the poisoned data.

Classes:
    AddGaussianNoise: A callable class that adds Gaussian noise to a given tensor.
    PoisonedDataset: A custom dataset class to mix backdoor and clean data in proper amounts.
    RepeatSampler: A sampler class to allow for sampling of test poison dataset N times.
"""

import torch
from torch.utils.data import Dataset, Sampler

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
    def __init__(self, clean_dataset, poison_dataset, clean_indices, poison_indices, poison_per_batch, attacker_target):
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
        self.attacker_target = attacker_target
        self.clean_per_batch = len(clean_indices) // (len(clean_indices) // poison_per_batch)

    def __getitem__(self, index):
        """
        Retrieves an item from the mixed dataset.
        The mixing is done on-the-fly based on the specified index.

        Args:
            index (int): The index of the item to fetch.
        """
        batch_num = index // (self.clean_per_batch + self.poison_per_batch)
        index_in_batch = index % (self.clean_per_batch + self.poison_per_batch)

        if index_in_batch < self.clean_per_batch:
            # Fetch from clean dataset
            clean_index = (batch_num * self.clean_per_batch + index_in_batch) % len(self.clean_indices)
            img, target = self.clean_dataset[self.clean_indices[clean_index]]
        else:
            # Fetch from poison dataset and change the target to attacker's target
            poison_index = ((index_in_batch - self.clean_per_batch) +\
                            batch_num * self.poison_per_batch) % len(self.poison_indices)
            img, _ = self.poison_dataset[self.poison_indices[poison_index]]
            target = self.attacker_target  # Change target to attacker's target

        return img, target

    def __len__(self):
        """
        Returns the total length of the mixed dataset,
        calculated based on the mixing ratio and the sizes of the clean and poison subsets.

        Returns:
            int: Total number of samples in the mixed dataset.
        """
        return len(self.clean_indices) + len(self.poison_indices)

class RepeatSampler(Sampler):
    """
    A custom sampler that repeats a given list of indices multiple times to achieve
    repeated sampling of specific items in a dataset, useful for repeated evaluation
    with data augmentations.

    Args:
        dataset_size (int): Length of the dataset
        num_samples (int): The total number of times to sample.
    """

    def __init__(self, dataset_size, num_samples):
        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.num_repeats = num_samples // dataset_size

    def __iter__(self):
        for idx in range(self.dataset_size):
            for _ in range(self.num_repeats):
                yield idx

    def __len__(self):
        return self.num_samples
