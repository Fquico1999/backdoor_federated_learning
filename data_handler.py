"""
Data Handler for loading and partitioning the CIFAR10 dataset among the participants.
"""

import json
import os
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

class DataHandler:
    """
    Class to handle loading of CIFAR10 and making dataloaders available for each participant.
    The training set is sampled using Dirichlet distribution to satisfy non-i.i.d requirements.
    """
    def __init__(self, config_path):
        # Load configurations
        self.config = self.load_config(config_path)
        self.num_participants = self.config['num_participants']
        self.alpha = self.config['alpha']

        # Download CIFAR-10 dataset
        self.dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())

        # Check if partitions already exist
        partition_path = self.config.get('partition_path', './partitions.json')
        if not os.path.exists(partition_path):
            # Partition dataset
            self.partition_dataset(partition_path)

        # Load partitions from file since JSON converts keys to strs
        self.partitions = self.load_partitions(partition_path)

    def load_config(self, config_path):
        """
        Reads general config file.
        """
        with open(config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)
        return config

    def load_partitions(self, partition_path):
        """
        Loads partitions from file.
        """
        with open(partition_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def partition_dataset(self, partition_path):
        """
        Partition CIFAR10 with Dirichlet, and saves partitioning to file.
        """
        # Initialize partitions
        partitions = {i: [] for i in range(self.num_participants)}

        # Get class distribution
        class_indices = [np.where(np.array(self.dataset.targets) == i)[0] for i in range(10)]

        # Partition dataset using Dirichlet distribution
        for class_idx in class_indices:
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_participants))
            proportions = (proportions * len(class_idx)).astype(int)

            # Ensure sum of proportions equals the class count
            proportions[-1] = len(class_idx) - proportions[:-1].sum()

            # Allocate data to participants
            np.random.shuffle(class_idx)
            allocated_idx = 0
            for i, p in enumerate(proportions):
                partitions[i].extend(class_idx[allocated_idx:allocated_idx + p].tolist())
                allocated_idx += p

        # Save partitions to a file
        with open(partition_path, 'w', encoding="utf-8") as f:
            json.dump(partitions, f)

    def get_dataloader(self, participant_id, batch_size=32):
        """
        Make partition for each participant available as a DataLoader
        """
        # Get dataset subset for the participant
        subset = Subset(self.dataset, self.partitions[str(participant_id)])
        # Create and return DataLoader
        return DataLoader(subset, batch_size=batch_size, shuffle=True)
