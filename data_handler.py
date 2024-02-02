"""
Data Handler for loading and partitioning the CIFAR10 dataset among the participants.
"""
import json
import configparser
import os
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomRotation, RandomHorizontalFlip, Normalize
from torch.utils.data import DataLoader, Subset

from utils import AddGaussianNoise, PoisonedDataset, RepeatSampler

class DataHandler:
    """
    Class to handle loading of CIFAR10 and making dataloaders available for each participant.
    The training set is sampled using Dirichlet distribution to satisfy non-i.i.d requirements.
    """
    def __init__(self, config_path):
        # Load configurations
        self.config = self.load_config(config_path)
        self.num_participants = self.config['Federated'].getint('num_participants')
        self.alpha = self.config['Federated'].getfloat('alpha')
        # Extract poison information from config
        self.poison_target_idx = self.config['Poison'].getint('target_idx')
        self.poison_train_indices = self.config['Poison'].get('train_idxs', '')
        self.poison_test_indices = self.config['Poison'].get('test_idxs', '')
        # Split string into integers
        self.poison_train_indices = [int(elem) for elem in self.poison_train_indices.split('\n')]
        self.poison_test_indices = [int(elem) for elem in self.poison_test_indices.split('\n')]

        # Download CIFAR-10 dataset
        if self.config['DEFAULT'].getboolean('data_augmentation'):
            train_transform = transforms.Compose([RandomRotation(10),
                                                  RandomHorizontalFlip(),
                                                  RandomCrop(size=24),
                                                  ToTensor(),
                                                  Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])
            # Poison data needs additional transform
            poison_transform = transforms.Compose([RandomRotation(10),
                                                  RandomHorizontalFlip(),
                                                  RandomCrop(size=24),
                                                  ToTensor(),
                                                  Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010)),
                                                  AddGaussianNoise(std=0.05)])
        else:
            train_transform = transforms.Compose([ToTensor(),
                                                  Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])
            poison_transform = transforms.Compose([ToTensor(),
                                                   Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010)),
                                                   AddGaussianNoise(std=0.05)])
        # Test transform never has augmentation
        test_transform = transforms.Compose([ToTensor(),
                                             Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])

        self.dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        # Load test dataset
        self.test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        # Load poison dataset - has an extra transform
        self.poison_dataset = CIFAR10(root="./data", train=True, transform=poison_transform)

        # Check if partitions already exist
        partition_path = self.config['Federated'].get('partition_path', './partitions.json')
        if not os.path.exists(partition_path):
            # Partition dataset
            self.partition_dataset(partition_path)

        # Load partitions from file since JSON converts keys to strs
        self.partitions = self.load_partitions(partition_path)

    def load_config(self, config_path):
        """
        Reads general config file.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
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
            # Exclude poison indices from class indices
            clean_class_idx = [idx for idx in class_idx if idx not in self.poison_train_indices+self.poison_test_indices]
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_participants))
            proportions = (proportions * len(clean_class_idx)).astype(int)

            # Ensure sum of proportions equals the class count
            proportions[-1] = len(clean_class_idx) - proportions[:-1].sum()

            # Allocate data to participants
            np.random.shuffle(clean_class_idx)
            allocated_idx = 0
            for i, p in enumerate(proportions):
                partitions[i].extend(clean_class_idx[allocated_idx:allocated_idx + p].tolist())
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

    def get_poison_dataloader(self, participant_id, attacker_target, batch_size=32, poison_per_batch=20):
        """
        Returns a DataLoader for a participant with a mix of clean and poison images in each batch.

        Args:
            participant_id (int): The participant's ID.
            attacker_target (int): The class the attack is targeting.
            batch_size (int): The total number of samples in each batch.
            poison_per_batch (int): The number of poison samples in each batch.
        """
        # Get indices for clean and poison samples
        clean_indices = self.partitions[str(participant_id)]

        # Initialize the mixed dataset
        mixed_dataset = PoisonedDataset(self.dataset,
                                        self.poison_dataset,
                                        clean_indices,
                                        self.poison_train_indices,
                                        poison_per_batch,
                                        attacker_target)

        # Create and return the DataLoader
        return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True)

    def get_test_poison_dataloader(self, batch_size=1, num_samples=1000):
        """
        Returns a DataLoader for poison test images, ensuring each image is evaluated
        multiple times with data augmentations.

        Args:
            batch_size (int): The batch size for the DataLoader.
            num_samples (int): The number of times the test images should be sampled for evaluation.
        """
        # Create a Subset of the poison_dataset using poison_test_indices
        subset_poison_test = Subset(self.poison_dataset, self.poison_test_indices)

        # Initialize the DataLoader with the RepeatSampler
        poison_test_loader = DataLoader(subset_poison_test,
                                        batch_size=batch_size,
                                        sampler=RepeatSampler(len(self.poison_test_indices),
                                                              num_samples))
        return poison_test_loader

    def get_test_dataloader(self, batch_size=32, shuffle=True):
        """
        Returns a DataLoader for the CIFAR-10 test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_global_train_dataloader(self, batch_size=32, shuffle=True):
        """
        Returns a DataLoader for the entire CIFAR-10 training dataset, excluding the backdoor
        training and test images.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: A DataLoader for the CIFAR-10 training dataset.
        """

        all_indices = set(range(len(self.dataset)))
        exclude_indices = set(self.poison_train_indices) | set(self.poison_test_indices)

        # Get the valid indices by excluding poison_indices and poison_test_indices
        valid_indices = list(all_indices - exclude_indices)

        # Create a Subset of the dataset using valid_indices
        valid_subset = Subset(self.dataset, valid_indices)
        return DataLoader(valid_subset, batch_size=batch_size, shuffle=shuffle)
