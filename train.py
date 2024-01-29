"""
Main Federated Learning Setup.
There are a set number of rounds, where m participants are chosen
at random to train for E epochs on their local datsets. After training, local models are
sent to update the global model by averaging.
"""

import json
import copy
import numpy as np

from torch import nn
from torch import optim
from .data_handler import DataHandler
from .resnet import resnet18

def load_config(config_path):
    """
    Loads training configuration from a JSON file.
    Args:
        config_path (str): The path to the configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def aggregate_models(global_model, local_models, eta, n):
    """
    Aggregates updates from local models to update the global model.

    Args:
        global_model (torch.nn.Module): The global model to be updated.
        local_models (list of torch.nn.Module): The local models with updates from participants.
        eta (float): The global learning rate used for aggregation.
        n (int): The number of participants selected in the current round.

    Returns:
        torch.nn.Module: The updated global model after aggregation.
    """
    global_state_dict = global_model.state_dict()
    local_updates = [model.state_dict() for model in local_models]
    for name in global_state_dict.keys():
        global_param = global_state_dict[name]
        local_sum = sum(local_update[name] - global_param for local_update in local_updates)
        global_state_dict[name] = global_param + eta / n * local_sum
    global_model.load_state_dict(global_state_dict)
    return global_model

def train_local_model(model, data_loader, epochs, lr):
    """
    Trains a local model on participant's data.

    Args:
        model (torch.nn.Module): The local model to be trained.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the participant's data.
        epochs (int): The number of epochs to train the model.
        lr (float): The learning rate for local training.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def train(config_path):
    """
    Main training function for federated learning setup.

    Args:
        config_path (str): The path to the training configuration file.
    """
    config = load_config(config_path)
    data_handler = DataHandler(config_path)

    global_model = resnet18()

    for federated_round in range(config['num_rounds']):
        selected_participants = np.random.choice(range(config['num_participants']),
                                                 size=config['num_selected'], replace=False)
        print(f"Round {federated_round+1}/{config['num_rounds']}:\
               Selected Participants: {selected_participants}")
        local_models = []

        for participant_id in selected_participants:
            local_model = copy.deepcopy(global_model)
            data_loader = data_handler.get_dataloader(participant_id,
                                                      batch_size=config['batch_size'])
            train_local_model(local_model, data_loader, config['local_epochs'], config['local_lr'])
            local_models.append(local_model)

        global_model = aggregate_models(global_model,
                                        local_models,
                                        config['global_lr'],
                                        config['num_selected'])

        #TODO: Evaluate the global model here to track progress

if __name__ == "__main__":
    train('config.json')
