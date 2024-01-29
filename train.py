"""
Main Federated Learning Setup.
There are a set number of rounds, where m participants are chosen
at random to train for E epochs on their local datsets. After training, local models are
sent to update the global model by averaging.
"""

import json
import copy
import numpy as np

import torch
from torch import nn
from torch import optim
from data_handler import DataHandler
from resnet import resnet18

def load_config(config_path):
    """
    Loads training configuration from a JSON file.
    Args:
        config_path (str): The path to the configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def evaluate_model(model, test_loader, device):
    """
    Evaluates the given model's performance on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

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
    # Accumulate updates from each local model
    for name in global_state_dict.keys():
        global_param = global_state_dict[name]
        local_sum = sum([local_update[name] - global_param for local_update in local_updates])
        global_state_dict[name] = global_param + eta / n * local_sum
    # Apply the aggregated updates to the global model
    global_model.load_state_dict(global_state_dict)
    return global_model

def train_local_model(model, data_loader, epochs, lr, device, verbose=False): #pylint: disable=too-many-arguments
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
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Participant Training - Epoch: {epoch+1}/{epochs}, "
                   f"Loss: {total_loss/len(data_loader)}")

def train(config_path):
    """
    Main training function for federated learning setup.

    Args:
        config_path (str): The path to the training configuration file.
    """
    config = load_config(config_path)
    data_handler = DataHandler(config_path)

    # Create test DataLoder for global evaluation
    test_loader = data_handler.get_test_dataloader(batch_size=config['batch_size'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_model = resnet18().to(device)
    global_model.train()

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
            if config['verbose']:
                print(f"Participant ID: {participant_id}")

            train_local_model(local_model,
                              data_loader,
                              config['local_epochs'],
                              config['local_lr'],
                              device,
                              verbose=config['verbose'])

            local_models.append(local_model)

        global_model = aggregate_models(global_model,
                                        local_models,
                                        config['global_lr'],
                                        config['num_selected'])

        # Evaluate the global model
        accuracy = evaluate_model(global_model, test_loader, device)
        print(f"Global Model Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train('config.json')
