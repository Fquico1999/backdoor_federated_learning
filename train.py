"""
Main Federated Learning Setup.
There are a set number of rounds, where m participants are chosen
at random to train for E epochs on their local datsets. After training, local models are
sent to update the global model by averaging.
"""
import configparser
import concurrent.futures
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For a nicer confusion matrix visualization

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
    config = configparser.ConfigParser()
    config.read(config_path)
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
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    return accuracy, average_loss

def evaluate_backdoor(model, poison_test_loader, attacker_target, device):
    """
    Evaluates the backdoor performance.

    Args:
        model (torch.nn.Module): The model to evaluate.
        poison_test_loader (DataLoader): DataLoader for the poison test dataset.

    Returns:
        float: The backdoor accuracy of the model on the poison test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in poison_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # These are the attackers target labels
            target_labels = torch.ones_like(labels, device=device)*attacker_target
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # Backdoor accuracy is the fraction of inputs that were misclassified as attacker_target
            total += labels.size(0)
            correct += (predicted == target_labels).sum().item()

    accuracy = correct / total
    return accuracy

def plot_confusion_matrix(model, dataloader, device, class_names, savefig=None):
    """
    Plots the confusion matrix for the predictions made by the global model on a given dataset.

    Args:
        model (torch.nn.Module): The trained global model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device on which the model and data are located.
        class_names (list of str): List of class names corresponding to dataset labels.
        savefig (str): Filename to save plot.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Storage for true labels and predictions
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to get outputs
            outputs = model(inputs)

            # Convert outputs to predicted class indices
            _, preds = torch.max(outputs, 1)

            # Append batch predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    if savefig:
        plt.savefig(savefig)
    plt.show()

def local_model_update_sum(global_model, local_state_dicts, device):
    """
    Aggregate local updates. Used to confirm if the assumption that local model updates,
    when aggregated, tend to zero holds.

    Args:
        global_model (torch.nn.Module): The current global model.
        local_state_dicts (list of dict):   The state dictionaries from local models.
                                            Excludes attacker model
        device (torch.device): Device where data and model is held.
    """
    global_state_dict = global_model.state_dict()
    local_updates_dict = {}

    for name in global_state_dict.keys():
        if "bn" not in name and "downsample" not in name:
            global_param = global_state_dict[name]
            local_updates = [local_state_dict[name].to(device) - global_param\
                             for local_state_dict in local_state_dicts]
            local_updates_dict[name] = sum(local_updates)
    local_updates_avg = [torch.mean(val).item() for val in local_updates_dict.values()]
    return np.mean(local_updates_avg), local_updates_dict


def aggregate_models(global_model, local_state_dicts, eta, n, device):
    """
    Aggregates updates from local models' state dictionaries to update the global model.

    Args:
        global_model (torch.nn.Module): The global model to be updated.
        local_state_dicts (list of dict): The state dictionaries from local models with updates.
        eta (float): The global learning rate used for aggregation.
        n (int): The number of participants selected in the current round.
        device (torch.device): The device on which to perform the aggregation.
    """
    global_state_dict = global_model.state_dict()

    # Accumulate updates from each local model's state dictionary
    for name in global_state_dict.keys():
        global_param = global_state_dict[name]
        # Ensure each parameter update is moved to the correct device before accumulation
        local_updates = [(local_state_dict[name].to(device)\
                         if torch.is_tensor(local_state_dict[name])\
                         else torch.tensor(local_state_dict[name], device=device)) - global_param\
                            for local_state_dict in local_state_dicts]

        global_state_dict[name] = global_param + sum(local_updates)*(float(eta)/float(n))

    # Apply the aggregated updates to the global model
    global_model.load_state_dict(global_state_dict)
    return global_model


def train_local_model(participant_id, global_state_dict, data_handler, device, config):
    """
    Trains a local model for a specific participant in the federated learning setup.

    Args:
        participant_id (int): The unique identifier for the participant.
        global_state_dict (dict): The state dictionary of the global model to ensure all
            local models start from the same state.
        data_handler (DataHandler): An instance of the DataHandler class that provides
            access to the dataset and dataloaders.
        device (torch.device): The device (CPU/GPU) on which the model will be trained.
        config (dict): A configuration dictionary containing training parameters such as
            'local_lr' for local learning rate, 'batch_size', 'local_epochs', and 'verbose'
            for logging verbosity.

    Returns:
        dict: The state dictionary of the locally trained model, ready to be aggregated
        into the global model.

    """
    if config['DEFAULT'].getboolean('verbose'):
        print(f"Participant ID: {participant_id}")

    local_model = resnet18().to(device)
    local_model.load_state_dict(global_state_dict)

    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=config['Federated'].getfloat('local_lr'))

    data_loader = data_handler.get_dataloader(
        participant_id,
        batch_size=config['Federated'].getint('batch_size')
    )

    for epoch in range(config['Federated'].getint('local_epochs')):
        total_loss = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if config['DEFAULT'].getboolean('verbose'):
            print(f"Participant Training"
                f" - Epoch: {epoch+1}/{config['Federated'].getint('local_epochs')}, "
                f"Loss: {total_loss/len(data_loader)}")

    return local_model.state_dict()

def train_poison_model(attacker_id, global_state_dict, data_handler, device, config, test_loader, poison_test_loader):
    """
    Trains a poison model targetting model replacement of the federated learning setup.

    """
    if config['DEFAULT'].getboolean('verbose'):
        print(f"Attacker ID: {attacker_id}")

    poison_model = resnet18().to(device)
    poison_model.load_state_dict(global_state_dict)

    poison_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(poison_model.parameters(), lr=config['Poison'].getfloat('local_lr'))
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          config['Poison'].getint('lr_step_size'),
                                          gamma=config['Poison'].getfloat('lr_gamma'),
                                          verbose = config['DEFAULT'].getboolean('verbose'))

    data_loader = data_handler.get_poison_dataloader(
        attacker_id,
        config['Poison'].getint('target_idx'),
        batch_size=config['Poison'].getint('batch_size'),
        poison_per_batch=config['Poison'].getint('poison_per_batch')
    )

    for epoch in range(config['Poison'].getint('local_epochs')):
        total_loss = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = poison_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Only step scheduler every epoch
        scheduler.step()
        if config['DEFAULT'].getboolean('verbose'):
            print(f"Poison Training"
                f" - Epoch: {epoch+1}/{config['Poison'].getint('local_epochs')}, "
                f"Loss: {total_loss/len(data_loader)}")

    # Evaluate the global model on main task
    accuracy, avg_loss = evaluate_model(poison_model, test_loader, device)
    backdoor_accuracy = evaluate_backdoor(poison_model,
                                        poison_test_loader,
                                        config['Poison'].getint('target_idx'),
                                        device)
    print(f"Poison Model: Test Accuracy: {accuracy:<5.2%} | Backdoor Accuracy: {backdoor_accuracy:<5.2%} | Loss: {avg_loss:<8.8}")

    torch.save(poison_model.state_dict(), "poison_model_state_dict.pt")
    # Update state dict for model replacement
    clip = config['Federated'].getint('num_participants') / config['Federated'].getfloat('global_lr')
    for key, value in poison_model.state_dict().items():
        global_value = global_state_dict[key]
        poison_model.state_dict()[key].copy_(global_value + clip*(value-global_value))
    return poison_model.state_dict()


def plot_history(history, title, savepath=None):
    """
    Helper method to plot training history.
    """
    fig = plt.gcf()
    ax = plt.gca()
    plt.cla()
    ax_loss = plt.twinx(ax)
    for i, metric in enumerate(history):
        if '_acc' in metric:
            ax.plot(history[metric], f"C{i}", label=metric)
        else:
            ax_loss.plot(history[metric],f"C{i}",label=metric)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax_loss.set_ylabel("")
    ax.set_ylim([0,1])
    ax.set_title(title)
    ax.legend(loc = 'upper left')
    ax_loss.legend(loc='upper right')
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)

def plot_federated_history(history, title, savepath=None):
    """
    Helper method to plot training history for federated learning.
    TODO
    - Add support for plotting when attack happens, and backdoor task accuracy
    """
    fig = plt.gcf()
    ax = plt.gca()
    plt.cla()
    ax_loss = plt.twinx(ax)
    for i, metric in enumerate(history):
        if '_acc' in metric:
            ax.plot(history[metric], f"C{i}", label=metric)
        else:
            ax_loss.plot(history[metric],f"C{i}",label=metric)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax_loss.set_ylabel("")
    ax.set_ylim([0,1])
    ax.set_title(title)
    ax.legend(loc = 'upper left')
    ax_loss.legend(loc='upper right')
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)

def pretrain_global_model(model, data_handler, device, config): #pylint: disable=too-many-locals
    """
    Pretrains the global model on the entire CIFAR10 training dataset.

    Args:
        model (torch.nn.Module): The global model to be pretrained.
        data_handler (DataHandler): The DataHandler object providing access to the CIFAR10 dataset.
        device (torch.device): The device (CPU or GPU) to train the model on.
        config (dict): A configuration dictionary containing training parameters such as
            'pretrain_lr' for the pretraining learning rate, 'batch_size', and 'pretrain_epochs'.

    Returns:
        The pretrained model.
    """
    print(f"Pre-training Global Model for {config['Pretraining'].getint('epochs')} epochs")

    # Set the model to training mode
    model.train()

    # Create history to track global model performance.
    history = {"global_model_loss":[], "global_model_acc":[]}
    # Setup figure formatting
    fig, _ = plt.subplots()
    fig.set_size_inches(12,6)

    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['Pretraining'].getfloat('lr'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)

    # Get the DataLoader for the full training dataset
    train_loader = data_handler.get_global_train_dataloader(
        batch_size=config['Pretraining'].getint('batch_size'),
        shuffle=True
    )
    test_loader = data_handler.get_test_dataloader(
        batch_size=config['Pretraining'].getint('batch_size'),
        shuffle=True
    )

    # Track best model
    best_accuracy = 0
    best_model = None
    # Avoid saving the model if best model hasn't changed
    best_model_saved = False

    # Training loop
    for epoch in range(config['Pretraining'].getint('epochs')):
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate the global model
        accuracy,_ = evaluate_model(model, test_loader, device)

        # Update the scheduler
        scheduler.step(accuracy)

        # Update best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            best_model_saved = False

        history["global_model_acc"].append(accuracy)
        history["global_model_loss"].append(total_loss/len(train_loader))

        # Print statistics
        print(f"Pretrain Epoch: {epoch+1}/{config['Pretraining'].getint('epochs')}, \
              Loss: {total_loss/len(train_loader)}, \
              Test Acc: {accuracy}")

        # Save global model and history
        if (epoch+1) % config['Pretraining'].getint('save_interval') == 0 or \
            (epoch + 1) == config['Pretraining'].getint('epochs'):
            if not best_model_saved:
                save_path = "./global_model_pretrain_best.pt"
                torch.save(best_model, save_path)
                best_model_saved = True
                print(f"Saved global model to {save_path}")

            #Plot history and save
            plot_history(history,
                         "Global Model Pretrain History",
                         "global_model_pretrain_history.png")

    plt.show()
    return model

def train(config_path): #pylint: disable=too-many-locals
    """
    Main training function for federated learning setup.

    Args:
        config_path (str): The path to the training configuration file.
    """
    config = load_config(config_path)
    data_handler = DataHandler(config_path)

    # Create test DataLoder for global evaluation
    test_loader = data_handler.get_test_dataloader(
        batch_size=config['Federated'].getint('batch_size')
    )
    # Create backdoor test Dataloader
    poison_test_loader = data_handler.get_test_poison_dataloader(batch_size=config['Poison'].getint('batch_size'),
                                                                 num_samples=config['Poison'].getint('num_eval'))

    # Create history to track global model performance.
    history = {"global_acc":[], "global_backdoor_acc":[], "global_loss":[], "local_updates_sum":[]}
    # Setup figure and axis formatting
    fig, ax = plt.subplots()
    fig.set_size_inches(12,6)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_title("Global Model Test Accuracy")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    global_model = resnet18().to(device)

    # Pretrain Global Model
    if config['Pretraining'].getboolean("pretrain"):
        global_model = pretrain_global_model(global_model, data_handler, device, config)

    global_state_dict = global_model.state_dict()

    # Which federated round to launch attack
    poison_round = config['Poison'].getint('poison_round')

    # Number of parallel local trainings
    parallel_streams = config['Federated'].getint('parallel_streams')
    # Check if we need to load from checkpoint
    if config['Federated']["load_from_checkpoint"]:
        print(f"Resuming Training with {config['Federated']['load_from_checkpoint']}")
        global_model.load_state_dict(torch.load(config['Federated']["load_from_checkpoint"],
                                                map_location=device))

    for federated_round in range(config['Federated'].getint('num_rounds')):
        attacker=None
        selected_participants = np.random.choice(
            range(config['Federated'].getint('num_participants')),
            size=config['Federated'].getint('num_selected'),
            replace=False)

        if (federated_round+1) == poison_round:
            # Sample the attacker index from participants, without replacement
            attacker_idx = np.random.randint(0, len(selected_participants))
            attacker = selected_participants[attacker_idx]
            selected_participants = np.delete(selected_participants, attacker_idx)

        print(f"Round {federated_round+1}/{config['Federated'].getint('num_rounds')}:\
               Selected Participants: {selected_participants}")
        if attacker:
            print(f"Selected Attacker: {attacker}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_streams) as executor:
            # Submit all selected_participants, these don't include the attacker.
            futures = [executor.submit(train_local_model,
                                       participant_id,
                                       global_state_dict,
                                       data_handler,
                                       device,
                                       config) for participant_id in selected_participants]

            local_state_dicts = [future.result() for future in\
                                  concurrent.futures.as_completed(futures)]
        
        # Compute local updates sum withotu attacker dict.
        local_updates_sum, _ = local_model_update_sum(global_model, local_state_dicts, device)
        history["local_updates_sum"].append(local_updates_sum)

        if attacker:
            # Train attacker after benign participants
            attacker_state_dict = train_poison_model(attacker,
                                                 global_state_dict,
                                                 data_handler,
                                                 device,
                                                 config,
                                                 test_loader,
                                                 poison_test_loader)

            local_state_dicts.append(attacker_state_dict)

        # Aggregate local models' updates into the global model
        global_model = aggregate_models(global_model,
                                        local_state_dicts,
                                        config['Federated'].getfloat('global_lr'),
                                        config['Federated'].getfloat('num_selected'),
                                        device)

        # Evaluate the global model on main task
        accuracy, avg_loss = evaluate_model(global_model, test_loader, device)
        # Evaluate the global model on backdoor task
        backdoor_accuracy = evaluate_backdoor(global_model,
                                              poison_test_loader,
                                              config['Poison'].getint('target_idx'),
                                              device)

        print(f"Global Model: Test Accuracy: {accuracy:<5.2%} | Backdoor Accuracy: {backdoor_accuracy:<5.2%} | Loss: {avg_loss:<8.8}")
        history["global_acc"].append(accuracy)
        history["global_loss"].append(avg_loss)
        history["global_backdoor_acc"].append(backdoor_accuracy)

        # Plot confusion matrix for test set
        # plot_confusion_matrix(global_model,
        #                       test_loader,
        #                       device,
        #                       data_handler.dataset.classes,
        #                       f"global_test_cm_{federated_round+1}.png")

        # Plot confusion matrix for backdoor set
        # plot_confusion_matrix(global_model,
        #                         poison_test_loader,
        #                         device,
        #                         data_handler.dataset.classes,
        #                         f"global_backdoor_cm_{federated_round+1}.png")

        # Save global model right after attack
        if (federated_round+1) % config['Poison'].getint('poison_round') == 0:
            save_path = f"./global_model_attacker_round_{federated_round + 1}.pt"
            torch.save(global_model.state_dict(), save_path)
            print(f"Saved global model to {save_path}")
        # Save global model and history
        if (federated_round+1) % config['Federated'].getint('save_interval') == 0 or \
            (federated_round + 1) == config['Federated'].getint('num_rounds'):
            save_path = f"./global_model_round_{federated_round + 1}.pt"
            torch.save(global_model.state_dict(), save_path)
            print(f"Saved global model to {save_path}")

            #Plot history and save
            plot_federated_history(history,
                         "Global Model Test Accuracy",
                         "global_model_acc.png")

if __name__ == "__main__":
    train('config.ini')
