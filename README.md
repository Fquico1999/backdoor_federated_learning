# Backdoor Federated Learning
Implementation of ["How to Backdoor Federated Learning"](https://arxiv.org/abs/1807.00459) paper.


-`non_iid_vs_iid.ipynb`: Motivation of the importance of non-i.i.d data distributions in federated learning.


## Usage
All experiment parameters are set with the `config.ini` file.
The experimental setup attempts to stay true to the paper as much as possible, with some key differences, mostly for efficiency purposes.

Attacking the federated learning setup works best when the global model has trained to some form of convergence, in fact, one of the requirements for 
model replacement to work assumes this to be true. For that to occur, the global model must be trained for some time. Rather than waiting for 10,000 epochs
of federated learning training, which is too lengthy, I've added the ability to first pretrain the global model directly on the CIFAR10 dataset for a 
set number of epochs. Note that this still requires the federated learning to run for some warmup rounds afterwards. The parameters for this are found under
the `[Pretraining]` section of the config file.

The other notable change is beign able to train the local benign models concurrently. This is set with `parallel_streams` in the config file.

There are some other slight changes. For instance, I was unable to achieve the 90%+ accuracy on the main task even with global model pretraining, despite applying 
data augmentation, using the `Adam` optimizer instead of `SGD` and using adaptive learning rate scheduling. Additionally, since the paper's implementation of 
ResNet18 is slightly different from the torchvision one, I cannot import Imagenet weights as a starting point. 

Launching the experiment is done with:
```python
from train import train

train("/path/to/config")
```
