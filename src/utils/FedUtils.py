from src.models.MNIST import NNMnist
import torch.nn as nn
import torch

def initialize_model(name):
    if name == 'MNIST':
        return NNMnist()
    else:
        raise Exception(f'Model {name} not implemented! Please check :)')


def initialize_control_state(model, experiment):
    control_state = initialize_model(experiment)
    for param in control_state.parameters():
        nn.init.constant_(param, 0.0)
    return control_state