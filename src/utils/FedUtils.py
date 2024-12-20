from src.models.MNIST import NNMnist
import torch

def initialize_model(name):
    if name == 'MNIST':
        return NNMnist()
    else:
        raise Exception(f'Model {name} not implemented! Please check :)')


def initialize_control_state(model, experiment):
    control_state = initialize_model(experiment)
    for k, v in model.named_parameters():
        control_state[k] = torch.zeros_like(v)
    return control_state