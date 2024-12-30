import torch
import numpy as np
import torch.nn as nn
from models.MNIST import NNMnist

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

def hard_non_iid_mapping(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1
    return distribution