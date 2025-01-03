import math
import random
import torch
import numpy as np
import torch.nn as nn
from models.MNIST import NNMnist
from torch.utils.data import Dataset, Subset, DataLoader

def initialize_model(name):
    if name == 'MNIST':
        return NNMnist()
    else:
        raise Exception(f'Model {name} not implemented! Please check :)')

def initialize_control_state(experiment):
    control_state = initialize_model(experiment)
    for param in control_state.parameters():
        nn.init.constant_(param, 0.0)
    return control_state.state_dict()

def hard_non_iid_mapping(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1 / len(elems)
    return distribution

def iid_mapping(areas: int, labels: int) -> np.ndarray:
    percentage = 1 / labels
    distribution = np.zeros((areas, labels))
    distribution.fill(percentage)
    return distribution

def partitioning(distribution: np.ndarray, data: Subset) -> dict[int, list[int]]:
    indices = data.indices
    targets = data.dataset.targets
    class_counts = torch.bincount(targets[indices])
    class_to_indices = {}
    for index in indices:
        c = targets[index].item()
        if c in class_to_indices:
            class_to_indices[c].append(index)
        else:
            class_to_indices[c] = [index]
    areas = distribution.shape[0]
    targets_cardinality = distribution.shape[1]
    max_examples_per_area = int(math.floor(len(indices) / areas))
    elements_per_class =  torch.floor(torch.tensor(distribution) * max_examples_per_area).to(torch.int)
    partitions = { a: [] for a in range(areas) }
    for area in range(areas):
        elements_per_class_in_area = elements_per_class[area, :].tolist()
        for c in range(targets_cardinality):
            elements = min(elements_per_class_in_area[c], class_counts[c].item())
            selected_indices = random.sample(class_to_indices[c], elements)
            partitions[area].extend(selected_indices)
    return partitions

def test_model(model, dataset, batch_size):
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return loss, accuracy