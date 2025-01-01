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

def partitioning(distribution: np.ndarray, dataset: Dataset) -> dict[int, list[int]]:
    targets = dataset.targets
    areas = distribution.shape[0]
    targets_cardinality = distribution.shape[1]
    class_counts = torch.bincount(targets)
    partitions = {}
    for area in range(areas):
        area_distribution = distribution[area, :]
        elements_per_class = torch.tensor(area_distribution) * class_counts
        elements_per_class = torch.floor(elements_per_class).to(torch.int)
        selected_indices = []
        for label in range(targets_cardinality):
            target_indices = torch.where(targets == label)[0]
            selected_count = min(len(target_indices), elements_per_class[label].item())
            if selected_count > 0:
                selected_indices.extend(target_indices[:selected_count].tolist())
        partitions[area] = selected_indices
    return partitions

def test_model(model, dataset):
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
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