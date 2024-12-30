from utils.FedUtils import initialize_model
from torch.utils.data import DataLoader
from torch import nn
import torch

class FedAvgClient:

    def __init__(self, dataset, batch_size, epochs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self._model = initialize_model(dataset)
        # TODO - add data
        self.training_set = None
        self.validation_set = None
        self.lr = 0.001
        self.weight_decay=1e-4

    def train(self):
        train_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(self.epochs):
            batch_losses = []
            for step, (images, labels) in enumerate(train_loader):
                with torch.enable_grad():
                    self._model.train()
                    outputs = self._model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
            mean_epoch_loss = sum(batch_losses) / len(batch_losses)
            losses.append(mean_epoch_loss)
        return sum(losses) / len(losses)

    def evaluate(self):
        criterion = nn.NLLLoss()
        self._model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        data_loader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)
        for batch_index, (images, labels) in enumerate(data_loader):
            outputs = self._model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct / total
        return loss, accuracy

    def notify_updates(self, global_model):
        self._model.load_state_dict(global_model.state_dict())

    @property
    def model(self):
        return self._model