import torch
from torch import nn
from utils.FedUtils import initialize_model
from torch.utils.data import DataLoader, random_split

class FedAvgClient:

    def __init__(self, mid, dataset_name, dataset, batch_size, epochs):
        self.mid = mid
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self._model = initialize_model(dataset_name)
        dataset_size = len(self.dataset)
        training_size = int(dataset_size * 0.8)
        validation_size = dataset_size - training_size
        self.training_set, self.validation_set = random_split(self.dataset, [training_size, validation_size])
        self.lr = 0.001
        self.weight_decay=1e-4

    def train(self):
        labels = [self.training_set[idx][1] for idx in range(len(self.training_set))]
        print(f'Client {self.mid} --> training set size {len(self.training_set)} classes {set(labels)}')
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