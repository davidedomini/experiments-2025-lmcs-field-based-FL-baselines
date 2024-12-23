from src.utils.FedUtils import initialize_control_state, initialize_model
from torch.utils.data import DataLoader
from torch import nn
import torch
import copy

class ScaffoldClient:

    def __init__(self, experiment, batch_size, epochs):
        self.experiment = experiment
        self.model = initialize_model(experiment)
        self.client_control_state = initialize_control_state(self.model)
        self.global_model_state = {}
        self.server_control_state = {}
        self.batch_size = batch_size
        # TODO - add data
        self.training_set = None
        self.validation_set = None
        self.lr = 0.001
        self.weight_decay=1e-4
        self.epochs = epochs

    def train(self):

        train_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        global_state_dict = copy.deepcopy(self.model.state_dict())
        scv_state = self.server_control_state.state_dict() # TODO - fix it
        ccv_state = self.client_control_state.state_dict()
        tau = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = nn.CrossEntropyLoss()
        losses = []

        for epoch in range(self.epochs):
            for step, (images, labels) in enumerate(train_loader):
                with torch.enable_grad():
                    self.model.train()
                    outputs = self.model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tau = tau + 1
                    losses.append(loss.item())
                    state_dict = self.model.state_dict()
                    for key in state_dict:
                        state_dict[key] = state_dict[key] - self.lr * (scv_state[key] - ccv_state[key])
                    self.model.load_state_dict(state_dict)

    def notify_updates(self, global_model_state, server_control_state):
        self.global_model_state = global_model_state
        self.server_control_state = server_control_state


