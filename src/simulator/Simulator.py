import torch
import random
import numpy as np
import pandas as pd
import utils.FedUtils as utils
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from client.FedAvgClient import FedAvgClient
from server.FedAvgServer import FedAvgServer
from client.ScaffoldClient import ScaffoldClient
from server.ScaffoldServer import ScaffoldServer

class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset_name, n_clients, data_folder):
        self.partitioning = partitioning
        self.algorithm = algorithm
        self.areas = areas
        self.dataset_name = dataset_name
        self.n_clients = n_clients
        self.export_path = f'{data_folder}/algorithm-{self.algorithm}_dataset-{dataset_name}_partitioning-{self.partitioning}_areas-{self.areas}_clients-{self.n_clients}'
        self.simulation_data = pd.DataFrame(columns=['Round','TrainingLoss', 'ValidationLoss', 'ValidationAccuracy'])
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    def start(self, global_rounds):
        for r in range(global_rounds):
            print(f'Starting global round {r}')
            self.notify_clients()
            self.clients_update(r)
            self.notify_server()
            self.server_update()
        self.save_data()

    def initialize_clients(self):
        client_data_mapping = self.map_client_to_data()
        if self.algorithm == 'fedavg':
            return [FedAvgClient(self.dataset_name, client_data_mapping[index], 32, 2) for index in range(self.n_clients)]
        elif self.algorithm == 'scaffold':
            return [ScaffoldClient(self.dataset_name, client_data_mapping[index],32, 2) for index in range(self.n_clients)]
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def initialize_server(self):
        if self.algorithm == 'fedavg':
            return FedAvgServer(self.dataset_name)
        elif self.algorithm == 'scaffold':
            return ScaffoldServer(self.dataset_name)
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def notify_clients(self):
        for client in self.clients:
            if self.algorithm == 'fedavg':
                client.notify_updates(self.server.model)
            elif self.algorithm == 'scaffold':
                client.notify_updates(self.server.model, self.server.control_state)

    def clients_update(self, global_round):
        training_losses = []
        evaluation_losses = []
        evaluation_accuracies = []
        for client in self.clients:
            train_loss = client.train()
            evaluation_loss, evaluation_accuracy = client.evaluate()
            training_losses.append(train_loss)
            evaluation_losses.append(evaluation_loss)
            evaluation_accuracies.append(evaluation_accuracy)
        average_training_loss = sum(training_losses) / len(training_losses)
        average_evaluation_loss = sum(evaluation_losses) / len(evaluation_losses)
        average_evaluation_accuracy = sum(evaluation_accuracies) / len(evaluation_accuracies)
        self.export_data(global_round, average_training_loss, average_evaluation_loss, average_evaluation_accuracy)

    def notify_server(self):
        client_data = {}
        for index, client in enumerate(self.clients):
            if self.algorithm == 'fedavg':
                client_data[index] = client.model
            elif self.algorithm == 'scaffold':
                client_data[index] = { 'model': client.model, 'client_control_state': client.client_control_state }
        self.server.receive_client_update(client_data)

    def server_update(self):
        self.server.aggregate()

    def map_client_to_data(self) -> dict[int, Subset]:
        d = self.get_dataset()
        clients_split = np.array_split(list(range(self.n_clients)), self.areas)
        mapping_area_clients = { areaId: list(clients_split[areaId]) for areaId in range(self.areas) }
        if self.partitioning.lower() == 'hard':
            mapping = utils.hard_non_iid_mapping(self.areas, len(d.classes))
        else:
            raise Exception(f'Partitioning {self.partitioning} not supported! Please check :)')
        distribution_per_area = utils.partitioning(mapping, d)
        mapping_client_data = {}
        for area in mapping_area_clients.keys():
            clients = mapping_area_clients[area]
            indexes = distribution_per_area[area]
            split = np.array_split(indexes, len(clients))
            for i, c in enumerate(clients):
                mapping_client_data[c] = Subset(d, split[i])
        return mapping_client_data

    def get_dataset(self, train = True):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='dataset', train=train, download=True, transform=transform)
            return dataset
        else:
            raise Exception(f'Dataset {self.dataset_name} not supported! Please check :)')

    def export_data(self, global_round, training_loss, evaluation_loss, evaluation_accuracy):
        """
        Registers new data, you can use it at each time stamp to store training and evaluation data.
        Important: it does not save the data on a file, you must call the specific method at the end of the simulation!
        :return: Nothing
        """
        self.simulation_data = self.simulation_data._append(
            {'Round': global_round,'TrainingLoss': training_loss, 'ValidationLoss': evaluation_loss, 'ValidationAccuracy': evaluation_accuracy},
            ignore_index=True
        )

    def save_data(self):
        """
        Saves the registered data on a file.
        :return: Nothing
        """
        self.simulation_data.to_csv(self.export_path, index=False)
