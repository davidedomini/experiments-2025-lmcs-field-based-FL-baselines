import torch
import random
import numpy as np
import pandas as pd
import utils.FedUtils as utils
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms
from client.FedAvgClient import FedAvgClient
from server.FedAvgServer import FedAvgServer
from client.ScaffoldClient import ScaffoldClient
from server.ScaffoldServer import ScaffoldServer

class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset_name, n_clients, data_folder):
        self.batch_size = 32
        self.local_epochs = 2
        self.dataset_name = dataset_name
        self.complete_dataset, self.training_data, self.validation_data = self.initialize_data()
        self.partitioning = partitioning
        self.algorithm = algorithm
        self.areas = areas
        self.n_clients = n_clients
        self.export_path = f'{data_folder}/algorithm-{self.algorithm}_dataset-{dataset_name}_partitioning-{self.partitioning}_areas-{self.areas}_clients-{self.n_clients}'
        self.simulation_data = pd.DataFrame(columns=['Round','TrainingLoss', 'ValidationLoss', 'ValidationAccuracy'])
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def start(self, global_rounds):
        for r in range(global_rounds):
            print(f'Starting global round {r}')
            self.notify_clients()
            training_loss = self.clients_update()
            self.notify_server()
            self.server_update()
            validation_loss, validation_accuracy = self.test_global_model()
            self.export_data(r, training_loss, validation_loss, validation_accuracy)
        self.test_global_model(False)
        self.save_data()

    def initialize_clients(self):
        client_data_mapping = self.map_client_to_data()
        if self.algorithm == 'fedavg':
            return [FedAvgClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'scaffold':
            return [ScaffoldClient(self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
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

    def clients_update(self):
        training_losses = []
        for client in self.clients:
            train_loss = client.train()
            training_losses.append(train_loss)
        average_training_loss = sum(training_losses) / len(training_losses)
        return average_training_loss

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

    def initialize_data(self):
        d = self.get_dataset()
        dataset_size = len(d)
        training_size = int(dataset_size * 0.8)
        validation_size = dataset_size - training_size
        training_data, validation_data = random_split(d, [training_size, validation_size])
        return d, training_data, validation_data

    def map_client_to_data(self) -> dict[int, Subset]:
        clients_split = np.array_split(list(range(self.n_clients)), self.areas)
        mapping_area_clients = { areaId: list(clients_split[areaId]) for areaId in range(self.areas) }
        if self.partitioning.lower() == 'hard':
            mapping = utils.hard_non_iid_mapping(self.areas, len(self.complete_dataset.classes))
        elif self.partitioning.lower() == 'iid':
            mapping = utils.iid_mapping(self.areas, len(self.complete_dataset.classes))
        else:
            raise Exception(f'Partitioning {self.partitioning} not supported! Please check :)')
        distribution_per_area = utils.partitioning(mapping, self.training_data)#, self.complete_dataset.targets[self.training_data.indices])
        mapping_client_data = {}
        for area in mapping_area_clients.keys():
            clients = mapping_area_clients[area]
            indexes = distribution_per_area[area]
            random.shuffle(indexes)
            split = np.array_split(indexes, len(clients))
            for i, c in enumerate(clients):
                mapping_client_data[c] = Subset(self.complete_dataset, split[i])
        return mapping_client_data

    def test_global_model(self, validation = True):
        model = self.server.model
        if validation:
            dataset = self.validation_data
        else:
            dataset = self.get_dataset(False)
        loss, accuracy = utils.test_model(model, dataset, self.batch_size)
        print(f'Test accuracy {accuracy} test loss {loss}')
        return loss, accuracy

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
