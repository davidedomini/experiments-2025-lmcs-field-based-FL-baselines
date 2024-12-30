from client.FedAvgClient import FedAvgClient
from client.ScaffoldClient import ScaffoldClient
from server.ScaffoldServer import ScaffoldServer

class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset, n_clients):
        self.partitioning = partitioning
        self.algorithm = algorithm
        self.areas = areas
        self.dataset = dataset
        self.n_clients = n_clients
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()

    def seed_everything(self, seed):
        pass

    def start(self, global_rounds):
        for round in range(global_rounds):
            print(f'Starting global round {round}')
            self.notify_clients()
            self.clients_update()
            self.notify_server()
            self.server_update()

    def initialize_clients(self):
        if self.algorithm == 'fedavg':
            return [FedAvgClient(self.dataset, 32, 2) for _ in range(self.n_clients)]
        elif self.algorithm == 'scaffold':
            return [ScaffoldClient(self.dataset, 32, 2) for _ in range(self.n_clients)]
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def initialize_server(self):
        if self.algorithm == 'scaffold':
            return ScaffoldServer(self.dataset)
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def notify_clients(self):
        for client in self.clients:
            if self.algorithm == 'fedavg':
                client.notify_updates(self.server.model)
            elif self.algorithm == 'scaffold':
                client.notify_updates(self.server.model, self.server.control_state)

    def clients_update(self):
        for client in self.clients:
            client.train()

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

    def export_data(self):
        """
        Registers new data, you can use it at each time stamp to store training and evaluation data.
        Important: it does not save the data on a file, you must call the specific method at the end of the simulation!
        :return: Nothing
        """
        pass

    def save_data(self):
        """
        Saves the registered data on a file.
        :return: Nothing
        """
        pass