from server.ScaffoldServer import ScaffoldServer


class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset, n_clients):
        self.partitioning = partitioning
        self.algorithm = algorithm
        self.areas = areas
        self.dataset = dataset
        self.n_clients = n_clients
        self.clients = []
        self.server = self.initialize_server()


    def seed_everything(self, seed):
        pass


    def start(self):
        for name, param in self.server.control_state.named_parameters():
            print(f'{name} --> {param}')


    def initialize_server(self):
        if self.algorithm == 'scaffold':
            return ScaffoldServer(self.dataset)
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')


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