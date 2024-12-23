from server.ScaffoldServer import ScaffoldServer


class Simulator:

    def __init__(self, experiment, algorithm, n_clients, areas):
        self.experiment = experiment
        self.algorithm = algorithm
        self.n_clients = n_clients
        self.areas = areas
        self.clients = []
        self.server = self.initialize_server(self.algorithm)


    def seed_everything(self, seed):
        pass


    def start(self):
        for name, param in self.server.control_state.named_parameters():
            print(f'{name} --> {param}')


    def initialize_server(self, algorithm):
        if algorithm == 'scaffold':
            return ScaffoldServer(self.experiment)
        else:
            raise Exception('Algorithm not supported! Please check :)')

