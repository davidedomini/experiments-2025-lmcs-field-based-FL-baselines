from src.simulator.Simulator import Simulator

if __name__ == '__main__':

    experiment = 'MNIST'
    algorithm  = 'scaffold'
    n_clients  = 10
    areas      = 3

    simulator = Simulator(experiment, algorithm, n_clients, areas)

    simulator.start()