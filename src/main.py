import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from simulator.Simulator import Simulator

def get_hyperparameters():
    hyperparams = os.environ[HYPERPARAMETERS_NAME]
    print(hyperparams)
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name, hyperparams

if __name__ == '__main__':

    HYPERPARAMETERS_NAME = 'LEARNING_HYPERPARAMETERS'
    experiment_name, hyperparameters = get_hyperparameters()

    print(f'{experiment_name} --> {hyperparameters}')

    # experiment = 'MNIST'
    # algorithm  = 'scaffold'
    # n_clients  = 10
    # areas      = 3
    #
    # simulator = Simulator(experiment, algorithm, n_clients, areas)
    #
    # simulator.start()