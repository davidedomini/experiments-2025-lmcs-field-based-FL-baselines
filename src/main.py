import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from simulator.Simulator import Simulator

def get_hyperparameters():
    """
    Fetches the hyperparameters from the docker compose config file
    :return: the experiment name and the hyperparameters (as a dictionary name -> values)
    """
    hyperparams = os.environ[HYPERPARAMETERS_NAME]
    print(hyperparams)
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name.lower(), hyperparams


def check_hyperparameters(hyperparams):
    """
    Checks if the hyperparameters fetched from the config file are valid
    :param hyperparams: all the hyperparameters fetched from the configuration file
    :raise: raises a ValueError if the hyperparameters are invalid
    """
    valid_hyperparams = ['partitioning', 'areas', 'seed', 'dataset', 'clients']
    for index, hp in enumerate(hyperparams.keys()):
        if hp != valid_hyperparams[index]:
            raise ValueError(f'''
                The hyperparameter {hp} is not valid! 
                Valid hyperparameters are: {valid_hyperparams} (they must be in this exact order)
            ''')


if __name__ == '__main__':

    HYPERPARAMETERS_NAME = 'LEARNING_HYPERPARAMETERS'
    data_dir = os.environ['DATA_DIR']
    experiment_name, hyperparameters = get_hyperparameters()
    check_hyperparameters(hyperparameters)
    all_experiments = list(product(*hyperparameters.values()))

    for partitioning, areas, seed, dataset, clients in all_experiments:
        print(partitioning, areas, seed, dataset, clients)
        simulator = Simulator(experiment_name, partitioning, areas, dataset, clients, data_dir)
        simulator.seed_everything(seed)
        simulator.start(50)
        break
