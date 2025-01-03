from utils.FedUtils import *

class ScaffoldServer:

    def __init__(self, dataset):
        self.dataset = dataset
        self.clients_data = {}
        self.old_client_control_state = {}
        self._model = initialize_model(dataset)
        self._control_state = initialize_control_state(dataset)

    def aggregate(self):
        n = len(self.clients_data.keys())

        for key in self._model:
            # First point of Eq (5) in Scaffold paper
            self._model[key] = self._model[key] + ((1/n) * self.sum_models_delta(key))

        for key in self._control_state:
            # Second point of Eq (5) in Scaffold paper
            self._control_state[key] = self._control_state[key] + ((1/n) * self.sum_clients_delta(key))

        # Store c_i
        self.old_client_control_state = { k: v['client_control_state'] for k, v in self.clients_data.items() }

    def receive_client_update(self, client_data):
        self.clients_data = client_data
        if not self.old_client_control_state:
            self.old_client_control_state = { k: initialize_control_state(self.dataset) for k, v in client_data.items() }

    @property
    def model(self):
        return self._model

    @property
    def control_state(self):
        return self._control_state

    def sum_clients_delta(self, key):
        acc = None
        for client in self.clients_data.keys():
            delta = self.clients_data[client]['client_control_state'][key] - self.old_client_control_state[client][key]
            if acc is None:
                acc = delta
            else:
                acc += delta
        return acc

    def sum_models_delta(self, key):
        acc = None
        global_model_dict = self._model.state_dict()
        for client in self.clients_data.keys():
            client_model_dict = self.clients_data[client]['model'].state_dict()
            delta = client_model_dict[key] - global_model_dict[key]
            if acc is None:
                acc = delta
            else:
                acc += delta
        return acc