from utils.FedUtils import *

class ScaffoldServer:

    def __init__(self, dataset):
        self.dataset = dataset
        self.clients_data = {}
        self._model = initialize_model(dataset)
        self._control_state = initialize_control_state(self._model, dataset)


    def aggregate(self):
        pass


    def receive_client_update(self):
        pass


    @property
    def model(self):
        return self._model


    @property
    def control_state(self):
        return self._control_state
    