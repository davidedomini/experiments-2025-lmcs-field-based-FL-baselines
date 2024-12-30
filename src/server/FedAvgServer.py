import copy
from utils.FedUtils import *

class FedAvgServer:

    def __init__(self, dataset):
        self.dataset = dataset
        self.clients_data = {}
        self._model = initialize_model(dataset)

    def aggregate(self):
        """
        Aggregates N models following the FedAvg algorithm.
        :return: Nothing
        """
        models = [m.state_dict() for m in self.clients_data.values()]
        w_avg = copy.deepcopy(models[0])

        for key in w_avg.keys():
            w_avg[key] = torch.mul(w_avg[key], 0.0)
        for key in w_avg.keys():
            for i in range(0, len(models)):
                w_avg[key] += models[i][key]
            w_avg[key] = torch.div(w_avg[key], len(models))
        self.model.load_state_dict(w_avg)

    def receive_client_update(self, client_data):
        self.clients_data = client_data

    @property
    def model(self):
        return self._model