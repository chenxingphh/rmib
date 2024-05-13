import os
from pprint import pprint
from .model import Model
from .interface import Interface
from .utils.loader import load_data


class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file

    def evaluate(self):
        data = load_data(*os.path.split(self.data_file))
        model, checkpoint = Model.load(self.model_path)
        args = checkpoint['args']
        interface = Interface(args)
        batches = interface.pre_process(data, training=False)
        _, stats = model.evaluate(batches)
        pprint(stats)
