import torch

from torch import nn


class Model:
    def __init__(self, model_type, input_size, output_size, num_layers):
        self.model = self._create_model(model_type, input_size, output_size, num_layers)

    @staticmethod
    def _create_model(model_type, input_size, output_size, num_layers):
        if model_type == "dqn":
            pass
        elif model_type == "classification":
            return Classifier(input_size, output_size, num_layers)
        else:
            print("Model type not supported")


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        self.model = _create_classifier()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

    def _create_classifier(self):
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(self.input_size, self.output_size))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)