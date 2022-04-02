import torch

from torch import nn


# class Model:
#     def __init__(self, model_type, input_size, output_size, hidden_layers):
#         self.model = self._create_model(model_type, input_size, output_size, hidden_layers)
#
#     @staticmethod
#     def _create_model(model_type, input_size, output_size, hidden_layers):
#         if model_type == "dqn":
#             pass
#         elif model_type == "classification":
#             return Classifier(input_size, output_size, hidden_layers)
#         else:
#             print("Model type not supported")


class Classifier(nn.Module):
    """
    Classification model
    """
    def __init__(self, input_size, output_size, hidden_layers):
        """
        Parameters
        ----------
        input_size : int
            Number of input features
        output_size : int
            Number of output classes
        hidden_layers : int
            Number of hidden layers

        Methods
        -------
        forward(x)
            Forward pass of the model
        save(path="model.pt")
            Saves the model to a file
        load(path="model.pt")
            Loads the model from a file

        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.model = self._create_classifier()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.model(x)

    def save(self, path):
        """
        Saves the model to a file

        Parameters
        ----------
        path : str
            Path to save the model to
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the model from a file

        Parameters
        ----------
        path : str
            Path to load the model from
        """
        self.model.load_state_dict(torch.load(path))

    def _create_classifier(self):
        """
        Creates a Dense Linear Classifier model

        Returns
        -------
        torch.nn.Sequential
            Dense Linear Classifier model
        """

        first_hidden_layer_nodes = 64

        # Create input layer
        layers = [nn.Linear(self.input_size, first_hidden_layer_nodes)]

        # Add hidden layers
        for i in range(self.hidden_layers):
            layers.append(nn.Linear((2**i)*first_hidden_layer_nodes, (2**(i+1))*first_hidden_layer_nodes))
            layers.append(nn.LeakyReLU())

        # Add output layer
        layers.append(nn.Linear((2**(self.hidden_layers+1))*first_hidden_layer_nodes, self.output_size))

        return nn.Sequential(*layers)
