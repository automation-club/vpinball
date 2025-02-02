import torch
import pandas as pd
import copy
import numpy as np

from torch import nn
from collections import deque


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

class DQN(nn.Module):
    """

    """
    def __init__(self, input_size, output_size):
        self.main = self._create_agent(input_size, output_size)
        self.target = copy.deepcopy(self.main)
        self.replay_memory = deque(maxlen=100000)  # [state, action, reward, state', game_over]
        self.steps_since_target_updated = 0
        self.LEARNING_RATE = 0.5
        self.BATCH_SIZE = 32
        self.DISCOUNT_FACTOR = 0.5
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.Adam()

    def _create_agent(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
        )

    def train_agent(self):
        if len(self.replay_memory) < 1000:
            return
        batch = np.random.choice(self.replay_memory, size=self.BATCH_SIZE)
        batch = torch.tensor(batch)

        states = batch[:, 0]
        q_values = self.model(states)

        future_states = batch[:, 3]
        future_q_values = self.target(future_states)

        temporal_difference = batch[:, 2] + self.DISCOUNT_FACTOR*torch.max(future_q_values, dim=1)*batch[:, 4]

        for index, (action) in enumerate(batch[:,4]):
            q_values[index, action] = (1-self.LEARNING_RATE)*q_values[index, action] + self.LEARNING_RATE*temporal_difference[index]





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
        save(path="experience.pt")
            Saves the model to a file
        load(path="experience.pt")
            Loads the model from a file

        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.model = self._create_classifier()
        self.action_space = self.assign_idx_to_class("../runs/experience-learning.txt")

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

    @staticmethod
    def assign_idx_to_class(label_file_path):
        """
        Assigns a class name to a class index

        Parameters
        ----------
        idx : int
            Class index
        label_file_path : str
            File path to the label file

        Returns
        -------
        dict
            Dictionary mapping class index to class name
        """
        names = ["Info Type", "X", "Y", "Z", "VelX", "VelY", "VelZ", "Action"]
        df = pd.read_table(label_file_path, sep=",", names=names)
        # Map actions to integers and save mapping
        df["Action"] = df["Action"].astype("category")
        idx_to_action = dict(enumerate(df["Action"].cat.categories))

        return idx_to_action

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
        layers.append(nn.Linear((2 ** self.hidden_layers) * first_hidden_layer_nodes, self.output_size))

        return nn.Sequential(*layers)

    def summary(self):
        """
        Prints a summary of the model

        Returns
        -------
        None
        """
        print(self.model)