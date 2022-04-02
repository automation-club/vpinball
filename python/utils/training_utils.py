"""
Collection of methods to carry out model training.

Classes
-------


Methods
-------
parse_txt_to_tensor(file_path=", device)
    Parse a text pinball data file to a tensor.
"""

import torch
import pandas as pd

from torch.utils.data import Dataset


def parse_txt_to_tensor(file_path, device):
    """
    Reads a text file and returns a tensor of shape (n_samples, n_features+labels)

    Parameters
    ----------
    file_path : str
        Path to the text file
    device : torch.device
        Device to load the data to

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_samples, n_features)
    dict
        Dictionary mapping of index to output label
    """
    # Read file to pandas dataframe
    names = ["Info Type", "X", "Y", "Z", "VelX", "VelY", "VelZ", "Action"]
    df = pd.read_table(file_path, sep=",", names=names)
    # Drop useless column
    df = df.drop(columns=["Info Type"])
    # Map actions to integers and save mapping
    df["Action"] = df["Action"].astype("category")
    idx_to_action = dict(enumerate(df["Action"].cat.categories))
    df["Action"] = df["Action"].cat.codes

    # Convert to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32, device=device)

    return tensor, idx_to_action


class PinballDataset(Dataset):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (n_samples, n_features+labels)
        """
        self.data = data

    def __len__(self):
        """
        Returns
        -------
        int
            Number of samples in the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the sample to return

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_features,)
        torch.Tensor
            Tensor of shape (labels,)
        """
        return self.data[idx, :-1], self.data[idx, -1]
