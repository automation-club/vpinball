import torch
import pandas as pd
import time

from torch.utils.data import Dataset


# Parse the fucking file
def parse_file(file_path):
    columns = ["_", "X", "Y", "Z", "VelX", "VelY", "VelZ", "Action"]
    df = pd.read_table(file_path, sep=",", names=columns)
    df.drop(columns=["_"], inplace=True)
    print(df["Action"].astype('category').cat.codes)
    print(dict(enumerate(df["Action"].astype('category').cat.categories)))
    torch_tensor = torch.tensor(df.values)
    idx_to_action = {idx: action for idx, action in enumerate(df["Action"].unique())}
    print(idx_to_action)

    return torch_tensor


x = parse_file("../runs/experience-learning.txt")
