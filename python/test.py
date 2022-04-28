import torch
import pandas as pd

from utils.old_models import Classifier
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


model = Classifier(input_size=6, output_size=3, hidden_layers=3)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

model.model.load_state_dict(torch.load("saved_models/experience.pt"))
model.eval()