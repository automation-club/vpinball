import torch
import pandas as pd
import time

from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, ):

start = time.time()
columns=["_", "X", "Y", "Z", "VelX", "VelY", "VelZ", "Action"]
df = pd.read_table("./runs/experience-learning.txt", sep=",", names=columns)

print(df.dtypes)
start = time.time()
print(df.head())
end = time.time()
print(end-start)


print(torch_dataset)