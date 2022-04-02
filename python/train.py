import torch
import itertools

from torch.utils.data import DataLoader
from utils.training_utils import PinballDataset
from utils.models import Classifier
from utils import training_utils


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the training data

    training_data, idx_to_action_mapping = training_utils.parse_txt_to_tensor(file_path="../runs/experience-learning.txt", device=device)

    # Create data loader
    pinball_dataset = PinballDataset(training_data)
    data_loader = torch.utils.data.DataLoader(pinball_dataset, batch_size=2, shuffle=True)

    # Testing data loader
    for x, y in itertools.islice(data_loader, 5):
        print(x, y.view(-1, 1))

