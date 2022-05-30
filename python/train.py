import torch
import itertools

from torch.utils.data import DataLoader
from utils.training_utils import ExperienceLearningDataset
from utils.models import Classifier
from utils import training_utils


def train_classifier():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the training data

    training_data, idx_to_action_mapping = training_utils.parse_txt_to_tensor(
        file_path="../runs/experience-learning.txt", device=device)

    print(idx_to_action_mapping)

    # Create data loader
    pinball_dataset = ExperienceLearningDataset(training_data)
    data_loader = torch.utils.data.DataLoader(pinball_dataset, batch_size=64, shuffle=True)

    # Create model
    model = Classifier(
        input_size=training_data.shape[1] - 1,
        output_size=len(idx_to_action_mapping),
        hidden_layers=3
    ).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Print model summary
    model.summary()

    # Train the model
    loss_history = []
    total_step = len(data_loader)
    for epoch in range(100):
        for i, (x, y) in enumerate(data_loader):
            # Send data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 100, i + 1, total_step, loss.item()))
                loss_history.append(loss.item())

    # Save the model
    model.save("./saved_models/experience1.pt")

    # Plot the loss history
    training_utils.plot_loss_history(loss_history)


def train_dqn():



if __name__ == "__main__":
    train_dqn()
