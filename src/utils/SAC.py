import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class VarianceEstimator(nn.Module):
    """
    Neural network to estimate log variance using multimodal hidden representations.
    """
    def __init__(self, input_dim):
        super(VarianceEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output log variance
    
    def forward(self, hidden):
        x = torch.relu(self.fc1(hidden))
        x = torch.relu(self.fc2(x))
        log_sigma = self.fc3(x)  # Output log variance
        return log_sigma

        

def train_variance_estimator(model, data_loader, device, epochs=10, lr=0.001):
    """
    Trains the variance estimator on multimodal predictions.
    
    Args:
        model (VarianceEstimator): The neural network model.
        data_loader (torch.utils.data.DataLoader): Data loader for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): "cuda" or "cpu" depending on available hardware.

    Returns:
        VarianceEstimator: Trained model.
    """
    # model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for y_multi in data_loader:
            y_multi = y_multi[0].to(device)  # Ensure tensor is on the right device

            optimizer.zero_grad()
            log_sigma = model(y_multi)
            loss = criterion(log_sigma, torch.zeros_like(log_sigma))  # Encourage stable variance
            loss.backward()
            optimizer.step()

        print(f"[Variance Estimator] Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    print("Variance Estimator training complete.")