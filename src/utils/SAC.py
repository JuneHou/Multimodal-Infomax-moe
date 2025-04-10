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

def compute_log_variance_from_hidden(hidden_tensor, predicted_mean):
    """
    Computes log variance using predicted mean instead of mean over hidden dims.
    Args:
        hidden_tensor: [N, D] tensor
        predicted_mean: [N, 1] tensor
    """
    var = ((hidden_tensor.mean(dim=1, keepdim=True) - predicted_mean) ** 2) + \
          ((hidden_tensor - hidden_tensor.mean(dim=1, keepdim=True)) ** 2).mean(dim=1, keepdim=True)
    log_var = torch.log(var + 1e-6)
    return log_var

def train_variance_estimator(model, data_loader, device, epochs=10, lr=0.001):
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for hidden, pred_mean in data_loader:
            hidden = hidden.to(device)
            pred_mean = pred_mean.to(device)

            optimizer.zero_grad()
            log_sigma_pred = model(hidden)
            log_sigma_true = compute_log_variance_from_hidden(hidden, pred_mean)

            loss = criterion(log_sigma_pred, log_sigma_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Variance Estimator] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model

