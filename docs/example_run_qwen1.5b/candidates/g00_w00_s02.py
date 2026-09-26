import torch
import numpy as np
from link_config import PILOT_MASK, PILOT_SYMBOLS, PILOT_OFDM_SYMBOLS, PILOT_SUBCARRIER_SPACING, FFT_SIZE, NUM_OFDM_SYMBOLS, SUBCARRIER_SPACING, CARRIER_FREQUENCY, OFDM_SYMBOL_DURATION
from sklearn.linear_model import Ridge
import sys
import os

def estimate_channel(y, no):
    if not 'sklearn' in sys.modules:
        print("Installing scikit-learn...")
        !pip install scikit-learn

    # Filter out the pilots
    train_mask = ~PILOT_MASK
    train_y = y[:, train_mask]
    train_no = no[train_mask]

    # Train the Ridge regressor
    model = Ridge(alpha=1.0)
    model.fit(train_y.view(-1, FFT_SIZE).numpy(), train_no.numpy())

    # Predict the channel on all REs
    y_pred = model.predict(y.view(-1, FFT_SIZE).numpy())
    channel_estimates = y_pred.reshape(y.shape)

    # Calculate the mean squared error
    mse = torch.mean((channel_estimates - y) ** 2, dim=(0, 2))
    return channel_estimates, mse.item()

if __name__ == "__main__":
    # Example usage
    # y = torch.randn(32, 14, 72)  # Replace with your actual input data
    # no = torch.randn(32)       # Replace with your actual noise variances
    # channel_estimates, mse = estimate_channel(y, no)
    # print(f"Mean Squared Error: {mse}")
