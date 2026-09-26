import torch
import numpy as np
from link_config import PILOT_MASK, PILOT_SYMBOLS, PILOT_OFDM_SYMBOLS, PILOT_SUBCARRIER_SPACING, FFT_SIZE, NUM_OFDM_SYMBOLS, SUBCARRIER_SPACING, CARRIER_FREQUENCY, OFDM_SYMBOL_DURATION
from sklearn.linear_model import Ridge

def estimate_channel(y, no):
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
