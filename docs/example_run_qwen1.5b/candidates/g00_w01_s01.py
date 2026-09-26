import torch
import numpy as np
from link_config import *
from sionna.ofdm.channel_estimation import *

def estimate_channel(y, no):
    # Normalize input
    y = y / torch.sqrt(no)

    # Create an RNN model
    model = SimpleRNNChannelEstimator(input_size=FFT_SIZE, hidden_size=128, num_layers=2)
    
    # Train the model on pilot and data signals
    train_dataset = PilotAndDataDataset(PILOT_SYMBOLS, PILOT_OFDM_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS, NOISE_DISTRIBUTION, CARRIER_FREQUENCY, OFDM_SYMBOL_DURATION)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/BATCH_SIZE:.4f}')

    # Evaluate the model on held-out split
    test_dataset = PilotAndDataDataset(PILOT_SYMBOLS, PILOT_OFDM_SYMBOLS, FFT_SIZE, NUM_OFDM_SYMBOLS, NOISE_DISTRIBUTION, CARRIER_FREQUENCY, OFDM_SYMBOL_DURATION)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    mse_ratios = []
    gains_db = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            mse = torch.mean((outputs - targets) ** 2, dim=(1, 2))
            mse_ratio = torch.mean(mse / target_noise_variance)
            mse_ratios.append(mse_ratio.item())
            gains_db.append(10 * torch.log10(target_noise_variance / mse))

    return outputs.numpy(), mse_ratios, gains_db
