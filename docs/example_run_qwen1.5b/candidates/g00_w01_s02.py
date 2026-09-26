import torch
import numpy as np
from link_config import *
from sionna.ofdm.channel_estimation import *
from sionna.ofdm.modulation import QPSK
from sionna.ofdm.tiled_signal import TiledSignal
from sionna.utils.dft_denoiser import DFTDenoiser

class SimpleRNNChannelEstimator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNNChannelEstimator, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, FFT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.shape[0], hidden_size).to(device=x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1])
        return out

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

def load_candidate(candidate_path):
    try:
        spec = importlib.util.spec_from_file_location("candidate_module", candidate_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ModuleNotFoundError:
        return None

def main():
    candidate_path = "path_to_your_candidate_model.pth"
    candidate_module = load_candidate(candidate_path)
    if candidate_module:
        candidate_model = candidate_module.SimpleRNNChannelEstimator(
            input_size=FFT_SIZE, hidden_size=128, num_layers=2
        )
        candidate_model.load_state_dict(torch.load(candidate_path))
        candidate_model.to(device=device)
        
        # Example usage
        y_example = torch.randn(BATCH_SIZE, 14, FFT_SIZE)
        no_example = torch.randn(BATCH_SIZE)
        _, mse_ratios, gains_db = estimate_channel(y_example, no_example)
        print(f"MSE Ratios: {mse_ratios}")
        print(f"Gains DB: {gains_db}")
    else:
        print("Failed to load candidate model.")

if __name__ == "__main__":
    main()
