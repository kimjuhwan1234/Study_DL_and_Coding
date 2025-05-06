# Predictive metric: one-step ahead prediction MAE using post-hoc RNN in PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error


def predictive_score_metrics(ori_data, generated_data, lengths, iterations=5000, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no, seq_len, dim = ori_data.shape
    hidden_dim = dim // 2

    # Post-hoc predictor
    class RNNPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out)  # (B, T, 1)

    model = RNNPredictor(dim - 1, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    # Training on generated data
    for _ in range(iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]
        X_mb = [generated_data[i][:-1, :(dim - 1)] for i in idx]
        Y_mb = [generated_data[i][1:, (dim - 1):] for i in idx]  # shape (T-1, 1)

        X_mb = torch.nn.utils.rnn.pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X_mb], batch_first=True)
        Y_mb = torch.nn.utils.rnn.pad_sequence([torch.tensor(y, dtype=torch.float32) for y in Y_mb], batch_first=True)

        X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
        y_pred = model(X_mb)

        loss = criterion(y_pred, Y_mb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on original data
    maes = []
    for i in range(no):
        x = torch.tensor(ori_data[i][:-1, :(dim - 1)], dtype=torch.float32).unsqueeze(0).to(device)
        y_true = ori_data[i][1:, (dim - 1)]
        with torch.no_grad():
            y_pred = model(x).squeeze(0).cpu().numpy().flatten()
        maes.append(mean_absolute_error(y_true, y_pred[:len(y_true)]))

    return np.mean(maes)
