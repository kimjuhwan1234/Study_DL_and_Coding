# Predictive metric: discriminative score using post-hoc RNN classifier in PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_posthoc_discriminator(input_dim, hidden_dim):
    return nn.GRU(input_dim, hidden_dim, batch_first=True), nn.Linear(hidden_dim, 1)


def discriminative_score_metrics(ori_data, generated_data, lengths, iterations=2000, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no, seq_len, dim = ori_data.shape
    hidden_dim = dim // 2

    # Split into train/test sets
    idx = np.random.permutation(no)
    train_size = int(no * 0.8)
    train_idx, test_idx = idx[:train_size], idx[train_size:]

    train_real = torch.tensor(ori_data[train_idx], dtype=torch.float32)
    train_fake = torch.tensor(generated_data[train_idx], dtype=torch.float32)
    test_real = torch.tensor(ori_data[test_idx], dtype=torch.float32)
    test_fake = torch.tensor(generated_data[test_idx], dtype=torch.float32)

    # Create datasets
    x_train = torch.cat([train_real, train_fake], dim=0)
    y_train = torch.cat([
        torch.ones(len(train_real), 1),
        torch.zeros(len(train_fake), 1)
    ], dim=0)

    x_test = torch.cat([test_real, test_fake], dim=0)
    y_test = torch.cat([
        torch.ones(len(test_real), 1),
        torch.zeros(len(test_fake), 1)
    ], dim=0)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model
    rnn, fc = build_posthoc_discriminator(dim, hidden_dim)
    rnn.to(device)
    fc.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=0.001)

    # Training loop
    for _ in range(iterations):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out, _ = rnn(x_batch)
            logits = fc(out[:, -1, :])
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    rnn.eval()
    fc.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.numpy().flatten()
        out, _ = rnn(x_test)
        logits = fc(out[:, -1, :])
        preds = torch.sigmoid(logits).cpu().numpy().flatten()
        acc = accuracy_score(y_test, preds > 0.5)

    return abs(acc - 0.5)
