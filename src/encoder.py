"""Temporal encoder, reproducing the classifier released with the benchmark.

Note the effective shape: the loader flattens a 10-frame window into a 140-dimensional
vector and feeds it as a length-1 sequence, so the LSTM does not see a time axis. That is
the reference design and is kept unchanged, but the model should not be described as
capturing temporal dependencies.

The head boundary is fc2. We freeze everything up to and including the ReLU after fc1 and
replace fc2 with the analytic head, so the frozen representation has FC_HIDDEN_DIM
dimensions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=C.INPUT_SIZE, hidden_dim=C.HIDDEN_SIZE,
                 output_dim=C.OUTPUT_SIZE, num_layers=C.NUM_LAYERS,
                 fc_hidden_dim=C.FC_HIDDEN_DIM, head_dropout=C.HEAD_DROPOUT):
        super().__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def _trunk(self, x):
        b = x.size(0)
        h0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return F.relu(self.fc1(out[:, -1, :]))

    def forward(self, x):
        return self.fc2(self.dropout(self._trunk(x)))

    @torch.no_grad()
    def features(self, X, batch=4096, device="cpu"):
        """Frozen representation for X of shape (N, 140); returns (N, fc_hidden_dim)."""
        self.eval()
        outs = []
        for s in range(0, len(X), batch):
            xb = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)
            outs.append(self._trunk(xb.view(-1, 1, C.INPUT_SIZE)).cpu().numpy())
        return np.concatenate(outs).astype(np.float64)


def train_base(model, X, y, epochs=C.EPOCHS, lr=C.LEARNING_RATE, bs=C.BATCH_SIZE,
               patience=C.PATIENCE, device="cpu", seed=0, verbose=False):
    """Backpropagation on the first domain, after which the encoder is frozen."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X, dtype=torch.float32).view(-1, 1, C.INPUT_SIZE)
    yt = torch.as_tensor(y, dtype=torch.long)
    n, best, bad = len(Xt), float("inf"), 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(device)), yt[idx].to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tot /= n
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep:3d}  loss {tot:.4f}")
        if tot < best - 1e-5:
            best, bad = tot, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def n_params(model):
    return sum(p.numel() for p in model.parameters())
