import json
import urllib.request
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
OUT = Path("model_artifacts")
OUT.mkdir(exist_ok=True)

SEED = 42
LOOKBACK = 4
TRAIN_SAMPLES = 413
EPOCHS = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "OneHealth-model-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    cases = np.asarray(data["FX"], dtype=np.float32)
    edges = np.asarray(data["edges"], dtype=np.int64)
    if cases.shape[1] != 20:
        raise ValueError(f"Expected 20 regions, got {cases.shape}")
    if edges.shape[1] != 2:
        raise ValueError(f"Unexpected edge shape: {edges.shape}")
    return cases, edges


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return self.linear(torch.matmul(adjacency, x))


class STGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(1, 32)
        self.gcn2 = GCNLayer(32, 32)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, adjacency):
        b, n, t, _ = x.shape
        temporal = []
        for i in range(t):
            h = torch.relu(self.gcn1(x[:, :, i, :], adjacency))
            h = torch.relu(self.gcn2(h, adjacency))
            temporal.append(h)
        h = torch.stack(temporal, dim=1).permute(0, 2, 1, 3).contiguous()
        h = h.reshape(b * n, t, 32)
        h, _ = self.gru(h)
        h = h[:, -1, :]
        return self.output_layer(h).reshape(b, n)


def main():
    set_seed(SEED)
    cases, edges = load_data()

    # The currently public benchmark source has 521 observations.
    # We use it as-is and never invent the missing historical row.
    train_end_week = TRAIN_SAMPLES
    mean = float(cases[:train_end_week].mean())
    std = float(cases[:train_end_week].std())

    z = (cases - mean) / (std + 1e-8)

    xs, ys = [], []
    for t in range(LOOKBACK, len(z)):
        xs.append(z[t-LOOKBACK:t].T[:, :, None])
        ys.append(z[t])

    X = np.asarray(xs, dtype=np.float32)
    Y = np.asarray(ys, dtype=np.float32)

    X_train = torch.from_numpy(X[:TRAIN_SAMPLES])
    Y_train = torch.from_numpy(Y[:TRAIN_SAMPLES])

    X_test = torch.from_numpy(X[TRAIN_SAMPLES:])
    Y_test = torch.from_numpy(Y[TRAIN_SAMPLES:])

    A = torch.zeros(20, 20, dtype=torch.float32)
    for a, b in edges:
        A[a, b] = 1.0
    A += torch.eye(20)
    d = A.sum(dim=1)
    dinv = torch.pow(d, -0.5)
    dinv[torch.isinf(dinv)] = 0.0
    A_norm = torch.diag(dinv) @ A @ torch.diag(dinv)

    model = STGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train, A_norm)
        loss = criterion(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test, A_norm).numpy()

    actual = Y_test.numpy()
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    ss_res = float(np.sum((pred - actual) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)

    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(OUT / "stgnn_disease_only_seed42.npz", **state)

    np.save(OUT / "graph_adjacency.npy", A_norm.numpy())

    metadata = {
        "model": "OneHealth disease-only ST-GNN V4 architecture",
        "seed": SEED,
        "epochs": EPOCHS,
        "lookback": LOOKBACK,
        "regions": 20,
        "public_source_rows": int(cases.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "normalization_mean": mean,
        "normalization_std": std,
        "mae_standardized": mae,
        "rmse_standardized": rmse,
        "r2_standardized": r2,
        "note": "Retrained from the currently public 521-row benchmark source; the historical 522-row Kaggle artifact is not recreated."
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
