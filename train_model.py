import json
import urllib.request
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Raw UCI mirror of the Hungarian Chickenpox Cases dataset.
# The JSON benchmark used previously contains transformed values; the app
# should display the original weekly case counts.
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/stmaletz/glmSTARMA/"
    "0bf08dd5ee22cf7d2da03f21b1db7a9701ffca24/"
    "data-raw/chickenpox/hungary_chickenpox.csv"
)

OUT = Path("model_artifacts")
OUT.mkdir(exist_ok=True)

SEED = 42
LOOKBACK = 4
TRAIN_SAMPLES = 413
EPOCHS = 100

# Graph-node order used by the PyTorch Geometric Temporal benchmark.
REGIONS = [
    "Bacs", "Baranya", "Bekes", "Borsod", "Budapest",
    "Csongrad", "Fejer", "Gyor", "Hajdu", "Heves",
    "Jasz", "Komarom", "Nograd", "Pest", "Somogy",
    "Szabolcs", "Tolna", "Vas", "Veszprem", "Zala"
]

RAW_COLUMNS = [
    "BUDAPEST", "BARANYA", "BACS", "BEKES", "BORSOD",
    "CSONGRAD", "FEJER", "GYOR", "HAJDU", "HEVES",
    "JASZ", "KOMAROM", "NOGRAD", "PEST", "SOMOGY",
    "SZABOLCS", "TOLNA", "VAS", "VESZPREM", "ZALA"
]

GRAPH_TO_RAW = [
    RAW_COLUMNS.index("BACS"),
    RAW_COLUMNS.index("BARANYA"),
    RAW_COLUMNS.index("BEKES"),
    RAW_COLUMNS.index("BORSOD"),
    RAW_COLUMNS.index("BUDAPEST"),
    RAW_COLUMNS.index("CSONGRAD"),
    RAW_COLUMNS.index("FEJER"),
    RAW_COLUMNS.index("GYOR"),
    RAW_COLUMNS.index("HAJDU"),
    RAW_COLUMNS.index("HEVES"),
    RAW_COLUMNS.index("JASZ"),
    RAW_COLUMNS.index("KOMAROM"),
    RAW_COLUMNS.index("NOGRAD"),
    RAW_COLUMNS.index("PEST"),
    RAW_COLUMNS.index("SOMOGY"),
    RAW_COLUMNS.index("SZABOLCS"),
    RAW_COLUMNS.index("TOLNA"),
    RAW_COLUMNS.index("VAS"),
    RAW_COLUMNS.index("VESZPREM"),
    RAW_COLUMNS.index("ZALA"),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    req = urllib.request.Request(
        RAW_DATA_URL,
        headers={"User-Agent": "OneHealth-model-builder"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8")

    lines = text.strip().splitlines()
    header = lines[0].split(",")
    if header[1:] != RAW_COLUMNS:
        raise ValueError("Unexpected Hungarian chickenpox column order.")

    raw = np.asarray(
        [[float(v) for v in line.split(",")[1:]] for line in lines[1:]],
        dtype=np.float32,
    )

    # Convert raw CSV order to graph/model node order.
    cases = raw[:, GRAPH_TO_RAW]

    if cases.shape != (521, 20):
        raise ValueError(f"Expected (521, 20), got {cases.shape}")

    return cases


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
    cases = load_data()

    # Standardize each region separately, using training-period statistics.
    train_end_week = TRAIN_SAMPLES + LOOKBACK
    mean = cases[:train_end_week].mean(axis=0)
    std = cases[:train_end_week].std(axis=0)
    z = (cases - mean) / (std + 1e-8)

    xs, ys = [], []
    for t in range(LOOKBACK, len(z)):
        xs.append(z[t - LOOKBACK:t].T[:, :, None])
        ys.append(z[t])

    X = np.asarray(xs, dtype=np.float32)
    Y = np.asarray(ys, dtype=np.float32)

    X_train = torch.from_numpy(X[:TRAIN_SAMPLES])
    Y_train = torch.from_numpy(Y[:TRAIN_SAMPLES])
    X_test = torch.from_numpy(X[TRAIN_SAMPLES:])
    Y_test = torch.from_numpy(Y[TRAIN_SAMPLES:])

    # Reuse the verified 20-node graph from the benchmark JSON.
    graph_url = (
        "https://raw.githubusercontent.com/benedekrozemberczki/"
        "pytorch_geometric_temporal/master/dataset/chickenpox.json"
    )
    req = urllib.request.Request(
        graph_url, headers={"User-Agent": "OneHealth-model-builder"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        graph = json.loads(r.read().decode("utf-8"))

    edges = np.asarray(graph["edges"], dtype=np.int64)
    A = torch.zeros(20, 20, dtype=torch.float32)
    for a, b in edges:
        A[a, b] = 1.0
    A += torch.eye(20)
    d = A.sum(dim=1)
    dinv = torch.pow(d, -0.5)
    dinv[torch.isinf(dinv)] = 0.0
    A_norm = torch.diag(dinv) @ A @ torch.diag(dinv)

    model = STGNN()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5
    )
    criterion = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train, A_norm)
        loss = criterion(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_z = model(X_test, A_norm).numpy()

    actual_z = Y_test.numpy()
    mean_flat = np.tile(mean, X_test.shape[0])
    std_flat = np.tile(std, X_test.shape[0])
    pred_cases = pred_z.reshape(-1) * std_flat + mean_flat
    actual_cases = actual_z.reshape(-1) * std_flat + mean_flat

    mae = float(np.mean(np.abs(pred_cases - actual_cases)))
    rmse = float(np.sqrt(np.mean((pred_cases - actual_cases) ** 2)))
    ss_res = float(np.sum((pred_cases - actual_cases) ** 2))
    ss_tot = float(np.sum((actual_cases - actual_cases.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)

    state = {
        k: v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
    }
    np.savez(OUT / "stgnn_disease_only_seed42.npz", **state)
    np.save(OUT / "graph_adjacency.npy", A_norm.numpy())

    metadata = {
        "model": "OneHealth disease-only ST-GNN V4 architecture",
        "seed": SEED,
        "epochs": EPOCHS,
        "lookback": LOOKBACK,
        "regions": 20,
        "region_order": REGIONS,
        "public_source_rows": int(cases.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "mae_cases": mae,
        "rmse_cases": rmse,
        "r2_cases": r2,
        "raw_source": RAW_DATA_URL,
        "note": (
            "Retrained on the original weekly case-count CSV. "
            "The graph/model node order is explicitly aligned with the "
            "PyTorch Geometric Temporal benchmark."
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
