import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="OneHealth | Disease Forecasting", page_icon="🧬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
MODEL_DIR = Path("model_artifacts")

REGIONS = [
    "Budapest", "Baranya", "Bács-Kiskun", "Békés",
    "Borsod-Abaúj-Zemplén", "Csongrád-Csanád", "Fejér",
    "Győr-Moson-Sopron", "Hajdú-Bihar", "Heves",
    "Jász-Nagykun-Szolnok", "Komárom-Esztergom", "Nógrád",
    "Pest", "Somogy", "Szabolcs-Szatmár-Bereg", "Tolna",
    "Vas", "Veszprém", "Zala"
]

@st.cache_data
def load_data():
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "OneHealth-Streamlit"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return np.asarray(data["FX"], dtype=np.float32)

@st.cache_resource
def load_model_artifacts():
    model_path = MODEL_DIR / "stgnn_disease_only_seed42.npz"
    meta_path = MODEL_DIR / "metadata.json"
    adj_path = MODEL_DIR / "graph_adjacency.npy"
    if not (model_path.exists() and meta_path.exists() and adj_path.exists()):
        return None
    weights = dict(np.load(model_path))
    metadata = json.loads(meta_path.read_text())
    adjacency = np.load(adj_path)
    return weights, metadata, adjacency

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def predict(weights, adjacency, x):
    g1_w = weights["gcn1.linear.weight"]
    g1_b = weights["gcn1.linear.bias"]
    g2_w = weights["gcn2.linear.weight"]
    g2_b = weights["gcn2.linear.bias"]
    h_seq = []
    for t in range(4):
        h = adjacency @ x[:, t, :]
        h = h @ g1_w.T + g1_b
        h = np.maximum(h, 0)
        h = adjacency @ h
        h = h @ g2_w.T + g2_b
        h = np.maximum(h, 0)
        h_seq.append(h)
    h = np.transpose(np.stack(h_seq, axis=0), (1, 0, 2))
    Wih = weights["gru.weight_ih_l0"]
    Whh = weights["gru.weight_hh_l0"]
    bih = weights["gru.bias_ih_l0"]
    bhh = weights["gru.bias_hh_l0"]
    hidden = np.zeros((20, 32), dtype=np.float32)
    for t in range(4):
        inp = h[:, t, :]
        gi = inp @ Wih.T + bih
        gh = hidden @ Whh.T + bhh
        ir, iz, inn = np.split(gi, 3, axis=1)
        hr, hz, hnn = np.split(gh, 3, axis=1)
        r = sigmoid(ir + hr)
        z = sigmoid(iz + hz)
        n = np.tanh(inn + r * hnn)
        hidden = (1.0 - z) * n + z * hidden
    return (hidden @ weights["output_layer.weight"].T + weights["output_layer.bias"]).reshape(-1)

st.title("🧬 OneHealth")
st.subheader("Spatio-Temporal Disease Forecasting")

cases = load_data()
artifacts = load_model_artifacts()

if artifacts is None:
    st.warning("The ST-GNN checkpoint is being prepared. The interface is temporarily showing the persistence baseline.")
else:
    st.success("Disease-only ST-GNN checkpoint connected.")

left, right = st.columns([1, 2])

with left:
    region = st.selectbox("Select region", REGIONS)
    idx = REGIONS.index(region)
    series = cases[:, idx]
    recent = series[-4:]

    if artifacts is not None:
        weights, metadata, adjacency = artifacts
        mean = float(metadata["normalization_mean"])
        std = float(metadata["normalization_std"])
        window = ((recent - mean) / (std + 1e-8)).reshape(1, 4, 1)
        pred_z = predict(weights, adjacency, np.tile(window, (20, 1, 1)))
        prediction = float(pred_z[idx] * std + mean)
        label = "ST-GNN forecast"
    else:
        prediction = float(recent[-1])
        label = "Persistence baseline"

    st.metric("Next-week forecast", f"{prediction:.2f}")
    st.caption(label)

with right:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="Observed")
    ax.scatter([len(series)], [prediction], marker="o", label="Forecast")
    ax.set_title(f"Weekly chickenpox cases — {region}")
    ax.set_xlabel("Weekly observation index")
    ax.set_ylabel("Cases")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("### Recent observations")
recent_df = pd.DataFrame({"Lag": ["t-3", "t-2", "t-1", "t"], "Observed value": recent})
st.dataframe(recent_df, use_container_width=True, hide_index=True)

if artifacts is not None:
    st.markdown("### Model information")
    st.write(
        f"Seed: {metadata['seed']} • Lookback: {metadata['lookback']} weeks • "
        f"Training samples: {metadata['train_samples']} • Public source rows: {metadata['public_source_rows']}"
    )
    st.caption(
        "This deployment retrains the notebook's disease-only V4 architecture from the currently public "
        "521-row benchmark source. It does not recreate the missing historical 522-row Kaggle artifact."
    )

st.warning("Research demonstration only. This application is not a clinical decision-support system and should not be used for medical or public-health decisions.")
