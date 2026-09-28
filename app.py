import base64
import io
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OneHealth | Disease Forecasting", page_icon="🧬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"

RAW_MODEL_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/stgnn_disease_only_seed42.npz"
ADJ_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/graph_adjacency.npy"
METADATA_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/metadata.json"

REGIONS = [
    "Budapest", "Baranya", "Bács-Kiskun", "Békés",
    "Borsod-Abaúj-Zemplén", "Csongrád-Csanád", "Fejér",
    "Győr-Moson-Sopron", "Hajdú-Bihar", "Heves",
    "Jász-Nagykun-Szolnok", "Komárom-Esztergom", "Nógrád",
    "Pest", "Somogy", "Szabolcs-Szatmár-Bereg", "Tolna",
    "Vas", "Veszprém", "Zala"
]

GRAPH_INDEX = {
    "Bács-Kiskun": 0, "Baranya": 1, "Békés": 2,
    "Borsod-Abaúj-Zemplén": 3, "Budapest": 4, "Csongrád-Csanád": 5,
    "Fejér": 6, "Győr-Moson-Sopron": 7, "Hajdú-Bihar": 8,
    "Heves": 9, "Jász-Nagykun-Szolnok": 10, "Komárom-Esztergom": 11,
    "Nógrád": 12, "Pest": 13, "Somogy": 14,
    "Szabolcs-Szatmár-Bereg": 15, "Tolna": 16, "Vas": 17,
    "Veszprém": 18, "Zala": 19,
}

@st.cache_data
def load_data():
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/stmaletz/glmSTARMA/0bf08dd5ee22cf7d2da03f21b1db7a9701ffca24/data-raw/chickenpox/hungary_chickenpox.csv",
        headers={"User-Agent": "OneHealth-Streamlit"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    lines = text.strip().splitlines()
    raw_columns = lines[0].split(",")[1:]
    raw = np.asarray(
        [[float(v) for v in line.split(",")[1:]] for line in lines[1:]],
        dtype=np.float32,
    )
    raw_index = {name: i for i, name in enumerate(raw_columns)}
    graph_columns = [
        "BACS", "BARANYA", "BEKES", "BORSOD", "BUDAPEST",
        "CSONGRAD", "FEJER", "GYOR", "HAJDU", "HEVES",
        "JASZ", "KOMAROM", "NOGRAD", "PEST", "SOMOGY",
        "SZABOLCS", "TOLNA", "VAS", "VESZPREM", "ZALA",
    ]
    return raw[:, [raw_index[name] for name in graph_columns]]

@st.cache_resource
def load_model_artifacts():
    with urllib.request.urlopen(RAW_MODEL_URL, timeout=30) as response:
        weights = dict(np.load(io.BytesIO(response.read())))
    with urllib.request.urlopen(ADJ_URL, timeout=30) as response:
        adjacency = np.load(io.BytesIO(response.read()))
    with urllib.request.urlopen(METADATA_URL, timeout=30) as response:
        metadata = json.loads(response.read().decode("utf-8"))
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
weights, metadata, adjacency = load_model_artifacts()
st.success("Disease-only ST-GNN checkpoint connected.")
st.info("The deployed model forecasts weekly chickenpox cases using the original case-count dataset.")

left, right = st.columns([1, 2])

with left:
    region = st.selectbox("Select region", REGIONS)
    idx = GRAPH_INDEX[region]
    series = cases[:, idx]

    mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization_std"], dtype=np.float32)
    window = ((cases[-4:] - mean) / (std + 1e-8)).T[:, :, None]
    pred_z = predict(weights, adjacency, window)
    prediction = float(pred_z[idx] * std[idx] + mean[idx])

    st.metric("Next-week chickenpox cases", f"{prediction:.1f}")
    st.caption("Forecast in reported weekly cases")

with right:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="Observed")
    ax.scatter([len(series)], [prediction], marker="o", label="Forecast")
    ax.set_title(f"Weekly chickenpox cases — {region}")
    ax.set_xlabel("Weekly observation index")
    ax.set_ylabel("Reported cases")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("### Recent observations")
recent_df = pd.DataFrame({
    "Lag": ["t-3", "t-2", "t-1", "t"],
    "Observed cases": series[-4:]
})
st.dataframe(recent_df, use_container_width=True, hide_index=True)

st.markdown("### Model information")
st.write(
    f"Seed: {metadata['seed']} • Lookback: {metadata['lookback']} weeks • "
    f"Training samples: {metadata['train_samples']} • Public source rows: {metadata['public_source_rows']}"
)
st.caption(
    "Deployment uses the disease-only V4 ST-GNN architecture retrained directly "
    "on the original Hungarian weekly chickenpox case-count dataset. Region order "
    "is explicitly aligned with the graph nodes, and predictions are inverse-transformed "
    "back to reported case counts."
)
st.warning(
    "Research demonstration only. This application is not a clinical decision-support system "
    "and should not be used for medical or public-health decisions."
)
