import io
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OneHealth | Disease Surveillance",
    page_icon="🧬",
    layout="wide",
)

RAW_MODEL_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/stgnn_disease_only_seed42.npz"
ADJ_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/graph_adjacency.npy"
METADATA_URL = "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/model_artifacts/metadata.json"

REGIONS = [
    "Budapest", "Baranya", "Bács-Kiskun", "Békés",
    "Borsod-Abaúj-Zemplén", "Csongrád-Csanád", "Fejér",
    "Győr-Moson-Sopron", "Hajdú-Bihar", "Heves",
    "Jász-Nagykun-Szolnok", "Komárom-Esztergom", "Nógrád",
    "Pest", "Somogy", "Szabolcs-Szatmár-Bereg", "Tolna",
    "Vas", "Veszprém", "Zala",
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

    return (
        hidden @ weights["output_layer.weight"].T
        + weights["output_layer.bias"]
    ).reshape(-1)


def trend_text(current, forecast):
    if current <= 0:
        return "No recent cases"
    change = (forecast - current) / current * 100
    if change >= 10:
        return "Increasing"
    if change <= -10:
        return "Decreasing"
    return "Broadly stable"


def trend_symbol(trend):
    return {
        "Increasing": "🔴",
        "Decreasing": "🟢",
        "Broadly stable": "🟡",
        "No recent cases": "⚪",
    }[trend]


# ---------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------
cases = load_data()
weights, metadata, adjacency = load_model_artifacts()

mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
std = np.asarray(metadata["normalization_std"], dtype=np.float32)

window = ((cases[-4:] - mean) / (std + 1e-8)).T[:, :, None]
pred_z = predict(weights, adjacency, window)
predictions = np.maximum(pred_z * std + mean, 0)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🧬 OneHealth")
st.subheader("Disease Surveillance & Next-Week Forecasting")
st.caption(
    "A simplified view of the OneHealth research system for interpreting "
    "weekly chickenpox patterns across 20 Hungarian regions."
)

st.success("Forecast model connected.")

# ---------------------------------------------------------------------
# National overview
# ---------------------------------------------------------------------
overview = pd.DataFrame({
    "Region": REGIONS,
    "Reported this week": [
        cases[-1, GRAPH_INDEX[r]] for r in REGIONS
    ],
    "Forecast next week": [
        predictions[GRAPH_INDEX[r]] for r in REGIONS
    ],
})

overview["Change (%)"] = np.where(
    overview["Reported this week"] > 0,
    (overview["Forecast next week"] - overview["Reported this week"])
    / overview["Reported this week"] * 100,
    0,
)

overview["Trend"] = [
    trend_text(current, forecast)
    for current, forecast in zip(
        overview["Reported this week"],
        overview["Forecast next week"],
    )
]

total_current = float(overview["Reported this week"].sum())
total_forecast = float(overview["Forecast next week"].sum())
total_change = (
    (total_forecast - total_current) / total_current * 100
    if total_current > 0 else 0
)

st.markdown("### 🩺 Situation overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reported this week", f"{total_current:.0f}")
c2.metric("Forecast next week", f"{total_forecast:.0f}", f"{total_change:+.1f}%")
c3.metric(
    "Regions increasing",
    str(int((overview["Trend"] == "Increasing").sum())),
)
c4.metric(
    "Regions decreasing",
    str(int((overview["Trend"] == "Decreasing").sum())),
)

st.info(
    "The forecast is an estimate of reported cases for the following week. "
    "An increasing/decreasing label compares the forecast with the most recent "
    "reported week; it is not a clinical risk score."
)

# ---------------------------------------------------------------------
# Selected region
# ---------------------------------------------------------------------
st.markdown("### 🔎 Regional forecast")

region = st.selectbox("Select region", REGIONS)
idx = GRAPH_INDEX[region]
series = cases[:, idx]

current = float(series[-1])
previous = float(series[-2])
prediction = float(predictions[idx])

if current > 0:
    change = (prediction - current) / current * 100
else:
    change = 0.0

trend = trend_text(current, prediction)

m1, m2, m3 = st.columns(3)
m1.metric("Reported this week", f"{current:.0f}")
m2.metric("Next-week forecast", f"{prediction:.1f}", f"{change:+.1f}%")
m3.metric("Expected trend", f"{trend_symbol(trend)} {trend}")

# ---------------------------------------------------------------------
# Recent trend chart
# ---------------------------------------------------------------------
st.markdown("#### 📈 Recent disease pattern")

recent = series[-8:]
x = np.arange(9)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x[:-1], recent, marker="o", label="Reported cases")
ax.scatter(
    [x[-1]],
    [prediction],
    s=80,
    marker="o",
    label="Next-week forecast",
)
ax.axvline(x[-1] - 0.5, linestyle="--", alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(
    ["8 weeks ago", "7 weeks ago", "6 weeks ago", "5 weeks ago",
     "4 weeks ago", "3 weeks ago", "2 weeks ago", "Latest", "Next week"],
    rotation=25,
    ha="right",
)
ax.set_ylabel("Reported cases")
ax.set_title(f"Chickenpox cases — {region}")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ---------------------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------------------
st.markdown("#### 💬 Plain-language interpretation")

if trend == "Increasing":
    st.write(
        f"For **{region}**, the model estimates approximately "
        f"**{prediction:.0f} reported cases next week**, compared with "
        f"**{current:.0f} this week**. The forecast is about "
        f"**{abs(change):.1f}% higher** than the latest reported value."
    )
elif trend == "Decreasing":
    st.write(
        f"For **{region}**, the model estimates approximately "
        f"**{prediction:.0f} reported cases next week**, compared with "
        f"**{current:.0f} this week**. The forecast is about "
        f"**{abs(change):.1f}% lower** than the latest reported value."
    )
elif trend == "No recent cases":
    st.write(
        f"No cases were reported in the latest week for **{region}**. "
        f"The model forecasts approximately **{prediction:.0f} cases next week**."
    )
else:
    st.write(
        f"For **{region}**, the model estimates approximately "
        f"**{prediction:.0f} reported cases next week**. This is broadly "
        f"similar to the **{current:.0f} cases reported this week**."
    )

with st.expander("What does this mean?"):
    st.write(
        "The number shown is a model forecast, not a diagnosis. "
        "It describes the expected number of reported chickenpox cases "
        "in the following week based on the recent disease pattern."
    )

# ---------------------------------------------------------------------
# Regional comparison
# ---------------------------------------------------------------------
st.markdown("### 🗺️ Regional situation")

table = overview.copy()
table["Reported this week"] = table["Reported this week"].round(0).astype(int)
table["Forecast next week"] = table["Forecast next week"].round(1)
table["Change (%)"] = table["Change (%)"].round(1)
table["Trend"] = [
    f"{trend_symbol(t)} {t}" for t in table["Trend"]
]

st.dataframe(
    table.sort_values("Forecast next week", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------
# Simple attention list — descriptive, not clinical advice
# ---------------------------------------------------------------------
st.markdown("#### 📌 Regions with an expected increase")

increasing = overview[
    overview["Trend"] == "Increasing"
].sort_values("Change (%)", ascending=False).head(5)

if increasing.empty:
    st.write("No region currently meets the dashboard's increasing-trend threshold.")
else:
    for _, row in increasing.iterrows():
        st.write(
            f"**{row['Region']}** — "
            f"{row['Reported this week']:.0f} reported → "
            f"{row['Forecast next week']:.1f} forecast "
            f"({row['Change (%)']:+.1f}%)"
        )

# ---------------------------------------------------------------------
# Explain the model simply
# ---------------------------------------------------------------------
st.markdown("### 🧠 How the forecast is produced")

with st.expander("For a non-technical user"):
    st.write(
        "**1. Recent disease history:** The system uses the most recent "
        "four weeks of reported chickenpox cases."
    )
    st.write(
        "**2. Regional relationships:** The model uses the spatial "
        "relationships between the 20 regions."
    )
    st.write(
        "**3. Pattern learning:** The ST-GNN learns patterns that change "
        "over time and across regions."
    )
    st.write(
        "**4. Next-week estimate:** The learned pattern is used to estimate "
        "the number of reported cases for the following week."
    )

with st.expander("Technical details"):
    st.write(
        f"Architecture: disease-only V4 ST-GNN • "
        f"20 regions • {metadata['lookback']}-week lookback • "
        f"{metadata['train_samples']} training samples"
    )
    st.write(
        f"Deployment checkpoint: seed {metadata['seed']} • "
        f"MAE {metadata['mae_cases']:.4f} cases • "
        f"RMSE {metadata['rmse_cases']:.4f} cases • "
        f"R² {metadata['r2_cases']:.4f}"
    )

# ---------------------------------------------------------------------
# Recent observations
# ---------------------------------------------------------------------
st.markdown("### 📋 Recent observations")

recent_df = pd.DataFrame({
    "Period": ["4 weeks ago", "3 weeks ago", "2 weeks ago", "Latest week"],
    "Reported cases": series[-4:].round(0).astype(int),
})

st.dataframe(
    recent_df,
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "Research demonstration only. This application is not a clinical "
    "decision-support system and should not be used for diagnosis, treatment, "
    "or public-health decisions."
)
