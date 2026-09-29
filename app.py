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

    return (
        hidden @ weights["output_layer.weight"].T
        + weights["output_layer.bias"]
    ).reshape(-1)

def trend_label(current, previous):
    if previous <= 0:
        return "Stable"
    change = (current - previous) / previous * 100
    if change >= 10:
        return "Increasing"
    if change <= -10:
        return "Decreasing"
    return "Stable"

def trend_icon(label):
    return {"Increasing": "🔴", "Decreasing": "🟢", "Stable": "🟡"}[label]

st.title("🧬 OneHealth")
st.subheader("Disease Surveillance & Next-Week Forecasting")
st.caption(
    "A doctor-friendly view of the OneHealth disease forecasting research system. "
    "The application forecasts weekly chickenpox cases across 20 Hungarian regions."
)

cases = load_data()
weights, metadata, adjacency = load_model_artifacts()

mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
std = np.asarray(metadata["normalization_std"], dtype=np.float32)
window = ((cases[-4:] - mean) / (std + 1e-8)).T[:, :, None]
pred_z = predict(weights, adjacency, window)
predictions = pred_z * std + mean
predictions = np.maximum(predictions, 0)

# ---------------------------------------------------------------------
# Doctor-friendly overview
# ---------------------------------------------------------------------
st.success("Forecast model connected.")

overview = pd.DataFrame({
    "Region": REGIONS,
    "Current cases": cases[-1, [GRAPH_INDEX[r] for r in REGIONS]],
    "Previous week": cases[-2, [GRAPH_INDEX[r] for r in REGIONS]],
    "Next-week forecast": predictions[[GRAPH_INDEX[r] for r in REGIONS]],
})

overview["Change vs current (%)"] = (
    (overview["Next-week forecast"] - overview["Current cases"])
    / overview["Current cases"].replace(0, np.nan) * 100
).fillna(0)

overview["Trend"] = [
    trend_label(c, p)
    for c, p in zip(overview["Next-week forecast"], overview["Current cases"])
]

st.markdown("### 🩺 Situation overview")

total_current = float(overview["Current cases"].sum())
total_forecast = float(overview["Next-week forecast"].sum())
overall_change = (
    (total_forecast - total_current) / total_current * 100
    if total_current else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cases this week", f"{total_current:.0f}")
c2.metric("Forecast next week", f"{total_forecast:.0f}", f"{overall_change:+.1f}%")
c3.metric("Regions increasing", f"{int((overview['Trend'] == 'Increasing').sum())}")
c4.metric("Regions decreasing", f"{int((overview['Trend'] == 'Decreasing').sum())}")

st.info(
    "How to read this page: the forecast is an estimate of reported chickenpox "
    "cases for the following week. The trend compares that estimate with the "
    "most recent reported week."
)

# ---------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------
st.markdown("### 🔎 Region-level forecast")

region = st.selectbox("Select a region", REGIONS)
idx = GRAPH_INDEX[region]

series = cases[:, idx]
prediction = float(predictions[idx])
current = float(series[-1])
previous = float(series[-2])
change = ((prediction - current) / current * 100) if current else 0
trend = trend_label(prediction, current)

m1, m2, m3 = st.columns(3)
m1.metric("Reported this week", f"{current:.0f}")
m2.metric("Next-week forecast", f"{prediction:.1f}", f"{change:+.1f}%")
m3.metric("Expected trend", f"{trend_icon(trend)} {trend}")

st.markdown("#### 📈 Recent disease pattern")

fig, ax = plt.subplots(figsize=(10, 4))
recent = series[-8:]
x = np.arange(len(recent) + 1)
ax.plot(x[:-1], recent, marker="o", label="Reported cases")
ax.scatter([x[-1]], [prediction], s=70, label="Next-week forecast")
ax.axvline(x[-1] - 0.5, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"W-{7-i}" for i in range(8)] + ["Next"])
ax.set_ylabel("Reported cases")
ax.set_title(f"Disease trend — {region}")
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("#### 💬 Plain-language interpretation")

if trend == "Increasing":
    interpretation = (
        f"For {region}, the model estimates approximately {prediction:.0f} "
        f"reported chickenpox cases next week, compared with {current:.0f} "
        f"this week. This corresponds to an estimated increase of {abs(change):.1f}%."
    )
elif trend == "Decreasing":
    interpretation = (
        f"For {region}, the model estimates approximately {prediction:.0f} "
        f"reported chickenpox cases next week, compared with {current:.0f} "
        f"this week. This corresponds to an estimated decrease of {abs(change):.1f}%."
    )
else:
    interpretation = (
        f"For {region}, the model estimates approximately {prediction:.0f} "
        f"reported chickenpox cases next week. This is broadly similar to "
        f"the {current:.0f} cases reported this week."
    )

st.write(interpretation)

st.caption(
    "The forecast is based on the recent four-week disease history and learned "
    "spatial relationships between the 20 regions. It is a research forecast, "
    "not a diagnosis or clinical recommendation."
)

# ---------------------------------------------------------------------
# Regions requiring attention
# ---------------------------------------------------------------------
st.markdown("### 🗺️ Regional situation")

display_df = overview.copy()
display_df["Current cases"] = display_df["Current cases"].round(0).astype(int)
display_df["Next-week forecast"] = display_df["Next-week forecast"].round(1)
display_df["Change vs current (%)"] = display_df["Change vs current (%)"].round(1)
display_df["Trend"] = [
    f"{trend_icon(t)} {t}" for t in display_df["Trend"]
]

st.dataframe(
    display_df.sort_values("Next-week forecast", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### ⚠️ Regions with the largest expected increases")

attention = overview.sort_values("Change vs current (%)", ascending=False).head(5)
attention = attention[attention["Change vs current (%)"] > 0]

if len(attention):
    for _, row in attention.iterrows():
        st.write(
            f"**{row['Region']}** — "
            f"{row['Current cases']:.0f} reported now → "
            f"{row['Next-week forecast']:.1f} forecast "
            f"({row['Change vs current (%)']:+.1f}%)"
        )
else:
    st.write("No region currently shows an expected increase over the latest week.")

# ---------------------------------------------------------------------
# Explain the model without requiring ML knowledge
# ---------------------------------------------------------------------
st.markdown("### 🧠 How the forecast is produced")

with st.expander("Explain this in simple terms"):
    st.write(
        "**1. Recent history:** The system looks at the most recent four weeks "
        "of reported chickenpox cases."
    )
    st.write(
        "**2. Regional relationships:** It also considers learned relationships "
        "between the 20 geographical regions."
    )
    st.write(
        "**3. Pattern learning:** The ST-GNN learns how disease patterns change "
        "over time and across regions."
    )
    st.write(
        "**4. Forecast:** The model produces an estimate for the following week, "
        "which is converted back to reported case counts."
    )

with st.expander("Technical details for researchers"):
    st.write(
        f"Architecture: disease-only V4 ST-GNN • "
        f"20 regions • {metadata['lookback']}-week lookback • "
        f"{metadata['train_samples']} training samples • "
        f"{metadata['public_source_rows']} source rows"
    )
    st.write(
        f"Deployment model: seed {metadata['seed']} • "
        f"MAE {metadata['mae_cases']:.4f} cases • "
        f"RMSE {metadata['rmse_cases']:.4f} cases • "
        f"R² {metadata['r2_cases']:.4f}"
    )

st.markdown("### 📋 Recent observations")

recent_df = pd.DataFrame({
    "Week": ["4 weeks ago", "3 weeks ago", "2 weeks ago", "Latest week"],
    "Reported cases": series[-4:].round(0).astype(int),
})
st.dataframe(recent_df, use_container_width=True, hide_index=True)

st.warning(
    "Research demonstration only. This application is not a clinical decision-support "
    "system and should not be used for diagnosis, treatment, or public-health decisions."
)
