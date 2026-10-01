import io
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OneHealth | Disease Surveillance",
    page_icon="🩺",
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
    return "Similar to latest week"


def trend_icon(trend):
    return {
        "Increasing": "🔴",
        "Decreasing": "🟢",
        "Similar to latest week": "🟡",
        "No recent cases": "⚪",
    }[trend]


cases = load_data()
weights, metadata, adjacency = load_model_artifacts()

mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
std = np.asarray(metadata["normalization_std"], dtype=np.float32)
window = ((cases[-4:] - mean) / (std + 1e-8)).T[:, :, None]
pred_z = predict(weights, adjacency, window)
predictions = np.maximum(pred_z * std + mean, 0)

overview = pd.DataFrame({
    "Region": REGIONS,
    "Reported now": [cases[-1, GRAPH_INDEX[r]] for r in REGIONS],
    "Forecast next week": [predictions[GRAPH_INDEX[r]] for r in REGIONS],
})

overview["Change (%)"] = np.where(
    overview["Reported now"] > 0,
    (overview["Forecast next week"] - overview["Reported now"])
    / overview["Reported now"] * 100,
    0,
)
overview["Trend"] = [
    trend_text(now, forecast)
    for now, forecast in zip(
        overview["Reported now"], overview["Forecast next week"]
    )
]

total_now = float(overview["Reported now"].sum())
total_next = float(overview["Forecast next week"].sum())
total_change = (
    (total_next - total_now) / total_now * 100 if total_now > 0 else 0
)

# ---------------------------------------------------------------------
# Doctor-first landing page
# ---------------------------------------------------------------------
st.title("🩺 OneHealth")
st.subheader("Chickenpox Surveillance Dashboard")

st.markdown(
    "**At a glance:** latest reported chickenpox activity and the model's "
    "estimate for the following week across 20 regions."
)

st.success("Forecast available")

st.markdown("### What is happening?")

a, b, c = st.columns(3)
a.metric("Reported cases — latest week", f"{total_now:.0f}")
b.metric("Estimated cases — next week", f"{total_next:.0f}")
c.metric("Change in estimated cases", f"{total_change:+.1f}%")

st.info(
    "This dashboard is designed to be read without knowledge of artificial "
    "intelligence or deep learning. Start with the regional forecast below."
)

# ---------------------------------------------------------------------
# Region-level view
# ---------------------------------------------------------------------
st.markdown("### 📍 Choose a region")

region = st.selectbox(
    "Region",
    REGIONS,
    label_visibility="collapsed",
)

idx = GRAPH_INDEX[region]
series = cases[:, idx]
current = float(series[-1])
prediction = float(predictions[idx])
change = (prediction - current) / current * 100 if current > 0 else 0
trend = trend_text(current, prediction)

if trend == "Increasing":
    st.error(
        f"**{region}: increasing pattern**\n\n"
        f"Approximately **{prediction:.0f} cases** are estimated next week, "
        f"compared with **{current:.0f} cases** in the latest week."
    )
elif trend == "Decreasing":
    st.success(
        f"**{region}: decreasing pattern**\n\n"
        f"Approximately **{prediction:.0f} cases** are estimated next week, "
        f"compared with **{current:.0f} cases** in the latest week."
    )
elif trend == "No recent cases":
    st.info(
        f"**{region}: no cases in the latest week**\n\n"
        f"Approximately **{prediction:.0f} cases** are estimated next week."
    )
else:
    st.warning(
        f"**{region}: similar to the latest week**\n\n"
        f"Approximately **{prediction:.0f} cases** are estimated next week, "
        f"compared with **{current:.0f} cases** in the latest week."
    )

m1, m2, m3 = st.columns(3)
m1.metric("Latest reported", f"{current:.0f}")
m2.metric("Next-week estimate", f"{prediction:.1f}")
m3.metric("Change from latest", f"{change:+.1f}%")

st.caption(
    "Increasing/decreasing/similar describe numerical change only. "
    "They are not clinical severity or patient-risk categories."
)

# ---------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------
st.markdown("### 📈 Disease pattern")

recent = series[-8:]
x = np.arange(9)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x[:-1], recent, marker="o", label="Reported")
ax.scatter(
    [x[-1]],
    [prediction],
    s=90,
    marker="o",
    label="Estimated next week",
)
ax.axvline(x[-1] - 0.5, linestyle="--", alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "8 wks ago", "7 wks ago", "6 wks ago", "5 wks ago",
        "4 wks ago", "3 wks ago", "2 wks ago", "Latest", "Next",
    ]
)
ax.set_ylabel("Number of reported cases")
ax.set_title(f"{region}: recent reported cases and next-week estimate")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ---------------------------------------------------------------------
# Plain-language answer
# ---------------------------------------------------------------------
st.markdown("### 💬 In simple words")

if trend == "Increasing":
    sentence = (
        f"The forecast suggests that **{region} may have more reported cases "
        f"next week**: about **{prediction:.0f}**, compared with **{current:.0f}** "
        f"in the latest week."
    )
elif trend == "Decreasing":
    sentence = (
        f"The forecast suggests that **{region} may have fewer reported cases "
        f"next week**: about **{prediction:.0f}**, compared with **{current:.0f}** "
        f"in the latest week."
    )
elif trend == "No recent cases":
    sentence = (
        f"**No cases were reported in the latest week** for {region}. "
        f"The forecast is approximately **{prediction:.0f} cases** next week."
    )
else:
    sentence = (
        f"The forecast for **{region} is broadly similar to the latest week**: "
        f"about **{prediction:.0f} cases** next week versus **{current:.0f}** latest."
    )

st.write(sentence)

with st.expander("How should I read this?"):
    st.write(
        "This is a population-level disease-surveillance forecast. It estimates "
        "the number of reported chickenpox cases for the following week. "
        "It does not evaluate an individual patient and does not provide a diagnosis, "
        "treatment recommendation, or prognosis."
    )

# ---------------------------------------------------------------------
# --- Prediction map (Mappls Web SDK) ---
# ---------------------------------------------------------------------

MAP_COORDS = {
    "Bács-Kiskun": (46.6700, 19.4800),
    "Baranya": (46.0700, 18.2300),
    "Békés": (46.6800, 21.0900),
    "Borsod-Abaúj-Zemplén": (48.1000, 21.3000),
    "Budapest": (47.4979, 19.0402),
    "Csongrád-Csanád": (46.2500, 20.1500),
    "Fejér": (47.1700, 18.5500),
    "Győr-Moson-Sopron": (47.6500, 17.2500),
    "Hajdú-Bihar": (47.4500, 21.4000),
    "Heves": (47.9000, 20.1000),
    "Jász-Nagykun-Szolnok": (47.3500, 20.2000),
    "Komárom-Esztergom": (47.6000, 18.2000),
    "Nógrád": (48.0000, 19.6000),
    "Pest": (47.4500, 19.5000),
    "Somogy": (46.4500, 17.7500),
    "Szabolcs-Szatmár-Bereg": (48.0000, 21.9000),
    "Tolna": (46.5000, 18.5500),
    "Vas": (47.1000, 16.7000),
    "Veszprém": (47.1000, 17.9000),
    "Zala": (46.8400, 16.8400),
}

st.subheader("🗺️ Prediction map")
st.caption(
    "Interactive Mappls map showing the model's next-week chickenpox "
    "predictions across the 20 regions."
)

import streamlit.components.v1 as components

mappls_token = st.secrets.get("MAPPLS_API_KEY", "")

if not mappls_token:
    st.warning(
        "Mappls map is not configured. Add MAPPLS_API_KEY to Streamlit Secrets "
        "to enable the Mappls map."
    )
else:
    map_points = []
    for region_name in overview["Region"]:
        if region_name in MAP_COORDS:
            lat, lon = MAP_COORDS[region_name]
            forecast = float(
                overview.loc[
                    overview["Region"] == region_name, "Forecast next week"
                ].iloc[0]
            )
            map_points.append(
                {
                    "region": region_name,
                    "lat": float(lat),
                    "lon": float(lon),
                    "forecast": forecast,
                    "current": float(
                        overview.loc[
                            overview["Region"] == region_name, "Reported now"
                        ].iloc[0]
                    ),
                }
            )

    points_json = json.dumps(map_points, ensure_ascii=False)

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta charset="utf-8">
      <style>
        html, body, #onehealth-map {{
          margin: 0;
          padding: 0;
          width: 100%;
          height: 560px;
          overflow: hidden;
        }}
      </style>
      <script>
        function initMap1() {{
          const points = {points_json};
          const map = new mappls.Map("onehealth-map", {{
            center: [47.20, 19.40],
            zoom: 6,
            zoomControl: true,
            location: false
          }});

          map.addListener("load", function() {{
            const bounds = [];

            points.forEach(function(p) {{
              let colorName = "none";
              let trend = "No recent cases";

              if (p.current > 0) {{
                const change = ((p.forecast - p.current) / p.current) * 100;
                if (change >= 10) {{
                  colorName = "increasing";
                  trend = "Increasing";
                }} else if (change <= -10) {{
                  colorName = "decreasing";
                  trend = "Decreasing";
                }} else {{
                  colorName = "similar";
                  trend = "Similar to latest week";
                }}
              }}

              const iconUrl =
                "https://raw.githubusercontent.com/sdb626462-star/onehealth/main/map_icons/" +
                colorName + ".svg";

              new mappls.Marker({{
                map: map,
                position: {{
                  lat: p.lat,
                  lng: p.lon
                }},
                icon_url: iconUrl,
                fitbounds: false,
                popupHtml:
                  "<div style='min-width:200px;font-family:Arial,sans-serif'>" +
                  "<b>" + p.region + "</b><br>" +
                  "<b>" + trend + "</b><br>" +
                  "Latest reported: <b>" + p.current.toFixed(0) + "</b><br>" +
                  "Next-week estimate: <b>" + p.forecast.toFixed(1) +
                  "</b> cases</div>"
              }});

              bounds.push([p.lat, p.lon]);
            }});

            if (bounds.length && map.fitBounds) {{
              map.fitBounds(bounds);
            }}
          }});
        }}
      </script>
      <script
        src="https://sdk.mappls.com/map/sdk/web?v=3.0&access_token={mappls_token}&callback=initMap1"
        defer
        async>
      </script>
    </head>
    <body>
      <div id="onehealth-map"></div>
    </body>
    </html>
    """
    components.html(html, height=580, scrolling=False)


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------
st.markdown("### 🤖 Model comparison")

st.caption(
    "Independent benchmark on the same 4-week lookback, 20-region dataset "
    "and held-out test period. The deployed forecast remains ST-GNN."
)

stgnn_mae = float(metadata.get("mae_cases", 17.6115))
stgnn_rmse = float(metadata.get("rmse_cases", 28.3550))
stgnn_r2 = float(metadata.get("r2_cases", 0.4804))

comparison = pd.DataFrame(
    [
        {
            "Model": "ST-GNN (deployed)",
            "MAE": stgnn_mae,
            "RMSE": stgnn_rmse,
            "R²": stgnn_r2,
        },
        {
            "Model": "LSTM baseline",
            "MAE": 26.5535,
            "RMSE": 35.4443,
            "R²": 0.1881,
        },
    ]
)

st.dataframe(
    comparison.style.format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("How was the LSTM benchmark evaluated?"):
    st.write(
        "The LSTM uses the same four-week historical lookback and the same "
        "20 regional disease-count series as the ST-GNN. It was evaluated "
        "on the same chronological held-out test period using MAE, RMSE "
        "and R². It is included as a baseline comparison and is not used "
        "to generate the dashboard's current forecast."
    )


# Regional overview
# ---------------------------------------------------------------------
st.markdown("### 🗺️ Regional overview")

regional = overview.copy()
regional["Reported now"] = regional["Reported now"].round(0).astype(int)
regional["Forecast next week"] = regional["Forecast next week"].round(1)
regional["Change (%)"] = regional["Change (%)"].round(1)
regional["Trend"] = [
    f"{trend_icon(t)} {t}" for t in regional["Trend"]
]

st.dataframe(
    regional.sort_values("Forecast next week", ascending=False),
    use_container_width=True,
    hide_index=True,
)

increasing = overview[
    overview["Trend"] == "Increasing"
].sort_values("Change (%)", ascending=False).head(5)

if not increasing.empty:
    st.markdown("#### Regions with an expected increase")
    for _, row in increasing.iterrows():
        st.write(
            f"**{row['Region']}** — "
            f"{row['Reported now']:.0f} reported → "
            f"{row['Forecast next week']:.1f} estimated "
            f"({row['Change (%)']:+.1f}%)"
        )

# ---------------------------------------------------------------------
# Minimal explanation of AI
# ---------------------------------------------------------------------
st.markdown("### ℹ️ How was this estimate made?")

with st.expander("Simple explanation"):
    st.write(
        "The system looks at recent reported chickenpox cases and the "
        "relationships between the 20 regions. It learns patterns from "
        "historical data and uses those patterns to estimate the number "
        "of reported cases one week ahead."
    )
    st.write(
        "**Input:** recent disease history.  "
        "**Output:** estimated reported cases next week."
    )

with st.expander("For technical users"):
    st.write(
        f"Disease-only V4 ST-GNN • 20 regions • "
        f"{metadata['lookback']}-week history • "
        f"{metadata['train_samples']} training samples"
    )
    st.write(
        f"Checkpoint seed {metadata['seed']} • "
        f"MAE {metadata['mae_cases']:.4f} cases • "
        f"RMSE {metadata['rmse_cases']:.4f} cases • "
        f"R² {metadata['r2_cases']:.4f}"
    )

with st.expander("Data and limitations"):
    st.write(
        "The deployment uses the original weekly Hungarian chickenpox "
        "case-count dataset. Four recent weeks are used to estimate the "
        "following week."
    )
    st.write(
        "Forecasts are estimates and may differ from subsequently reported "
        "cases. This research demonstration should not replace clinical "
        "judgment or official surveillance."
    )

st.caption(
    "OneHealth research demonstration • Forecasts are model estimates, "
    "not clinical diagnoses or treatment recommendations."
)
