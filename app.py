import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="OneHealth | Disease Forecasting", page_icon="🧬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"

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
    cases = np.asarray(data["FX"], dtype=float)
    if cases.ndim != 2 or cases.shape[1] != 20:
        raise ValueError(f"Unexpected dataset shape: {cases.shape}")
    return cases

st.title("🧬 OneHealth")
st.subheader("Spatio-Temporal Disease Forecasting")

st.info(
    "This deployed version is the OneHealth demonstration interface. "
    "The trained ST-GNN checkpoint has not yet been connected, so the "
    "current forecast is an explicitly labelled persistence baseline."
)

try:
    cases = load_data()
except Exception as exc:
    st.error(f"Could not load the public demonstration dataset: {exc}")
    st.stop()

left, right = st.columns([1, 2])

with left:
    region = st.selectbox("Select region", REGIONS)
    idx = REGIONS.index(region)
    series = cases[:, idx]
    recent = series[-4:]
    prediction = float(recent[-1])
    st.metric("Next-step baseline", f"{prediction:.2f}")
    st.caption("Baseline = most recent observed value")

with right:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="Observed")
    ax.scatter([len(series)], [prediction], marker="o", label="Next-step baseline")
    ax.set_title(f"Weekly chickenpox cases — {region}")
    ax.set_xlabel("Weekly observation index")
    ax.set_ylabel("Cases (dataset scale)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("### Recent observations")
recent_df = pd.DataFrame({"Lag": ["t-3", "t-2", "t-1", "t"], "Observed value": recent})
st.dataframe(recent_df, use_container_width=True, hide_index=True)

st.markdown("### Model status")
st.write(
    "The production interface is ready. The next step is to connect the trained "
    "OneHealth ST-GNN checkpoint, graph structure, normalization parameters, and "
    "exact preprocessing pipeline from the training notebook."
)

st.warning(
    "Research demonstration only. This application is not a clinical decision-support "
    "system and should not be used for medical or public-health decisions."
)
