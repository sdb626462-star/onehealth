import json
import urllib.request
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"

REGIONS = [
    "Budapest", "Baranya", "Bács-Kiskun", "Békés",
    "Borsod-Abaúj-Zemplén", "Csongrád-Csanád", "Fejér",
    "Győr-Moson-Sopron", "Hajdú-Bihar", "Heves",
    "Jász-Nagykun-Szolnok", "Komárom-Esztergom", "Nógrád",
    "Pest", "Somogy", "Szabolcs-Szatmár-Bereg", "Tolna",
    "Vas", "Veszprém", "Zala"
]

def load_data():
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "OneHealth-demo"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    cases = np.asarray(data["FX"], dtype=float)
    if cases.ndim != 2 or cases.shape[1] != 20:
        raise ValueError(f"Unexpected dataset shape: {cases.shape}")
    return cases

def run_demo(region):
    try:
        cases = load_data()
    except Exception as exc:
        return None, f"Could not load public chickenpox dataset: {exc}", None

    # Dataset's canonical 20-node order is used; label names are a display convenience.
    idx = REGIONS.index(region)
    series = cases[:, idx]
    recent = series[-4:]
    prediction = float(recent[-1])  # transparent persistence baseline; not ST-GNN inference

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(series)), series, label="Observed")
    ax.scatter([len(series)], [prediction], label="Next-step persistence baseline")
    ax.set_title(f"Weekly chickenpox cases — {region}")
    ax.set_xlabel("Weekly observation index")
    ax.set_ylabel("Cases (dataset scale)")
    ax.legend()
    ax.grid(alpha=0.2)

    recent_df = pd.DataFrame({
        "Lag": ["t-3", "t-2", "t-1", "t"],
        "Observed value": recent
    })
    summary = (
        f"**OneHealth demo — {region}**\n\n"
        f"Most recent 4 values: {', '.join(f'{x:.2f}' for x in recent)}\n\n"
        f"Next-step persistence baseline: **{prediction:.2f}**\n\n"
        "**Important:** This is a baseline demonstration, not the trained OneHealth ST-GNN. "
        "The notebook was provided without its trained checkpoint and required preprocessing artifacts. "
        "No clinical or public-health decision should rely on this demo."
    )
    return fig, summary, recent_df

with gr.Blocks(title="OneHealth | Disease Forecasting") as demo:
    gr.Markdown("# OneHealth\n### Spatio-temporal disease forecasting — interactive demo")
    gr.Markdown(
        "Explore the public Hungarian chickenpox time series. This starter Space currently "
        "shows a transparent persistence baseline while the trained ST-GNN checkpoint is being connected."
    )
    region = gr.Dropdown(REGIONS, value=REGIONS[0], label="Select region")
    btn = gr.Button("Generate demo")
    plot = gr.Plot(label="Historical series and baseline")
    output = gr.Markdown()
    table = gr.Dataframe(label="Most recent four observations", interactive=False)
    btn.click(run_demo, inputs=region, outputs=[plot, output, table])
    demo.load(run_demo, inputs=region, outputs=[plot, output, table])

if __name__ == "__main__":
    demo.launch()
