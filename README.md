---
title: OneHealth Disease Forecasting
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# OneHealth — Disease Forecasting Demo

This is a deployable Hugging Face Spaces starter generated from the uploaded OneHealth notebook context.

## Included
- Gradio interface
- Region selector
- Historical chickenpox series visualization
- Explicit persistence-baseline forecast
- Dataset fetched from the public PyTorch Geometric Temporal repository

## Important model status
This starter does **not** load the trained OneHealth ST-GNN. The notebook references trained checkpoints and preprocessing artifacts, but those binary artifacts were not included with the notebook upload. The current forecast is a clearly labelled persistence baseline, not a trained-model prediction.

## Deploy
Create a new Hugging Face Space with the **Gradio** SDK, then upload `app.py`, `requirements.txt`, and this `README.md`.

Once the trained checkpoint, exact model class, graph, and normalization artifacts are recovered, replace the baseline inference path with the trained ST-GNN pipeline.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

This is a research demonstration, not a clinical decision-support tool.
