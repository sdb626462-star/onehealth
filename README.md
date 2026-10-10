# OneHealth

Streamlit research demonstration for disease-data exploration and model-output visualisation.

## Current interface

The application contains three distinct analytical sections:

1. **West Bengal disease surveillance exploration** — descriptive district summaries, reported disease counts, time trends, and temperature/precipitation/LAI context from the public EpiClim dataset.
2. **West Bengal dengue graph visualisation** — reads saved test predictions, district coordinates and DCMG graph-edge weights to display actual/predicted values, absolute error and model-derived graph associations.
3. **Hungary chickenpox forecasting benchmark** — uses a separate weekly dataset for 20 Hungarian regions and the disease-only ST-GNN model artifacts in `model_artifacts/`.

These are separate analyses. The Hungarian chickenpox benchmark is not a West Bengal forecast, and graph weights are not proof of disease transmission or causal effects.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app requires network access to download its public datasets and model/output artifacts from GitHub and Zenodo. If a remote source is unavailable, the relevant section may show an error.

## Research-use limitation

This is a research demonstration, not a clinical decision-support tool. Forecasts and descriptive counts should not be used for diagnosis or operational public-health decisions without independent data-quality checks and temporal validation against appropriate baselines.
