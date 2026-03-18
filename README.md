---
title: Singapore HDB Resale Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
license: mit
---

# 🏠 Singapore HDB Resale Price Predictor

A machine learning web app that estimates HDB resale flat prices in Singapore and provides a 3-year appreciation forecast (2025–2028).

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Regressor |
| R² Score | **0.9675** |
| MAE | **~$22,000 SGD** |
| Training data | 181,262 transactions (2017–2024) |

## Features

- Select town, flat type, flat model, floor area, storey, and lease commence year
- Instant price estimates for the current year plus 3 years ahead
- Bar chart showing price trajectory with appreciation percentage
- Remaining lease auto-computed from lease commence year

## Input Features (9 model features)

| Feature | Description |
|---------|-------------|
| `town_encoded` | Town name encoded to integer (26 towns) |
| `flat_type_encoded` | 1 ROOM=1 … MULTI-GENERATION=7 |
| `floor_area_sqm` | Floor area in square metres (30–280) |
| `storey_midpoint` | Floor level (1–50) |
| `remaining_lease_years` | Auto-computed: 99 − (year − lease_commence_year) |
| `lease_commence_date` | Year lease began (1960–2024) |
| `flat_model_encoded` | Flat model encoded to integer (21 models) |
| `year` | Prediction year |
| `month_num` | Current calendar month |

## Run Locally

```bash
git clone https://github.com/<your-username>/hdb-price-predictor
cd hdb-price-predictor
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

> **Note:** `xgb_model.json` (30MB) is tracked with Git LFS. Run `git lfs pull` if the file appears as a pointer.

## Data Source

Singapore HDB Resale Flat Prices dataset from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view).
Transactions from January 2017 to December 2024.

## Full Analysis

The full exploratory data analysis, feature engineering, and model training notebooks are available in the [CA6002 assignment repository](https://github.com/<your-username>/CA6002).

## Disclaimer

This model provides statistical estimates based on historical transaction data. Predictions are **not** official HDB valuations. Actual resale prices depend on market conditions, flat condition, negotiation, and other factors. Consult HDB or a licensed appraiser for official valuations.

## License

MIT
