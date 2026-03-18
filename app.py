import json
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import xgboost as xgb

# ── Load model and encoders at startup ──────────────────────────────────────
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

with open("town_encoder.json") as f:
    town_encoder = json.load(f)

with open("flat_model_encoder.json") as f:
    flat_model_encoder = json.load(f)

# ── Constants ────────────────────────────────────────────────────────────────
FLAT_TYPE_ENCODING = {
    "1 ROOM": 1,
    "2 ROOM": 2,
    "3 ROOM": 3,
    "4 ROOM": 4,
    "5 ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7,
}

TOWNS = sorted(town_encoder.keys())
FLAT_MODELS = sorted(flat_model_encoder.keys())
FLAT_TYPES = list(FLAT_TYPE_ENCODING.keys())

current_year = datetime.datetime.now().year
current_month = datetime.datetime.now().month


def predict_prices(town, flat_type, flat_model, floor_area, storey, lease_commence_year):
    """Run inference for 4 years (current + 3) and return chart + summary."""
    town_enc = town_encoder[town]
    flat_type_enc = FLAT_TYPE_ENCODING[flat_type]
    flat_model_enc = flat_model_encoder[flat_model]

    years = [current_year, current_year + 1, current_year + 2, current_year + 3]
    prices = []

    for yr in years:
        remaining_lease = 99 - (yr - lease_commence_year)
        features = np.array([[
            town_enc,
            flat_type_enc,
            floor_area,
            storey,
            remaining_lease,
            lease_commence_year,
            flat_model_enc,
            yr,
            current_month,
        ]], dtype=float)
        price = float(model.predict(features)[0])
        prices.append(max(price, 0))

    # ── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563EB"] + ["#F97316"] * 3
    bars = ax.bar([str(y) for y in years], prices, color=colors, width=0.55, edgecolor="white")

    for bar, price in zip(bars, prices):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5000,
            f"${price / 1000:.0f}k",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_title(f"Estimated Resale Price — {town} {flat_type}", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Price (SGD)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.set_ylim(0, max(prices) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#2563EB", label=f"{current_year} (current)"),
        plt.Rectangle((0, 0), 1, 1, color="#F97316", label="Projected"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, framealpha=0.7)
    plt.tight_layout()

    # ── Summary text ─────────────────────────────────────────────────────────
    current_price = prices[0]
    future_price = prices[-1]
    appreciation = ((future_price - current_price) / current_price) * 100

    summary = (
        f"**{current_year} Estimate:** ${current_price:,.0f}\n\n"
        f"**{current_year + 3} Projection:** ${future_price:,.0f}\n\n"
        f"**3-Year Appreciation:** {appreciation:+.1f}%\n\n"
        f"*Floor area: {floor_area} sqm · Storey: {storey} · Lease from: {lease_commence_year} "
        f"({99 - (current_year - lease_commence_year)} yrs remaining)*"
    )

    disclaimer = (
        "**Disclaimer:** This model was trained on HDB resale transactions from 2017–2024 "
        "(181,262 records). Predictions are statistical estimates only and do not constitute "
        "official valuations. Actual prices depend on market conditions, flat condition, and "
        "negotiation. Always consult HDB or a licensed appraiser for official valuations."
    )

    return fig, summary, disclaimer


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="HDB Resale Price Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🏠 Singapore HDB Resale Price Predictor
Estimate the resale value of an HDB flat and see a 3-year appreciation forecast.
Model: **XGBoost** trained on 181,262 transactions (2017–2024) · R² = 0.9675 · MAE ≈ $22k SGD
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            town = gr.Dropdown(choices=TOWNS, label="Town", value="QUEENSTOWN")
            flat_type = gr.Dropdown(choices=FLAT_TYPES, label="Flat Type", value="4 ROOM")
            flat_model = gr.Dropdown(choices=FLAT_MODELS, label="Flat Model", value="Model A")
            floor_area = gr.Slider(minimum=30, maximum=280, value=93, step=1, label="Floor Area (sqm)")
            storey = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Storey")
            lease_commence = gr.Slider(minimum=1960, maximum=2024, value=1985, step=1, label="Lease Commence Year")
            predict_btn = gr.Button("Predict Price", variant="primary", size="lg")

        with gr.Column(scale=2):
            chart = gr.Plot(label="Price Forecast")
            summary_md = gr.Markdown()
            disclaimer_md = gr.Markdown()

    predict_btn.click(
        fn=predict_prices,
        inputs=[town, flat_type, flat_model, floor_area, storey, lease_commence],
        outputs=[chart, summary_md, disclaimer_md],
    )

    gr.Examples(
        examples=[
            ["QUEENSTOWN", "4 ROOM", "Model A", 93, 10, 1985],
            ["CENTRAL AREA", "5 ROOM", "Apartment", 120, 25, 2000],
            ["PUNGGOL", "3 ROOM", "New Generation", 68, 8, 2015],
            ["BISHAN", "EXECUTIVE", "Maisonette", 146, 5, 1992],
        ],
        inputs=[town, flat_type, flat_model, floor_area, storey, lease_commence],
        outputs=[chart, summary_md, disclaimer_md],
        fn=predict_prices,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
