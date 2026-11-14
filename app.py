import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "artifacts/car_model.pkl"
if not os.path.exists(MODEL_PATH):
    import train  # Auto-trains if artifacts/car_model.pkl does not exist yet

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

preprocessor = artifact["preprocessor"]
nn_model = artifact["nn_model"]
df = artifact["dataset"]


def find_cars(price_egp, transmission=None, design_type=None, cc=None, hp=None, torque_nm=None, tolerance_pct=0.15):
    # Filter within user's budget range (+/- tolerance, default 15%)
    low = max(50000, price_egp * (1 - tolerance_pct))
    high = min(4000000, price_egp * (1 + tolerance_pct))

    filtered = df[(df["price_egp"] >= low) & (df["price_egp"] <= high)].copy()

    if transmission and transmission != "Any":
        filtered = filtered[filtered["transmission"].str.lower() == transmission.lower()]
    if design_type and design_type != "Any":
        filtered = filtered[filtered["design_type"].str.lower() == design_type.lower()]

    # Fallback to NearestNeighbors if strict filter returns empty
    if filtered.empty:
        query_df = pd.DataFrame([{
            "price_egp": price_egp,
            "transmission": transmission if transmission and transmission != "Any" else "Automatic",
            "design_type": design_type if design_type and design_type != "Any" else "Sedan",
            "cc": cc or 1500,
            "hp": hp or 120,
            "torque_nm": torque_nm or 150,
        }])
        q_vec = preprocessor.transform(query_df)
        _, indices = nn_model.kneighbors(q_vec, n_neighbors=25)
        filtered = df.iloc[indices[0]].copy()

    # Rank closest matches by price & optional technical specs
    filtered["score"] = (filtered["price_egp"] - price_egp).abs() / max(price_egp, 1)
    if cc:
        filtered["score"] += (filtered["cc"] - cc).abs() / 1500.0
    if hp:
        filtered["score"] += (filtered["hp"] - hp).abs() / 150.0
    if torque_nm:
        filtered["score"] += (filtered["torque_nm"] - torque_nm).abs() / 200.0

    # Deduplicate by Brand + Model + Transmission + Design + Year for clean results
    results = (
        filtered.sort_values("score")
        .drop_duplicates(subset=["brand", "model", "year", "transmission", "design_type"])
        .head(30)
    )
    return results.drop(columns=["score"]).to_dict(orient="records")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    try:
        price_egp = float(data.get("price_egp", 500000))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid price_egp (50000 - 4000000) is required"}), 400

    transmission = data.get("transmission", "Any")
    design_type = data.get("design_type", "Any")
    cc = float(data["cc"]) if data.get("cc") not in (None, "", 0) else None
    hp = float(data["hp"]) if data.get("hp") not in (None, "", 0) else None
    torque_nm = float(data["torque_nm"]) if data.get("torque_nm") not in (None, "", 0) else None

    cars = find_cars(price_egp, transmission, design_type, cc, hp, torque_nm)
    return jsonify({
        "query_price_egp": price_egp,
        "total_matches": len(cars),
        "cars": cars
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=False)
