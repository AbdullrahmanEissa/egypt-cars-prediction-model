import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/egypt_cars_dataset.csv")

FEATURE_COLS = ["price_egp", "transmission", "design_type", "cc", "hp", "torque_nm"]
X = df[FEATURE_COLS]
y = df["price_egp"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["price_egp", "cc", "hp", "torque_nm"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["transmission", "design_type"]),
    ]
)

X_processed = preprocessor.fit_transform(X)

# 1. NearestNeighbors model to return all matching cars for a user's budget & specs
nn_model = NearestNeighbors(n_neighbors=min(50, len(df)), metric="euclidean")
nn_model.fit(X_processed)

# 2. RandomForest model to verify spec-to-price consistency
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_processed, y)
train_r2 = r2_score(y, rf_model.predict(X_processed))

artifact = {
    "preprocessor": preprocessor,
    "nn_model": nn_model,
    "rf_model": rf_model,
    "dataset": df,
}

with open("artifacts/car_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print(f"Training complete on {len(df)} Egyptian car listings.")
print(f"Model Fit Score (R2): {train_r2 * 100:.2f}% | Saved to artifacts/car_model.pkl")
