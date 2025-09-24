import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (compute_model_metrics, inference, load_model,
                      performance_on_categorical_slice, save_model,
                      train_model)

# ----------------------------
# Paths & data
# ----------------------------
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
print(f"[INFO] Loading: {data_path}")
data = pd.read_csv(data_path)

# ----------------------------
# Split
# ----------------------------
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data["salary"]
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# ----------------------------
# Process
# ----------------------------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

print(f"[INFO] X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"[INFO] X_test : {X_test.shape}  | y_test : {y_test.shape}")

# ----------------------------
# Train & save artifacts
# ----------------------------
model = train_model(X_train, y_train)

model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
print(f"[INFO] Saved model  -> {model_path}")
print(f"[INFO] Saved encoder-> {encoder_path}")

# ----------------------------
# Reload & inference
# ----------------------------
model = load_model(model_path)
preds = inference(model, X_test)

# ----------------------------
# Global metrics
# ----------------------------
p, r, fb = compute_model_metrics(y_test, preds)
print(f"[METRICS] Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# ----------------------------
# Slice metrics
# ----------------------------
slice_file = os.path.join(project_path, "slice_output.txt")
with open(slice_file, "w") as f:
    print("[SLICE] Writing per-slice metrics...", file=f)

for col in cat_features:
    # use dropna() so we don’t compute on NaN slices
    for slicevalue in sorted(test[col].dropna().unique()):
        count = test[test[col] == slicevalue].shape[0]
        # function expected to compute masking internally and return (p, r, f1)
        p, r, fb = performance_on_categorical_slice(
            test, y_test, preds, feature=col, slice_value=slicevalue
        )
        with open(slice_file, "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n", file=f)

print(f"[INFO] Slice metrics written to {slice_file}")
