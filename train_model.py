import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census data
data = pd.read_csv("data/census.csv")

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Define categorical features
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

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = train_model(X_train, y_train)

# Save model and encoder
model_path = os.path.join("model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join("model", "encoder.pkl")
save_model(encoder, encoder_path)

# Load model for inference
model = load_model(model_path)

# Make predictions
preds = inference(model, X_test)

# Calculate metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute performance on slices
# Compute performance on slices
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        precision, recall, fbeta = performance_on_categorical_slice(
            test, 
            col, 
            slicevalue, 
            cat_features, 
            "salary", 
            encoder, 
            lb, 
            model
        )
        
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}", file=f)

