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

#project_path = '/home/lylewilliams/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/'
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print('this is the path', data_path)
data = pd.read_csv(data_path, header=0)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2, random_state=42)

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

X_train, y_train, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=True,
    encoder=None,
    lb=None,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,  # Pass the encoder object used during training
    lb=lb,
)
print('model set xtran ytrain')
model = train_model(X_train, y_train)

# save the model and the encoder
print('this is still the project path', project_path)
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
print('this is the model path',model_path)
model = load_model(
    model_path
) 

print('before preds')
preds = inference(model, X_test)
print('after preds')
print(preds)

print('before compute')
# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")
print('after compute')

# iterate through the categorical features
for col in cat_features:
    print(col)
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(test, col, slicevalue, cat_features, "salary", encoder, lb, model)
        with open(project_path + 'slice_output.txt', "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)