import os

from ml.model import train_and_save_final_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = "/home/mha2112/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
#data = None # your code here
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
#X_train, X_test, y_train, y_test = train_test_split
#train, test = None, None# Your code here
#train, test = train_test_split(data, test_size=.20, random_state=42)
###K-fold
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


k=5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
#some scores-precision_score, recall_scores, f1_scores
precision_score=[]
recall_scores=[]
f1_scores=[]

#train and eval
for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    train= data.iloc[train_idx]
    test= data.iloc[test_idx]
    
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )



# TODO: use the train_model function to train the model on the training dataset
#model = None # your code here
    model = train_model(X_train, y_train)

# save the model and the encoder
#model_path = os.path.join(project_path, "model", "model.pkl")
#save_model(model, model_path)
#encoder_path = os.path.join(project_path, "model", "encoder.pkl")
#save_model(encoder, encoder_path)

# load the model
#model = load_model(
    #model_path
#) 

# TODO: use the inference function to run the model inferences on the test dataset.
#preds = None # your code here
    preds = inference(model, X_test)


# Calculate and print the metrics
    p, r, fb = compute_model_metrics(y_test, preds)
    precision_score.append(p)
    recall_scores.append(r)
    f1_scores.append(fb)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")




# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
    for col in cat_features:
    # iterate through the unique values in one categorical feature
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
            # your code here
            # use test, col and slicevalue as part of the input
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label='salary',
                encoder=encoder,
                lb=lb,
                model=model
            )
            with open("slice_output.txt", "a") as f:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

#metrics for folds
print(f"\n=== Average Metrics Across Folds ===")
print(f"avg precision: {np.mean(precision_score): .4f}")
print(f"avg recall: {np.mean(recall_scores): .4f}")
print(f"avg f1_score: {np.mean(f1_scores): .4f}")

#added because of k-fold model            
final_model, encoder, lb = train_and_save_final_model(
    data=data,
    categorical_features=cat_features,
    label="salary",
    model_dir=os.path.join(project_path, "model")
)
