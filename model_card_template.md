# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: FastAPI using Random Forest Classifier on Census Data
Version: 1.0
Date: 7/16/2024
Author: Ryan Merrithew

## Intended Use
The model created is designed to predict salary levels based on demographic features from the U.S. Census data. It is inteded to be used by data analysts and those interested in researching more.

## Training Data
The model was trained on a dataset consisting of demographic information, including features such as work class, education level, marital status, occupation, and more. The training dataset includes approximately 32500 samples.

## Evaluation Data
The evaluation data used to assess model performance was a separate portion of the U.S. Census data not used in training, ensuring that the model generalizes well to unseen data.

## Metrics
Here are the metrics that were used to evaluate success for this model: 
Precision: 0.7424
Recall: 0.6346
F1 Score: 0.6843

This is a good sign as the model is accurate but there is still plenty of room for improvement.

## Ethical Considerations
Users should be mindful of potential biases in the training data, as the model's predictions may reflect underlying societal biases present in the dataset. Care should be taken when interpreting results, particularly for sensitive demographic groups.

## Caveats and Recommendations
The model may underperform for certain demographic groups; additional data may be required to improve performance. It is also recommended to continuously monitor model performance and retrain as necessary with updated data to maintain accuracy.