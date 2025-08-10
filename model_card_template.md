# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This classification model was trained using the 1994 Census Bureau dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The goal is to predict whether an individual’s annual income exceeds $50,000 based on a set of demographic and socio-economic features, including:
* Sex
* Race
* Marital status
* Age
* Native country
* Education
* Relationship status
* Occupation
* Hours worked per week
* Work class
* Capital gain
* Capital loss

## Intended Use
Primary use: Predicting income category for individuals based on demographic and economic data.
Not intended for: Making real-world financial, hiring, or legal decisions without thorough fairness and bias evaluation.

## Training Data
Source: 1994 Census Bureau dataset (UCI Machine Learning Repository).
Size: 48,842 records after preprocessing.

## Evaluation Data

## Metrics
Precision : 0.7807 | Recall: 0.5379 | F1: 0.6369

## Ethical Considerations
Dataset reflects social and economic patterns from 1994, which may not represent current demographics or job markets.
Potential bias in predictions related to sensitive attributes such as race, sex, or marital status.
Misuse could perpetuate existing inequalities if deployed in sensitive decision-making contexts.

## Caveats and Recommendations
Model performance may degrade on modern census or employment datasets without retraining.
Bias analysis should be conducted before deployment.
Should not be the sole decision-making tool in critical domains such as hiring or lending.

