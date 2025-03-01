# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Developed by: Stephen Byrd, Feburary 2025
- Model Type: This model uses a **Random Forest Classifier** for binary classification.
- Dataset: The model is trained on the **Adult Census Dataset** from the UCI Machine Learning Repository.

## Intended Use

- The model is intended to predict whether an individual earns more than $50,000 per year based on demographic features from the census data.

## Training Data

- Dataset Source: The data is extracted from the 1994 Census database.
- Features:
  - age
  - workclass
  - education
  - education-num
  - marital-status
  - occupation
  - relationship
  - race
  - sex
  - capital-gain
  - capital-loss
  - hours-per-week
  - native-country
- Target Label: 'salary' with values '>50K' (1) and '<=50K' (0).

## Evaluation Data

- Validation Method: The model is evaluated using a test dataset split from the original data.
- Metrics Used: Precision, Recall, F1 Score.

## Metrics

- Metrics Used: Precision, Recall, F1 Score.
- Model Performance:
  - Precision: **0.7419**
  - Recall: **0.6384**
  - F1 Score: **0.6863**

## Ethical Considerations

- The dataset may have biases towards certain demographics (e.g., more men than women, predominantly white individuals).
- No direct human life risks are associated with this model.

## Caveats and Recommendations

- Limitations: The model's performance could be improved with further tuning or using different classifiers.
- Future Work: Consider using techniques like SHAP for feature importance analysis or exploring other classification models.
