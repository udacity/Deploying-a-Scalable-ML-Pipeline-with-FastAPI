# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Random Forest Classifier
Version: v1.0
Type: Supervised Classification (Binary Classification)
Model Architecture: Random Forest with hyperparameter tuning (n_estimators: 100-200, max_depth: None, 10, 20) using GridSearchCV
Training Time: Approximately 2 hours (estimated)
Last Trained: 2025-05-04
## Intended Use
The model is designed to predict whether an individual's income exceeds $50k per year, based on demographic and employment-related features from U.S. Census data. Intended for educational purposes, especially for demonstrating machine learning workflows and deployment pipelines. 

## Training Data
Source: UC Irvine Adult Census Income Dataset (loaded from census.csv)
Features:
    -Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
    -Numerical: age, hours-per-week, education-num, capital-gain, capital-loss, etc
Target Label: salary (<=50k, >50k)
Preprocessing:
    -OneHotEncoding for categorical variables
    -LabelBinarizer for the target variable
    -Train-test split: 80/20

## Evaluation Data
The test dataset is a 20% hold-out sample from the original dataset
Used to evaluate generalization performance and perform slice-based fairness analysis

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7866
Recall: 0.6149
F1 Score: 0.6902

These results suggest that the model strikes a moderate balance between identifying positive cases (recall) and minimizing false positives (precision)

See slice_output.txt

## Ethical Considerations
Bias & Fairness: The model may reflect historical biases present in the U.S. Census dataset. For instance, features such as race and gender could lead to disparate performance on different groups.

Interpretability: Random Forests are not easily interpretable by default, which may limit their use in regulated environments.

Use Limitations: This model should not be used for automated income predicitions affecting real-world decisions without additional audits and mitigation techniques.

Data Privacy: The dataset does not contain personally identifiable information (PII), but proper care should be taken if extended to real-world data.

## Caveats and Recommendations
Performance may vary significantly across different population subgroups. Always check model performance using slice-based analysis.
Further model robustness checks are recommended before deployment.
For production use, consider model interpretability enhancements such as SHAP or LIME for local explanations. 
LIME https://lime-ml.readthedocs.io/en/latest/
SHAP https://shap.readthedocs.io/en/latest/ 

