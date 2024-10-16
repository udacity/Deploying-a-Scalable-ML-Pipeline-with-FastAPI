# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
-**Model Name:** Income Classification Model
-**Algorithm:** Random Forest Classifier
-**Version:** 1.0
-**Input Features:**
    -**Categorical:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
    - **Continuous:** `age`, `hours-per-week`, `capital-gain`, `capital-loss`
- **Output:** Binary classification (`>50K` or `<=50K` income)

## Intended Use
The model is designed to predict whether an individual's income exceeds $50,000 per year based on demographic and economic factors. This can be used for analyzing income distribution patterns and identifying factors associated with higher incomes. It is intended for educational and analytical purposes only, not for making decisions that directly affect individuals' lives.

## Training Data
The model was trained on a publicly available census dataset, containing demographic and economic data of individuals. The dataset consists of approximately 32,000 samples, with features ranging from age and workclass to educational background and marital status.

- **Data Preprocessing:** 
  - Categorical features were one-hot encoded.
  - Continuous features were normalized for consistency.

## Evaluation Data
The evaluation was performed on a test set, consisting of 20% of the dataset, which was separated from the training data using a random split. The model was evaluated using metrics such as Precision, Recall, and F1 Score.

### Overall Performance on Test Data:
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The following metrics were used to evaluate the model:
- **Precision:** Measures the accuracy of positive predictions made by the model.
- **Recall:** Measures the ability of the model to identify all relevant instances.
- **F1 Score:** A balanced metric that combines Precision and Recall.

## Ethical Considerations
- **Bias:** The model may exhibit bias based on demographic features like race, gender, and work class. Since the model is trained on historical data, any biases present in the data may be reflected in the model's predictions.
- **Privacy:** No personally identifiable information (PII) was included in the dataset, but caution should be taken if applying this model to real-world data containing PII.
- **Fairness:** Efforts were made to analyze model performance across different slices of data to ensure fair outcomes across demographic groups. However, users should remain aware of potential biases and not use this model for decisions affecting individuals' lives without further fairness testing.

## Caveats and Recommendations
- **Generalization:** The model was trained on a specific dataset and may not generalize well to populations that differ significantly from the training data.
- **Use with Caution:** The model's predictions should be used for analysis and educational purposes only, not for making critical decisions. Further testing and retraining may be necessary for real-world applications.
- **Bias Mitigation:** It is recommended to regularly monitor and evaluate the model's performance on different demographic groups and take steps to mitigate any identified biases.
