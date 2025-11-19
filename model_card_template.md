# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

---

## Model Details

This project uses a Random Forest classifier implemented with scikit learn.  
The purpose of the model is to predict whether an individual earns more than fifty thousand dollars per year.  
The model was trained locally using Python 3.10 as part of the Udacity Machine Learning DevOps project.  

The model is saved as `model/model.pkl` and the preprocessing objects (encoder and label binarizer) are stored in the same folder.

---

## Intended Use

This model is intended purely for educational and demonstration purposes.  
It shows how to build an end to end ML pipeline including data processing, model training, automated testing, performance evaluation, CI, and API deployment with FastAPI.

It is **not** intended for real world decision making in hiring, lending, housing, insurance, or other areas that affect people’s lives.

---

## Training Data

The model was trained on the provided **Census Income dataset**, which contains demographic and employment related features.  
Features include workclass, education, marital status, occupation, race, sex, and native country.

Eighty percent of the dataset is used for training.  
Categorical features are one hot encoded using the provided `process_data` function.

---

## Evaluation Data

Twenty percent of the dataset is held out as a test set.  
The same preprocessing pipeline (encoder and label binarizer) is applied to the test data.

Evaluation includes both overall performance on the test set and performance on slices of data across individual categories (workclass, sex, race, etc.).  
Slice performance is saved to `slice_output.txt`.

---

## Metrics

Overall performance on the held out test set:

- **Precision:** 0.7419  
- **Recall:** 0.6384  
- **F1 Score:** 0.6863  

Slice performance varies across groups. Examples include:

- **Sex = Male:** Higher recall and F1 compared to female  
- **Workclass = Private:** Moderate precision and recall  
- **Education levels:** Significant variation depending on category  

These results indicate that model performance is not uniform across demographic groups.

---

## Ethical Considerations

The Census dataset contains real world demographic information and may reflect historical social biases.  
A model trained on such data can inherit or amplify these biases.  
Because of this, the model should **not** be used for any real decision making that can affect individuals.

Potential risks include:

- Unequal performance for different demographic groups  
- Unintended discrimination if used in hiring or income related screening  
- Misinterpretation of model predictions due to limited context  

Care must be taken to avoid misuse.

---

## Caveats and Recommendations

- The model is not tuned or optimized; default Random Forest parameters are used.  
- Data is outdated and does not reflect current populations.  
- Model should be used only for learning purposes.  
- If extended further, a fairness analysis, hyperparameter tuning, and updated datasets are recommended.  

The model should always be accompanied by transparency about its limitations and intended educational use.
