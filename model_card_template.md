# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Salary prediction from census data using a random forest classifier. Random forest uses multiple decision trees to train the data that improves training efficiency being an ensemble. 
OneHotEncoding is used for the categorical column information and the labelBinarizer for the Salary column.

## Intended Use
Salary prediction from census data shows what common categories make less than or greater than 50K a year. 

## Training Data
The model used k-fold validation with standard k=5 parameter. Consistency metrics are done on the slicing of the k-folds.
## Evaluation Data
When the model predicts the salary to be over 50K with applied variables at testing it is 73% of time, correct.
From the slice_output.txt file there are some categories that have a higher count creating metrics that come together, but the under represented categories with few counts have a high metric in one score and a low in the others:

relationship: Own-child, Count: 1,019
Precision: 1.0000 | Recall: 0.1765 | F1: 0.3000

the own=child category has a high count and a low recall and f1
everyone in this category did make over 50K, when predicting over 50K it is 100% right. But it still underpredicts with 1,019 samples. 

native-country: Cambodia, Count: 3
Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000

Native country being Cambodia only has 3 people representing that country so Precision is high but recall and f1 are low. The sample is small. The test data found 1 that made over 50K from Cambodia, but recall and f1 are zero showing not definite or confident results for this categoric data. 


## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision, Recall and F1 score were used to test the model with the following values:

Precision: 0.7451 | Recall: 0.6308 | F1: 0.6832
Precision: 0.7087 | Recall: 0.6164 | F1: 0.6593
Precision: 0.7346 | Recall: 0.6180 | F1: 0.6713
Precision: 0.7475 | Recall: 0.6222 | F1: 0.6791
Precision: 0.7453 | Recall: 0.6307 | F1: 0.6832

The average of these scores were calculated as:
avg precision:  0.7363
avg recall:  0.6236
avg f1_score:  0.6752

Precision being what values were actually correcty or true positive is 73%
recall or sensitivity of values that could be missed is 62%
F1 score the balance between both precision and recall is 67%

## Ethical Considerations
The model will predict salary based on the other categorical information and is using performance slicing to id bias or gender, sex, race. 

## Caveats and Recommendations
The missing values could even be question marks: occupation: ?, Count: 389
Precision: 0.6923 | Recall: 0.4286 | F1: 0.5294, this slice_output found 39 question marks for the census question occupattion. The data is not clean and can create distraction from the true numbers or data analyzed. 