# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier designed to predict income level based on a variety of features. It was developed for the  Census Bureau.

## Intended Use
The model is intened for use in analytics. It should be applied to socioeconomic fields.
## Training Data
The training data consist of a Census dataset that includes age, work class, education, marital status, occupation, relationship, race, sex, and native country.
## Evaluation Data
The evaluation was conducted using a random split from the orginal data. This test data is just as diverse as the real world data. 
## Metrics
Overall Performance:

Precision: 0.7317
Recall: 0.6377
F1 Score: 0.6815

Performance on Data Slices:
Workclass: Private
Precision: 0.7259
Recall: 0.6371
F1 Score: 0.6786

Education: Bachelors
Precision: 0.7251
Recall: 0.7773
F1 Score: 0.7503



## Ethical Considerations
The diversity in the data set is reflective of the real world. Users of the model should be aware of the possiblity of residual biases. 


## Caveats and Recommendations

While it does create good results. The model should be not be used as a primary decision making tool. I would recommend continuing to train this model to relfect changing socioecconmic trends. 
