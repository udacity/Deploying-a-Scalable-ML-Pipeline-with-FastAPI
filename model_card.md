## Model Details
Here in this script, we use sklearn RandomForestClassifier for our model—a random forest classifier. 
A random forest is a meta-estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree. Here is a link to more documentation on sklearns RandomForestClassifier https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.

Here we also looked into some additional models that did not perform as well we tested a LogisticRegression model which returned an f1 score of 0.4066 so we went with the random forest classifier model.
## Intended Use
The intended use of this model is to be used on the census data to download the census data or review more information about the data you can visit this site. https://archive.ics.uci.edu/dataset/20/census+income.
## Training Data
Here we slice information 20% of the population was used to train this model we see the train information here
train_test_split(data, test_size=0.2, random_state=42). We used these category features "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country",
## Evaluation Data
Our evaluation data can be found in the completed output file. slice_output.txt.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Here are the metrics for our models and the performance.
Precision: 0.8539 | Recall: 0.4316 | F1: 0.5734
## Ethical Considerations
This model takes into account the census details which include a few sensitive information fields such as workclass, education, marital-status, occupation, relationship, race, sex,      and native-country to predict the salary of individuals this model should be further investigated to ensure there are no biases in the representation of the detail in the census details where one category maybe unfairly predicted or has biases in the data itself.
## Caveats and Recommendations
Additional testing should be considered when using this model to ensure its accuracy of the model, There could be additional factors or dimensions that were not included in the dataset that might give additional insight or reduce the biases in the data.
