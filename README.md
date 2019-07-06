# defect_prediction
software bug predicton using machine learning algorithms

# dependencies
* pandas
* numpy
* matplotlib
* sklearn

## introduction
 Software bug is one of the major issues in a computer industry. It is always desirable to have minimum software bug and software system to reach at the maximum accuracy level. machine learning can plays a vary important role in software bug prediction.
 
 In this we use various classification techniques like-support vector machines(svm),naive bayes,knn to evaluate whether a module is 
 defect prone or not.
 
 ## dataset collection
 
 For the experiment/analysis dataset is collected from the open source Promise repository which is authentic and publically available.
 
 ## data cleaning
 after collection of data it is very important to preprocess or clean data before applying it to our model.
 
 ### various steps of prepocessing:
 * changing categorical variable into numerical variable : since in this dataset our dependent variable is categorical so we have to convert this.
 * splitting our dataset: we have to split our dataset because we want to both train as well as train our dataset.common practice is 
 to use 70 percent of data for training and remaining for testing purpose.
 * feature scaling: feature scaling is important when all/some independent variables of our dataset is not in same range. so we have to
 make them into same range by using mathemetical tools like standardisation and normalisation.
 
## feature extraction
feature extraction is important beacause if the numbers of independent variables are more than optimal than machine learning algorithm show a decrease of accuracy as well as it becomes complex.
here I uses principal component analysis as a tool to extract features or attributes with high variance.

## applying dataset to our machine learning model
here I uses three classification algorithms namely support vector machines(svm),naive bayes,knn for prediction.

## evaluation of model
after using different models for our problem it is important to evaluate their performance.Here I uses various performance metrics to 
evaluate perforamnce namely:
* accuracy_score
* precision(measure of false positive)
* recall ( measure of false negative)
* fscore( harmonic mean of precision and recall)
* roc_auc curve





 
 
 
 
