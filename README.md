# defect_prediction
software bug predicton using machine learning algorithms

## introduction
 Software bug is one of the major issues in a computer industry. It is always desirable to have minimum software bug and software system to reach at the maximum accuracy level. machine learning can plays a vary important role in software bug prediction.
 
 In this we use various classification techniques like-support vector machines(svm),naive bayes,knn to evaluate whether a module is 
 defect prone or not.
 
 ## dataset collection
 
 For the experiment/analysis dataset is collected from the open source Promise repository which is authentic and publically available.
 
 ## data cleaning
 after collection of data it is very important to preprocess or clean data before applying it to our model.
 ### various steps of prepocessing:
 *changing categorical variable into numerical variable - since in this our dependent variable is categorical so we have to convert this.
 
from sklearn.preprocessing import LabelEncoder
encoder_y=LabelEncoder()
y=encoder_y.fit_transform(y)
 
 
 
 
