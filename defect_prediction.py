# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:35:35 2019

@author: saurabh kumar jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
data=pd.read_excel('defect_prediction.xlsx',sheet_name='Sheet1')

X=data.iloc[:,:28]
y=data.iloc[:,-1]

#changing categorical into numerical values
from sklearn.preprocessing import LabelEncoder
encoder_y=LabelEncoder()
y=encoder_y.fit_transform(y)


#preprocessing part
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#
#principal component analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_



#fitting our model with k-nearest-neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier1=KNeighborsClassifier()
classifier1.fit(X_train,y_train)

#predicting output
y_pred_knn=classifier1.predict(X_test)

#accuracy measurement
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred_knn)
accuracy_score_knn=metrics.accuracy_score(y_test,y_pred_knn)
print(cm)
print("accuracy_score",accuracy_score_knn)

#for seeing macro average of precision
precision_knn=metrics.precision_score(y_test,y_pred_knn,average='macro')
print("precision=",precision_knn)

#for seeing macro average of recall
recall_knn=metrics.recall_score(y_test,y_pred_knn,average='macro')
print("recall=",recall_knn)

#for seeing macro average of f-score
fscore_knn=metrics.f1_score(y_test,y_pred_knn,average='macro')
print("f-score=",fscore_knn)



#fitting our model with svm
from sklearn.svm import SVC
classifier2=SVC(kernel='rbf',random_state=0)
classifier2.fit(X_train,y_train)

#predicting output
y_pred_svm=classifier2.predict(X_test)

#accuracy measurement
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred_svm)
accuracy_score_svm=metrics.accuracy_score(y_test,y_pred_svm)
print(cm)
print(accuracy_score_svm)

#for seeing macro average of precision
precision_svm=metrics.precision_score(y_test,y_pred_svm,average='macro')
print("precision=",precision_svm)

#for seeing macro average of recall
recall_svm=metrics.recall_score(y_test,y_pred_svm,average='macro')
print("recall=",recall_svm)

#for seeing macro average of f-score
fscore_svm=metrics.f1_score(y_test,y_pred_svm,average='macro')
print("f-score=",fscore_svm)



#fitting model with naive bayes
from sklearn.naive_bayes import GaussianNB
classifier3=GaussianNB()
classifier3.fit(X_train,y_train)

#predicting output
y_pred_nb=classifier3.predict(X_test)

#accuracy measurement
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred_nb)
accuracy_score_nb=metrics.accuracy_score(y_test,y_pred_nb)
print(cm)
print(accuracy_score_nb)

#for seeing macro average of precision
precision_nb=metrics.precision_score(y_test,y_pred_nb,average='macro')
print("precision=",precision_nb)

#for seeing macro average of recall
recall_nb=metrics.recall_score(y_test,y_pred_nb,average='macro')
print("recall=",recall_nb)

#for seeing macro average of f-score
fscore_nb=metrics.f1_score(y_test,y_pred_nb,average='macro')
print("f-score=",fscore_nb)

#by applying k fold validation
from sklearn.model_selection import cross_val_score
score_knn=cross_val_score(classifier1,X,y,cv=10,scoring='roc_auc').mean()
#score_svm=cross_val_score(classifier2,X,y,cv=10,scoring='roc_auc').mean()
#score_nb=cross_val_score(classifier3,X,y,cv=10,scoring='roc_auc').mean()

#visualising plots
accuracy=np.array([accuracy_score_knn,accuracy_score_svm,accuracy_score_nb])
precision=np.array([precision_knn,precision_svm,precision_nb])
recall=np.array([recall_knn,recall_svm,recall_nb])
fscore=np.array([fscore_knn,recall_svm,fscore_nb])
x=np.arange(len(accuracy))

import matplotlib.patches as mpatches
bar_width=0.15
plt.bar(x,accuracy,width=bar_width,color='green',zorder=2)
plt.bar(x+bar_width,precision,width=bar_width,color='orange',zorder=2)
plt.bar(x+bar_width*2,recall,width=bar_width,color='red',zorder=2)
plt.bar(x+bar_width*3,fscore,width=bar_width,color='purple',zorder=2)


#for labeling part
plt.xticks(x+bar_width*1.5,['knn','svm',"naive_bayes"])
plt.title('evaluation matrics')

#for making patches
green=mpatches.Patch(color='green',label='accuracy')
orange=mpatches.Patch(color='orange',label='precision')
red=mpatches.Patch(color='red',label='recall')
purple=mpatches.Patch(color='purple',label='fscore')
plt.legend(handles=[green,orange,red,purple])
plt.ylim(0,1.5)

#grid
plt.grid(axis='y')

plt.show()






"""

#predicting probability
y_prob=classifier1.predict_proba(X_test)[:,1]
plt.hist(y_prob,bins=8)
plt.show()

#adjusting threshold value
from sklearn.preprocessing import binarize
y_pred_class=binarize([y_prob],0.3)[0]
cm=metrics.confusion_matrix(y_test,y_pred_class)
accuracy_score=metrics.accuracy_score(y_test,y_pred_class)
print(cm)
print(accuracy_score)"""



