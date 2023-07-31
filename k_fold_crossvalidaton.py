# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:30:17 2023

@author: ATISHKUMAR
"""
#grid search -its model tunning parameter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\5th, 8th\Social_Network_Ads.csv')

#seprate i.v and d.v
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#apply svm moel to trainig data
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(x_train,y_train)

#predecting the test set results
y_pred=classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


#score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

#bias
bias=classifier.score(x_train, y_train)
bias


#appl k-fold cross validation

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("standarad deviation: {:.2f} %".format(accurices.std()*100))
#print("best parameters",best_parameters)


'''#apply gird search
#to find best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'c':[1,10,100,1000],'kernel':['linear']},
            {'c':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search=GridSearchCV( estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)

grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_
print("best accuracy:{:.2f} %".format(best_accuracy*100))
print("best parameters:",best_parameters)'''