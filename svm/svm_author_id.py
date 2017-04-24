#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
#########################################################
### your code goes here ###
print("start svc")
clf = SVC(C=10000, kernel='rbf')
print("defined svc")
# train the model
print("start train")
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")
#predict
print("start predict")
t0 = time()
prediction = clf.predict(features_test)
print("training time: ", round(time()-t0, 3), "s")

print(prediction[10], prediction[26], prediction[50])
num_chris_mails = 0
num_sarah_mails = 0
for x in prediction:
    if x == 1:
        num_chris_mails += 1
    else:
        num_sarah_mails += 1

print(num_chris_mails, " of the predicted mails are labeled 'Chris' ", num_sarah_mails, " are labeled 'Sarah'") 

accuracy = accuracy_score(prediction, labels_test)
precision, recall, thresholds = precision_recall_curve(prediction, labels_test)
print(precision_recall_curve)
print("The accuracy score is: ", str(accuracy))
#########################################################


