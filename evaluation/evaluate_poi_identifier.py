#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import matplotlib.pyplot as plt
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from eval_metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]
class_names = "poi"
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!

### decision tree classifier
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)
print("The accuracy score is: " + str(accuracy))

cnf_matrix = confusion_matrix(pred, labels_test)
tn, fp, fn, tp = cnf_matrix.ravel()
print pred
print labels_test
print "True Negatives: " + str(tn)
print "False Negatives: " + str(fn)
print "True Positives: " + str(tp)
print "False Positives: " + str(fp)
print "Precision Score: " + str(precision_score(labels_test, pred))
print "Recall Score: " + str(recall_score(labels_test, pred))
