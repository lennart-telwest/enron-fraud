#!/usr/bin/python

import sys
import pickle
import numpy as np
from numpy import mean
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, SelectFpr
from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import cross_validation
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_email = ['to_messages', 'from_messages',  'from_poi_to_this_person',
           'from_this_person_to_poi', 'shared_receipt_with_poi']
# finance data
features_finance = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
             'deferral_payments', 'loan_advances', 'other', 'expenses',
             'director_fees', 'total_payments',
             'exercised_stock_options', 'restricted_stock',
             'restricted_stock_deferred', 'total_stock_value']
# all features
features_list = features_email + features_finance
# all features column names
features_column_names = ['poi'] + ['email_address'] + features_email + features_finance
# all features data type
features_dtype = [bool] + [str] + list(np.repeat(float, 19))

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# converting the data into a data frame
df = DataFrame.from_dict(data_dict, orient='index')

# reordering the columns
df = df.loc[:, features_column_names]

# converting the data type
for i in xrange(len(features_column_names)):
    df[features_column_names[i]] = df[features_column_names[i]].astype(features_dtype[i], errors='ignore')

### Task 2: Remove outliers
df = df[df.index != 'TOTAL']
df = df[df.index != 'THE TRAVEL AGENCY IN THE PARK']

df.loc['BELFER ROBERT', features_finance] = \
    [0, 0, 0, -102500, 0, 0, 0, 3285,
     102500, 3285, 0, 44093, -44093, 0]
df.loc['BHATNAGAR SANJAY', features_finance] = \
    [0, 0, 0, 0, 0, 0, 0, 137864, 0, 137864,
     15456290, 2604490, -2604490, 15456290]

### Task 3: Create new feature(s)
# calculate ratio
df['recieved_from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['sent_to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_receipt_with_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
# add labels to df
features_email_new = ['recieved_from_poi_ratio', 'sent_to_poi_ratio', 'shared_receipt_with_poi_ratio']
features_all = features_list + features_email_new

# log scaling
for f in features_list:
    df[f] = [np.log(abs(v)) if v != 0 else 0 for v in df[f]]
# replace null values
df.replace(0, np.NaN)
df.fillna(df.mean(), inplace=True)

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')


### Extract features and labels from dataset for local testing

#Select the labels
my_feature_list = ['poi'] + features_email + features_finance
# my_feature_list = ['poi'] + ['sent_to_poi_ratio'] + ['restricted_stock'] + ['exercised_stock_options'] + ['from_poi_to_this_person'] + ['shared_receipt_with_poi'] + ['bonus'] + ['long_term_incentive']
# print my_feature_list[1]


# Format and splot to labels and features
data = featureFormat(my_dataset, my_feature_list, remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=2000,gamma = 0.0001,random_state = 42, class_weight = 'auto')

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)

evaluate_clf(clf, features, labels)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
