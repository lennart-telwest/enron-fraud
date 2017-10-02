#!/usr/bin/python

import sys
import pickle
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Selecting the features that will be used.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# email data
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
df = df.ix[:, features_column_names]

# converting the data type
for i in xrange(len(features_column_names)):
    df[features_column_names[i]] = df[features_column_names[i]].astype(features_dtype[i], errors='ignore')

# Replace missing data with 0
for f in features_finance:
    df.loc[df[f].isnull(), f] = 0

"""
Data Modification
"""
# Remove invalid data points
df = df[df.index != 'TOTAL']
df = df[df.index != 'THE TRAVEL AGENCY IN THE PARK']

"""
Basic Exploratory Analysis
"""
print(
"""
##############################
Dataset Shape:
##############################

""")
print df.shape
print(
"""

##############################
% of missing values:
##############################

""")
print df.isnull().sum() / df.shape[0]
print(
"""

##############################
Positional Parameters:
##############################

""")
print df.describe()
print(
"""

##############################
Correlation coefficient:
##############################

""")
print df.corr()


### Task 3: Create new feature(s)
"""
Feature Engineering - Ratio of Email
"""

df['recieved_from_poi_ratio'] = \
    df['from_poi_to_this_person'] / df['to_messages']
df['sent_to_poi_ratio'] = \
    df['from_this_person_to_poi'] / df['from_messages']
df['shared_receipt_with_poi_ratio'] = \
    df['shared_receipt_with_poi'] / df['to_messages']


# Update column definition
features_email_new = ['recieved_from_poi_ratio', 'sent_to_poi_ratio',
               'shared_receipt_with_poi_ratio']
features_all = features_list + features_email_new

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
pipe = make_pipeline(
          Imputer(axis=0, copy=True, missing_values='NaN',
                  strategy='median', verbose=0),
          ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
                               criterion='gini', max_depth=None,
                               max_features='sqrt', max_leaf_nodes=None,
                               min_samples_leaf=3, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=30,
                               n_jobs=-1, oob_score=False,
                               random_state=20160308, verbose=0,
                               warm_start=False))

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, df.to_dict(orient='index'), ['poi'] + features_all)
