import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas import DataFrame

"""
Data Structure Definition
"""
# Feature names - email data
F_EMAIL = ['to_messages', 'from_messages',  'from_poi_to_this_person',
           'from_this_person_to_poi', 'shared_receipt_with_poi']
# Feature names - finance data
F_FINANCE = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
             'deferral_payments', 'loan_advances', 'other', 'expenses',
             'director_fees', 'total_payments',
             'exercised_stock_options', 'restricted_stock',
             'restricted_stock_deferred', 'total_stock_value']
# All features
F_ALL = F_EMAIL + F_FINANCE
# All column names
COLUMNS = ['poi'] + ['email_address'] + F_EMAIL + F_FINANCE
# Data type of all columns
DTYPE = [bool] + [str] + list(np.repeat(float, 19))


"""
Data Preparation
"""
# Load the dictionary containing the dataset
with open("./input/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# Convert dictionary into DataFrame
df = DataFrame.from_dict(data_dict, orient='index')
# Reorder columns
df = df.ix[:, COLUMNS]
# Convert data type
for i in xrange(len(COLUMNS)):
    df[COLUMNS[i]] = df[COLUMNS[i]].astype(DTYPE[i], raise_on_error=False)
# Assign 0 to NaN in finance features
# (Assuming that "-" in enron61702insiderpay.pdf means 0.)
for f in F_FINANCE:
    df.loc[df[f].isnull(), f] = 0

"""
Basic Exploratory Analysis
"""
print df.shape
print df.isnull().sum() / df.shape[0]
print df.describe()
print df.corr()


"""
NaN Check for email features
"""
missing_email_idx = []
for row in df.iterrows():
    nan_num = sum(row[1][2:7].isnull())
    if nan_num > 0:
        missing_email_idx.append(row[0])
df_missing_email = df.loc[missing_email_idx]


"""
Outlier Confirmation [Before data modification]
"""
plt.hist(df['to_messages'].dropna())
print df[df['to_messages'] > 4500]
print df[df['to_messages'] > 10000]

plt.hist(df['from_messages'].dropna())
print df[df['from_messages'] > 2800]

plt.hist(df['from_poi_to_this_person'].dropna())
print df[df['from_poi_to_this_person'] > 500]

plt.hist(df['from_this_person_to_poi'].dropna())
print df[df['from_this_person_to_poi'] > 300]

plt.hist(df['shared_receipt_with_poi'].dropna())
print df[df['shared_receipt_with_poi'] > 4000]

plt.hist(df['salary'])
print df[df['salary'] > 2.4e+07]
# -> [Modification] "TOTAL" is an invalid data point
print df[df['salary'] > 1.0e+06]

plt.hist(df['bonus'])
print df[df['bonus'] > 4.0e+06]

plt.hist(df['long_term_incentive'])
print df[df['long_term_incentive'] > 2.0e+06]

plt.hist(df['deferred_income'])
print df[df['deferred_income'] < -2.8e+06]
print df['deferred_income'][df['deferred_income'] < -2.8e+06]

plt.hist(df['deferral_payments'])
print df[df['deferral_payments'] < 0]
# -> [Modification] BELFER ROBERT - miss-alignment of columns
print df[df['deferral_payments'] > 5.7e+06]

plt.hist(df['loan_advances'])
print df[df['loan_advances'] > 7.3e+07]
plt.hist(df.loc[df['loan_advances'] < 7.3e+07, 'loan_advances'])
print df[df['loan_advances'] > 4.0e+04]
plt.hist(df.loc[df['loan_advances'] < 4.0e+04, 'loan_advances'])
print sum(df['loan_advances'][df['loan_advances'] < 4.0e+04])
# -> Only few people took loan advances, most of them are 0.

plt.hist(df['other'])
print df[df['other'] > 7.0e+06]

plt.hist(df['expenses'])
print df[df['expenses'] > 1.37e+05]

plt.hist(df['director_fees'])
print df[df['director_fees'] > 1.0e+05]

plt.hist(df['total_payments'])
print df[df['total_payments'] > 9.3e+07]

plt.hist(df['exercised_stock_options'])
print df[df['exercised_stock_options'] > 2.7e+07]

plt.hist(df['restricted_stock'])
print df[df['restricted_stock'] < 0]
# -> [Modification] BHATNAGAR SANJAY - miss-alignment of columns
print df[df['restricted_stock'] > 1.3e+07]

plt.hist(df['restricted_stock_deferred'])
print df[df['restricted_stock_deferred'] < -1.5e+06]

plt.hist(df['total_stock_value'])
print df[df['total_stock_value'] > 2.4e+07]
