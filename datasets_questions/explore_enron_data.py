#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

num_persons = 0
num_features = 0
num_pois = 0

print enron_data["PRENTICE JAMES"]["total_stock_value"]

# for key in enron_data.iterkeys():
#     if enron_data[key]["poi"] == 1:
#         num_pois += 1
# print num_pois

# for person in enron_data:
#     num_persons += 1 
#     for feature in person:
#         num_features += 1

# for person["poi"] in enron_data:
#     print person["poi"]
    # if person["poi"] == 1:
    #     num_pois += 1

# avg_num_features = num_features/num_persons
# print("num_persons", num_persons)
# print("num_features", num_features)
# print("avg_num_features", avg_num_features)
# print("num_pois", num_pois)