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
num_persons_with_salary = 0
num_persons_with_mail = 0
num_persons_with_total_payments = 0
num_pois_with_total_payments = 0
# for key in enron_data.iterkeys():
# 	for feature in enron_data[key]:
# 		num_features += 1
# 	break
# print num_features
#print enron_data["PRENTICE JAMES"]["total_stock_value"]

for key in enron_data.iterkeys():
	num_persons += 1
	if enron_data[key]["poi"] == 1:
		num_pois += 1
	if enron_data[key]["salary"] != 'NaN':
		num_persons_with_salary += 1
	if enron_data[key]["email_address"] != 'NaN':
 		num_persons_with_mail += 1
	if enron_data[key]["total_payments"] != 'NaN':
		num_persons_with_total_payments += 1
	if enron_data[key]["total_payments"] != 'NaN' and enron_data[key]["poi"] == 1 == 1:
		num_pois_with_total_payments += 1

print 'total number of persons:', num_persons
print 'num_pois:', num_pois
print 'persons with salary:', num_persons_with_salary
print 'persons with mail:', num_persons_with_mail
print 'num_persons_with_total_payments', num_persons_with_total_payments
print 'percentage of persons w total payments:', num_persons_with_total_payments*1.0/num_persons
print 'percentage pois with total_payments:', num_pois_with_total_payments*1.0/num_pois

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