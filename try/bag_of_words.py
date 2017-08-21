from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
string1 = "Hi Katie"
string2 = "Hi sebastian"
string3 = "Hi Katie"

email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)
print(bag_of_words)
