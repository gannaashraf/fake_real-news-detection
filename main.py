import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

# 22 - 4 - 22
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
import re
import string

news = pd.read_csv("C:/Users/hebaa/Downloads/fake_or_real_news.csv", usecols=['text', 'label'])

shuffle(news)

x = news['text']
y = news['label']

stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


stemmer = PorterStemmer()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


# change to lower case
news['text'] = news['text'].str.lower()
# Remove punctuation
news['text'] = news['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
# Remove numbers
news['text'] = news['text'].str.replace('\d+', '')
# Remove symbols
news['text'] = news['text'].str.encode("ascii", "ignore").str.decode('ascii')
# Remove any links
news['text'] = news['text'].apply(lambda x: re.sub('(http[s]?S+)|(w+.[A-Za-z]{2,4}S*)', '', x))
# Remove stem word(return the word to its root)
news["text"] = news["text"].apply(lambda x: stem_words(x))
# Remove Extra Spaces
news["text"] = news["text"].apply(lambda x: re.sub(' +', ' ', x))
# Remove the stopwords
news['text'] = news['text'].apply(lambda x: remove_stopwords(x))

# news.drop_duplicates(inplace = True)
# news = news.dropna()


# Encode to 0,1 for label column
LabelEncoder = LabelEncoder()
LabelEncoder.fit(news['label'])
news['label'] = LabelEncoder.transform(news['label'])

# print(news)

'''
#K-Fold
kf = RepeatedKFold(n_splits=20, n_repeats=2, random_state=1)
for train_index, test_index in kf.split(news):
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
'''

'''
#Train-Test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
'''

'''
#not working
#Iloc
test_size = int(len(news) * 0.2)
test_x_aux, test_y_aux = x[:test_size], y[:test_size]
train_x_aux, train_y_aux = x[test_size:], y[test_size:]
x_train, y_train = x.iloc[train_x_aux], y.iloc[train_y_aux]
x_test, y_test = x.iloc[train_x_aux], y.iloc[train_y_aux]
'''

# ComputeTF

newsList = news['text'].tolist()

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(newsList)
vectorizer.get_feature_names_out()
feature_name = vectorizer.get_feature_names_out()
data = pd.DataFrame(x.todense().tolist(), columns=feature_name)
# print(data)

x = data.values

kf = RepeatedKFold(n_splits=20, n_repeats=2, random_state=1)
for train_index, test_index in kf.split(x):
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]


# # Train-Test split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# #not working
# #Iloc
# test_size = int(len(news) * 0.2)
# test_x_aux, test_y_aux = x[:test_size], y[:test_size]
# train_x_aux, train_y_aux = x[test_size:], y[test_size:]
# x_train, y_train = x.iloc[train_x_aux], y.iloc[train_y_aux]
# x_test, y_test = x.iloc[train_x_aux], y.iloc[train_y_aux]

# (heba)
# data_train = x.iloc[1:20000, :]
# data_test = x.iloc[3451: , :]
# clf = DecisionTreeClassifier()
#
# clf = clf.fit(data_train.iloc[0], data_train["Type"])
# res =clf.predict(data_test.iloc[0])


# Support vector machines (SVM)
SVM = svm.LinearSVC(random_state=0, tol=1e-05)

# train the model
SVM.fit(x_train, y_train)

# testing and predicting by using the testing data set x_test and y_test
y_pred = SVM.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(y_pred)
print(report)

# Logistic Regression
model = LogisticRegression(solver='liblinear', C=10, random_state=0)
# train the model
model.fit(x_train, y_train)

# testing and predicting by using the testing data set x_test and y_test
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(y_pred)
print(report)

# MLP Classifier
clf = MLPClassifier(solver='adam', random_state=1, max_iter=300)
# train the model
clf.fit(x_train, y_train)
# testing and predicting by using the testing data set x_test and y_test
y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(y_pred)
print(report)

# KNeighbors Classifier
neigh = KNeighborsClassifier(n_neighbors=3)

# train the model
neigh.fit(x_train, y_train)

# testing and predicting by using the testing data set x_test and y_test
y_pred = neigh.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(y_pred)
print(report)

# Decision Tree Classifier
clf = DecisionTreeClassifier()
# train the model
clf.fit(x_train, y_train)
# testing and predicting by using the testing data set x_test and y_test
y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(y_pred)
print(report)

# Gaussian Naive Bayes
gnb = GaussianNB()
# train the model
gnb.fit(x_train, y_train)
# testing and predicting by using the testing data set x_test and y_test
y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(accuracy)
print(y_pred)
print(report)
