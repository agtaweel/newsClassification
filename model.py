import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

nltk.download('stopwords')


def read_data(path):
    print('reading data')
    dataset = pd.read_csv(path)
    print('nfnjdf')
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    dataset['class'] = imputer.fit_transform(dataset[['class']]).ravel()
    test_dataset = pd.read_csv('test.csv')
    print('nvmfnbmnmbnm')
    y_train = dataset.iloc[:, 0]
    x_train = dataset.iloc[:, 1:]
    x_train= bag_of_words(x_train)
    x_test = test_dataset.iloc[:, 1:]
    y_test = test_dataset.iloc[:, 0]
    x_test = bag_of_words(x_test)
    print('bagofwords')
    # Splitting the dataset into the Training set and Test set
    print('data was read')
    return x_train, x_test, y_train, y_test


# Creating Bag of Words model
def bag_of_words(X):

    corpus_articles = []
    corpus_titles = []
    for i in range(0, len(X)):
        content = X['title'][i]
        page = re.sub('[^a-zA-Z]', ' ', str(content))
        page = page.lower()
        page = page.split()
        ps = PorterStemmer()
        page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
        page = ' '.join(page)
        corpus_titles.append(page)
        content = X['article'][i]
        page = re.sub('[^a-zA-z]', ' ', str(content))
        page = page.lower()
        page = page.split()
        ps = PorterStemmer()
        page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
        page = ' '.join(page)
        corpus_articles.append(page)
        print((i + 1) / len(X) * 100, '%')
    cv = CountVectorizer(max_features=750)
    print('good')
    corpus_titles = cv.fit_transform(corpus_titles).toarray()
    corpus_articles = cv.fit_transform(corpus_articles).toarray()
    x = np.column_stack((corpus_titles, corpus_articles))
    print('done')
    return x


def train_data(x_train, y_train):

    print('start training data')
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    print('fitting data')
    classifier.fit(x_train, y_train)
    print('training finished')
    return classifier


def get_results(classifier, x_test, y_test):
    # Predicting the Test set results
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    # score = f1_score(y_test,predictions)
    cm = confusion_matrix(y_test, predictions)
    print('testing results: ')
    return accuracy, cm


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read_data('train.csv')
    # print(x_train)
    # print(y_train)
    classifier = train_data(x_train, y_train)
    accuracy, cm = get_results(classifier, x_test, y_test)
    # print('score = ', score*100, '%')
    print('accuracy = ', accuracy*100, '%')
    print('Confusion matrix = ', cm)
    # predictions = classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, predictions)
    # # score = f1_score(y_test, predictions)
    # cm = confusion_matrix(y_test, predictions)
    # print(cm)
    # print(accuracy*100, '%')
    # print(score*100, '%')
