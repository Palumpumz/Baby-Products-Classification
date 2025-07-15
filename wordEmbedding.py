from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Open the files
import numpy as np
trainfile = open("trainData.dat")

# Separate the review ratings and texts from a train data file
def trainfile_split(file):
    sentimentalValue = []
    texts = []
    for x in file:
        sentimentalValue.append(x[:2]) 
        texts.append(x[3:].strip())    
    return np.array(sentimentalValue), texts

sentimentalValue, train_texts = trainfile_split(trainfile) 

# Prepare the text by removing non alphanumerics, punctuation marks, and coverting text to lowercase
import re
def preprocess_text(allText):
    finishedText = []
    for text in allText:
        lowerText = text.lower()
        no_punct = re.sub(r'[\W]', ' ', lowerText)
        finishedText.append(no_punct)
    return finishedText

train_texts  = preprocess_text(train_texts)

# Create Bag of Words representation
vectorizer = CountVectorizer()
bow_train = vectorizer.fit_transform(train_texts)

# Calculate tf-idf value to find the word frequency
vectorizer = TfidfVectorizer()
tfidef_train = vectorizer.fit_transform(train_texts)

# Create bigrams (n=2)
vectorizer = CountVectorizer(ngram_range=(2, 2))
ngram_train = vectorizer.fit_transform(train_texts)

# Create TF-IDF Vectorizer with n-grams (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidef_ngram_train = vectorizer.fit_transform(train_texts)

# To get the accuracy of the model by using 70% training and 30% testing
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size=0.30)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, model.predict(X_test))*100, "\n")
    return model

print("Bag Of Words")
model = train_model(bow_train, sentimentalValue)

print("TF-IDF-weighted BoW")
model = train_model(tfidef_train, sentimentalValue)

print("n-gram")
model = train_model(ngram_train, sentimentalValue)

print("TF-IDF-weighted n-gram")
model = train_model(tfidef_ngram_train, sentimentalValue)
