
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("/content/spam.csv", encoding='latin-1')


data = data[['v1','v2']]
data.columns = ['label','message']


data['label'] = data['label'].map({'ham':0, 'spam':1})


cv = CountVectorizer()
X = cv.fit_transform(data['message'])
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = MultinomialNB()
model.fit(X_train, y_train)


msg = ["Congratulations! You won a free ticket"]
msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

print(prediction)
