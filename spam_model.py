import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

cv = CountVectorizer()
X = cv.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

def predict_spam(text):
    msg_cv = cv.transform([text])
    return model.predict(msg_cv)[0]

sample = "Congratulations! You have won a 1000$ gift card. Click here to claim."
print("Result:", "Spam" if predict_spam(sample) == 1 else "Not Spam")
