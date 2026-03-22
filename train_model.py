import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real])

# Combine title + text
data['content'] = data['title'] + " " + data['text']
data = data[['content', 'label']]

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text (optimized)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

data['content'] = data['content'].apply(clean_text)

print("Cleaning done")

# Split
X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ⚡ Balanced speed + accuracy
vectorizer = TfidfVectorizer(
    max_features=8000,      # faster
    ngram_range=(1, 2),     # good context
    min_df=2
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ⚡ Faster model but still accurate
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model updated successfully")