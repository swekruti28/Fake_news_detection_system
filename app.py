from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import re
from nltk.corpus import stopwords

# Initialize FastAPI
app = FastAPI()

# Enable CORS (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Request format
class NewsRequest(BaseModel):
    news: str

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Home route
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

# Prediction route
@app.post("/predict")
def predict(request: NewsRequest):
    cleaned = clean_text(request.news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    result = "Real News" if prediction == 1 else "Fake News"

    return {"prediction": result}