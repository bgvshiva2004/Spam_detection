from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def predict_category(text):
            
            model = joblib.load("model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")

            text = [text]
            text_features = vectorizer.transform(text)
            prediction = model.predict(text_features)
            category = "Alert ! It's a Spam Mail" if prediction[0] == 0 else "Relax, the mail is Non-spam"

            return category

# Create your views here.

def index(request):
    if request.method == "POST":
        text = request.POST.get('text')
        return render(request,'index.html',{'category':predict_category(text)})

    return render(request,'index.html')

def login(request):
    return render(request,'login.html')