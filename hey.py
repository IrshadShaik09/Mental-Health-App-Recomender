import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv("mental_health_apps.csv",encoding='latin-1')

# Create the features
features = data["SPECIALITY"].tolist()

# Create the target
target = data["APP NAME"].tolist()

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the features
vectorizer.fit(features)

# Transform the features
x = vectorizer.transform(features)

# Create the logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(x, target)

# Predict the app name for a given specialty
def predict_app_name(specialty):
  x = vectorizer.transform([specialty])
  prediction = model.predict(x)[0]
  return prediction

# Create a Streamlit app
import streamlit as st

# Title
st.title("Mental Health App Recommendation")

# Input
specialty = st.text_input("Enter your specialty:")

# Prediction
if specialty:
  prediction = predict_app_name(specialty)
  st.write("The recommended app for your specialty is:", prediction)