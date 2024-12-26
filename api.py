from fastapi import FastAPI
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

tfidf = pickle.load(open("tf_idf.pkt", "rb"))
nb_model = pickle.load(open("toxicity_model.pkt", "rb"))

@app.post("/predict")
async def predict(text: str):
    text_tfidf = tfidf.transform([text]).toarray()
    

    prediction = nb_model.predict(text_tfidf)
    

    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    
    return {
        "text":text,
        "class":class_name
    }
    

