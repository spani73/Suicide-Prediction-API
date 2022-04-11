import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle

pickle_in = open("classifier.pkl", "rb")
clf = pickle.load(pickle_in)

pickle_in_tfidf = open("tfidf.pkl","rb")
tfidf = pickle.load(pickle_in_tfidf)

app = FastAPI()



class TweetData(BaseModel):
	X : str


@app.get("/")
def index():
	return {'message' : f'Hello Stranger'}


@app.post("/predict")
def preProcessData(data:TweetData):
    data = data.dict()
    X= data['X']
    vect = tfidf.transform([X])
    label = {0:'negative', 1:'positive'}
    prediction = label[clf.predict(vect)[0]]
    probability = np.max(clf.predict_proba(vect))*100
    print()
    results = {"prediction" : prediction , "probability" : probability}
    return results
