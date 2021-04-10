from fastapi import FastAPI
import pika
import pymongo
import uuid
from pydantic import BaseModel
import uvicorn
import threading
import pika
from time import sleep
import pandas as pd
import uuid
import os

from controllers.model import Model

app = FastAPI()
model = Model()

@app.get("/root")
async def root():
    return {"message": "Hello World Test"}

@app.post("/predict")
def predict(data):
    df = pd.read_json(data)
    prediction = model.predict(df)
    return {"message": {"prediction" :  prediction}}

@app.get("/update")
def update():
    message = model.load_model()
    return {"message": message}