from fastapi import FastAPI, Request
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
from logging.config import dictConfig
import logging
import json


from utils.log_config import get_config

dictConfig(get_config())
app = FastAPI(debug=True)

from controllers.model import Model
model = Model()

logger = logging.getLogger('custom-logger')

@app.get("/root")
async def root():
    return {"message": "Hello Serasa!"}

class InferenceBody(BaseModel):
    data: list[float] = None

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame(payload,index=[0])
    logger.debug(f"Prediction for: {df}")
    prediction = model.predict(df)
    logger.debug(f"Predicted for: {prediction}")
    return {"message": {"predictions" :  str(prediction)}}

@app.get("/update/{model_version}")
def update(model_version: int):
    if(model_version > 0):
        message = model.load_model(model_version)
    else:
        message = model.load_model()
    return {"message": message}