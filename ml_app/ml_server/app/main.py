from fastapi import FastAPI, Request, BackgroundTasks
import pika
import pymongo
import uuid
from pydantic import BaseModel
from typing import List
import uvicorn
import threading
import pika
from time import sleep
import pandas as pd
import uuid
import os
from logging.config import dictConfig
import logging
import time
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
    data_to_predict: List[dict] = None

@app.post("/predict")
def predict(data: InferenceBody, background_tasks: BackgroundTasks):
    time_start = time.time()
    df = pd.DataFrame(data.data_to_predict)
    logger.debug(f"Prediction for: {df}")
    predictions = model.predict(df)
    logger.debug(f"Predicted for: {predictions}")

    background_tasks.add_task(
        model.log_predictions,
        time_start,
        df,
        predictions)

    return {"message": {"predictions" :  str(predictions)}}

# @app.get("/update/{model_version}")
# def update(model_version: int):
#     if(model_version > 0):
#         message = model.load_model(model_version)
#     else:
#         message = model.load_model()
#     return {"message": message}