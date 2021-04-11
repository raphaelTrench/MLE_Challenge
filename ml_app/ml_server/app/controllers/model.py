#!/usr/bin/env python
import mlflow
from mlflow.tracking import MlflowClient
import os
import socket
import logging
import datetime
import pymongo
import time

logger = logging.getLogger('custom-logger')

class Model():
    def __init__(self):
        super().__init__()
        self._model_name  = os.environ['MODEL_NAME']
        self._model = None
        self._model_details = None

        self._db_client = self._get_db_client()
        self._monitoring_db = (
            self._db_client[os.environ['MODEL_NAME']][os.environ['MONITORING_COLLECTION']])

        self.load_model()

    def _get_db_client(self):
        mongo_client = pymongo.MongoClient(
            f"mongodb://{os.environ['DB_HOST']}:27017")

        return mongo_client        
        
    def load_model(self,version=0):
        client = MlflowClient()
        if(version):
            model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self._model_name}'") if mv.version == str(version)]
        else:
            model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self._model_name}'")]
        if(bool(model_details)):
            self._model_details= model_details[0]
            self._model = mlflow.pyfunc.load_model(
                self._model_details.source)

            model_version = self._model_details.version
            message = f'''Model {self._model_name} sucessfully updated to version {model_version}!'''
        else:
            message = f"Model {self._model_name}  of version {version} not found in registry!"

        return message

    def predict(self,X):
        features = (self
            ._model
            .metadata
            .signature
            .inputs
            .column_names()
        )
        return self._model.predict(X[features])

    def log_predictions(self,time_start,df,predictions):
        time_end = time.time()
        time_elapsed = time_end - time_start
        time_per_prediction = (time_elapsed)/len(df)
        timestamp = datetime.datetime.utcnow()
        predictions_log = [{
            "prediction" : float(predictions[idx]),
            "model_input_signature" : self._model.metadata.signature.inputs.to_dict(),
            "feature_values" : {col:float(df.iloc[idx][col]) for col in df},
            "model_run_id" : self._model.metadata.run_id,
            "avg_time_prediction"  : time_per_prediction,
            "timestamp" : timestamp,
            "origin" : "api",
            "total_batch_time" : time_elapsed,
            "batch_size"  :  len(df)
            } for idx in range(len(df))]
        self._monitoring_db.insert_many(predictions_log)
        