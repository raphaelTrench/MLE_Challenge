#!/usr/bin/env python
import mlflow
from mlflow.tracking import MlflowClient
import os
import socket
import logging

logger = logging.getLogger('custom-logger')

class Model():
    def __init__(self):
        super().__init__()
        self.model_name  = os.environ['MODEL_NAME']
        self.model = None
        self.model_details = None

        self.load_model()

    def load_model(self,version=0):
        client = MlflowClient()
        if(version):
            self.model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self.model_name}'") if mv.version == str(version)]
        else:
            self.model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self.model_name}'")]
        if(bool(self.model_details)):
            self.model_details= self.model_details[0]
            self.model = mlflow.pyfunc.load_model(
                self.model_details.source)

            model_version = self.model_details.version
            message = f'''Model {self.model_name} sucessfully updated to version {model_version}!'''
        else:
            message = f"Model {self.model_name}  of version {version} not found in registry!"

        return message

    def predict(self,X):
        features = (self
            .model
            .metadata
            .signature
            .inputs
            .column_names()
        )
        return self.model.predict(X[features])
        