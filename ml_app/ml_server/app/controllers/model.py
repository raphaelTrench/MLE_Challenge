#!/usr/bin/env python
import mlflow
from mlflow.tracking import MlflowClient
from controllers.injector import Injector
import os

class Model(Injector):
    def __init__(self):
        super().__init__()
        self.model_name  = os.environ['MODEL_NAME']
        self.model = None
        self.model_details = None

        self.load_model()

    def load_model(self):
        client = MlflowClient()
        self.model_details  = [
            mv for mv in client.search_model_versions(
                f"name='{self.model_name}'")]
        if(bool(self.model_details)):
            self.model_details= self.model_details[0]
            self.model = mlflow.pyfunc.load_model(
                self.model_details.source)

            latest_model_version = self.model_details.version
            message = f'''Model {latest_model_version} sucessfully
            updated to version {self.model_name}!'''
        else:
            message = f"Model {self.model_name} not found in registry!"

        return message

    def predict(self,X):
        features = (self
            .model_details
            .metadata
            .signature
            .inputs
            .column_names()
        )
        return self.model.predict(X[features])

        