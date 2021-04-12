#!/usr/bin/env python
from controllers.injector import Injector
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import time
import datetime

class InferenceWorker(Injector):
    def __init__(self,inference_type,model_version,model_name=None):
        super().__init__()
        self._model_name = model_name
        self._model_details  = None
        self._model = None
        self._model_udf = None
        self._inference_type = inference_type

        self._predictions_db = (
            self.db_client[self._model_name]['INFERENCE'])
        self._monitoring_db = (
            self.db_client[self._model_name]['MONITORING'])
        self._load_model(model_version)

    def _load_model(self,version):
        client = MlflowClient()
        if(version>0):
            model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self._model_name}'") if mv.version == str(version)]
        else:
            model_details  = [
                mv for mv in client.search_model_versions(
                    f"name='{self._model_name}'")]

        
        if(bool(model_details)):
            self._model_details= model_details[0]

            if(self._inference_type != "spark"):
                self._model = mlflow.pyfunc.load_model(
                    self._model_details.source)
            else:
                from pyspark.sql.types import FloatType

                self._model_udf = mlflow.pyfunc.spark_udf(
                    spark,
                    model_uri=self._model_details.source,
                    result_type=FloatType()
                )

    def _spark_predict(self,spark_df):
        spark_df = (spark_df.
            withColumn('prediction', self._model_udf(spark_df))        
        )
        ##
        # save data
        ##

    def _standard_predict(self,df):
        features = (self
            ._model
            .metadata
            .signature
            .inputs
            .column_names()
        )

        start = time.time()
        y_hat = (self._model.predict(df[features]))
        end =  time.time()
        time_elapsed = end-start

        timestamp = datetime.datetime.utcnow()
        predictions = [{
            "prediction" : float(y),
            "model_run_id" : self._model.metadata.run_id,
            "timestamp" : timestamp
            } for y in y_hat]
        self._predictions_db.insert_many(predictions)

        self._save_monitoring_data(predictions,timestamp,df[features],time_elapsed)

    def predict(self,data):
        if(self._inference_type != "spark"):
            df = pd.DataFrame(data)
            self._standard_predict(df)
        else:
            spark_df = self.get_spark_df(data)
            self._spark_predict(spark_df)

    def _save_monitoring_data(self,predictions,timestamp,df,time_elapsed):
        time_per_prediction = time_elapsed/len(df)
        predictions = [{
            "prediction" : float(predictions[idx]['prediction']),
            "model_input_signature" : self._model.metadata.signature.inputs.to_dict(),
            "feature_values" : {col:float(df.iloc[idx][col]) for col in df},
            "model_run_id" : predictions[0]['model_run_id'],
            "avg_time_prediction"  : time_per_prediction,
            "total_batch_time" : time_elapsed,
            "timestamp" : timestamp,
            "origin"  :  "microservice",
            "batch_size" :  len(df)
            } for idx in range(len(df))]
        self._monitoring_db.insert_many(predictions)
        



            

