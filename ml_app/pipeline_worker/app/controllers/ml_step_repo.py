#!/usr/bin/env python
from controllers.injector import Injector
from glob import glob
import os
import zipfile
import pandas as pd
import mlflow
import tempfile
from mlflow.tracking import MlflowClient
from pycaret.classification import setup,compare_models,models,tune_model,finalize_model

class MLStepRepo(Injector):
    def __init__(self,pipeline_params):
        super().__init__()
        self.pipeline_params = pipeline_params

    def get_step(self,step):
        return getattr(self,step)

    def load_data_from_file(self,path,extension='csv'):
        zipped = zipfile.ZipFile(path)
        temp_dir = tempfile.TemporaryDirectory()
        
        zipped.extractall(path=temp_dir.name)
        zipped.close() 

        unzipped = glob(os.path.join(
            temp_dir.name,f"*.{extension}"))[0]
        df = pd.read_csv(unzipped)    
        return df

    def load_data_from_kaggle(self):
        pass

    def validate_data(self,df):
        return True

    def generate_features(self,df):
        return df

    def evaluate_model(self):
        pass

    def autoML(self,df,target, main_metric):
        clf = setup(df,
            log_experiment=True,
            target=target,
            experiment_name=self.pipeline_params['mlflow_config']['experiment_name'],
            silent=True,
            n_jobs=10
        )

        best_model = compare_models(
            include=list(models('tree').index),
            fold=2,
            sort=main_metric)

        tuned_best = tune_model(
            best_model,
            fold=2,
            n_iter=2,
            optimize=main_metric,
            choose_better=True)

        finalize_model(tuned_best)

        return tuned_best

    def validate_model(self,model,df):
        return True

    def register_model(self,main_metric=None,run_id=None):
        if(bool(run_id)):
            mlflow.register_model(
                model_uri=os.path.join('runs:/',run_id,'model'),
                name=self.pipeline_params['mlflow_config']['experiment_name']
            )
        elif(bool(main_metric)):
            experiment = mlflow.get_experiment_by_name(
                self.pipeline_params['mlflow_config']['experiment_name'])
            client = MlflowClient()
            best_run = client.search_runs([experiment.experiment_id],
            order_by=[f"metrics.{main_metric} DESC"])[0]
            mlflow.register_model(
                model_uri=os.path.join('runs:/',best_run.info.run_id,'model'),
                name=self.pipeline_params['mlflow_config']['experiment_name']
            )

            

