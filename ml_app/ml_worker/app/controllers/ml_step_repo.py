#!/usr/bin/env python
from controllers.injector import Injector
from glob import glob
import os
import zipfile
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pycaret.classification import setup,compare_models,models,tune_model,finalize_model

class MLStepRepo(Injector):
    def __init__(self):
        super().__init__()
        self.MLFLOW_EXPERIMENT_NAME  = 'PIPE_TEST'

    def get_step(self,step):
        return getattr(self,step)

    def load_data_from_file(self,path):
        zipped = zipfile.ZipFile(path)
        zipped.extractall()
        zipped.close()

        unzipped = path.replace('.zip','.csv')
        df = pd.read_csv(unzipped)    
        return df

    def load_data_from_kaggle(self):
        pass

    def validate_data(self,df):
        return True

    def generate_features(self,df):
        pass

    def evaluate_model(self):
        pass

    def autoML(self,df,target, main_metric):
        clf = setup(df,
            log_experiment=True,
            target=target,
            experiment_name=self.MLFLOW_EXPERIMENT_NAME,
            silent=True,
            n_jobs=10
        )

        best_model = compare_models(
            include=models(type=['tree']),
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

    def deploy_model(self,run_id=None):
        if(bool(run_id)):
            mlflow.register_model(
                model_uri=os.path.join('runs:/',run_id,'model'),
                name=self.MLFLOW_EXPERIMENT_NAME.split('-')[0]
            )
        else:
            experiment = mlflow.get_experiment_by_name(
                self.MLFLOW_EXPERIMENT_NAME)
            client = MlflowClient()
            best_run = client.search_runs([experiment.experiment_id],
            order_by=["metrics.F1 DESC"])[0]
            mlflow.register_model(
                model_uri=os.path.join('runs:/',best_run.run_id,'model'),
                name=self.MLFLOW_EXPERIMENT_NAME.split('-')[0]
            )

            

