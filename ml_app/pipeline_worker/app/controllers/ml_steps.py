#!/usr/bin/env python
from controllers.injector import Injector
from glob import glob
import os
import zipfile
import pandas as pd
import mlflow
import tempfile
import kaggle
from mlflow.tracking import MlflowClient

class MLSteps(Injector):
    def __init__(self,pipeline):
        super().__init__()
        self.pipeline = pipeline

    def load_data_from_file(self,path):
        zipped = zipfile.ZipFile(path)
        temp_dir = tempfile.TemporaryDirectory()
        
        zipped.extractall(path=temp_dir.name)
        zipped.close() 

        unzipped = glob(os.path.join(
            temp_dir.name,"*.csv"))[0]
        self.pipeline.training_data = pd.read_csv(unzipped)    

    def load_data_from_kaggle(self,kaggle_dataset_name,training_file_name):
        temp_dir = tempfile.TemporaryDirectory()

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('kaggle_dataset_name',
         path=temp_dir.name,
         unzip=True)
        self.pipeline.training_data = pd.read_csv(os.path.join(temp_dir.name,training_file_name))

    def validate_data(self):
        return True

    def train_model(self,feature_engineering_params,modeling_params,finalize):
        mlflow.set_experiment(self.pipeline.experiment_name)

        if(self.pipeline.problem_type == "classification"):
            import pycaret.classification as pyc
        else:
            import pycaret.regression as pyc

        pycaret_setup = pyc.setup(self.pipeline.training_data,
            log_experiment=True,
            target=self.pipeline.target,
            experiment_name=self.pipeline.experiment_name,
            silent=True,
            n_jobs=10,
            **feature_engineering_params
        )

        estimator_list = modeling_params.get('estimator_list')
        if(bool(estimator_list)):
            if(len(estimator_list) > 1):
                model = pyc.compare_models(
                    include=estimator_list,
                    fold=modeling_params.get('fold',5),
                    sort=self.pipeline.main_metric,
                    n_select=int(modeling_params.get('n_select',1))
                )
            else:
                model = pyc.create_model(
                    estimator=estimator_list[0],
                    fold=modeling_params.get('fold',5),
                    fit_kwargs=modeling_params.get('fit_kwargs',{})
                )
        else:
            model = pyc.compare_models(
                    fold=modeling_params.get('fold',5),
                    sort=self.pipeline.main_metric,
                    n_select=int(modeling_params.get('n_select',1))
                )

        multiple_models = isinstance(model,list)
        if(not multiple_models):
            model = [model]

        if(modeling_params.get('tune',False)):
            model = [pyc.tune_model(
                m,
                fold=modeling_params.get('fold',5),
                n_iter=modeling_params.get('n_iter',5),
                optimize=self.pipeline.main_metric,
                choose_better=True) for m in model]

        if(multiple_models and modeling_params.get('ensemble',False)):
            blender = pyc.blend_models(model)
            stacker = pyc.stack_models(model)

        if(finalize):
            final = pyc.finalize_model(pyc.automl(self.pipeline.main_metric))

    def register_model(self,run_id=None):
        if(not bool(run_id) and bool(self.pipeline.main_metric)):
            experiment = mlflow.get_experiment_by_name(
                self.pipeline.experiment_name)
            client = MlflowClient()
            best_run = client.search_runs(
                [experiment.experiment_id],
                order_by=[f"metrics.{self.pipeline.main_metric} DESC"])[0]
            run_id = best_run.info.run_id

        mlflow.register_model(
            model_uri=os.path.join('runs:/',best_run.info.run_id,'model'),
            name=self.pipeline.model_name
        )

        self.pipeline.final_model_id = run_id

            

