#!/usr/bin/env python
from controllers.injector import Injector
from controllers.ml_steps import MLSteps

import datetime
import time
import  os
import mlflow
import  traceback

class MLPipeline(Injector):
    """Dynamically builds ML pipelines based on the pipeline especification and experiment configurations received.
    All experiments are directly logged to the mlflow tracking server,where they can be analyzed later.
    If desired, then the best model resulting from the experiment session will also be registered in the model registry, to be used for
    for inference by the API (ml_server) and the inference workers(inference_worker).
    Lastly, this worker can also be used to solely register a desired model, given its run_id.
    Example message body:
    {
        "pipeline":[
            {
                "MLSteps":[
                    {
                    "load_data_from_kaggle":{
                        "kaggle_dataset_name":"mlg-ulb/creditcardfraud",
                        "training_file_name":"creditcard.csv"
                    }
                    },
                    {
                    "train_model":{
                        "feature_engineering_params":{
                            "fix_imbalance":true
                        },
                        "modeling_params":{
                            "estimator_list":[
                                "dt",
                                "lr"
                            ],
                            "fold":2,
                            "n_select":2,
                            "tune":false,
                            "ensemble":true
                        },
                        "finalize":true
                    }
                    },
                    {
                    "register_model":{
                        
                    }
                    }
                ]
            }
        ],
        "experiment_config":{
            "experiment_name":"test",
            "model_name":"test",
            "main_metric":"F1",
            "problem_type":"classification",
            "target":"Class"
        }
    }

    Args:
        Injector ([type]): [description]
    """
    def __init__(self, rabbit_message, pipeline_id):
        """[summary]

        Args:
            rabbit_message ([type]): [description]
            pipeline_id ([type]): [description]
        """
        super().__init__()
        self._pipeline_id = pipeline_id or datetime.datetime.utcnow()
        self._worker_input = rabbit_message
        self._experiment_config = rabbit_message['experiment_config']
        self._pipeline = rabbit_message['pipeline']
        self.experiment_name = None 
        self.model_name = None
        self.main_metric = None
        self.problem_type = None
        self.target = None

        self.training_data = None
        self.final_model_id = None

        self._init_configs()
        self._pipelines_db = (
            self.db_client[self.model_name][os.environ['PIPELINE_MONITORING_COLLECTION']])

        self._ml_steps = MLSteps(self)
        self._step_enum = {
            "MLSteps" : self._ml_steps
        }
        
    def _init_configs(self):
        self.experiment_name = self._experiment_config['experiment_name'] 
        self.model_name = self._experiment_config['model_name']
        self.main_metric = self._experiment_config['main_metric'] 
        self.problem_type = self._experiment_config['problem_type']
        self.target = self._experiment_config['target']   

    def _build_step(self,step_type,step):        
        return getattr(self._step_enum[step_type],step)

    def _save_pipeline_tracking(self,start_time,end_time=None,status="running",fail_log=None):
        pipeline_doc = {
            "_id" : self._pipeline_id,
            "pipeline"  : self._pipeline,
            "experiment_config" :  self._experiment_config,
            "status" : status,
            "final_model_id" : self.final_model_id,
            "start_time" : start_time,
            "end_time"  : end_time,
            "total_time" : end_time - start_time  if end_time else None,
            "fail_log" : fail_log
        }
        self._pipelines_db.replace_one(
            {"_id" : pipeline_doc['_id']},
            replacement=pipeline_doc,
            upsert=True)
    
    def run_pipeline(self):
        """Reads the pipeline specifications and runs each step according to the provided parameters
        """
        start_time = time.time()
        self._save_pipeline_tracking(start_time)
        try:
            for step_phase in self._pipeline:
                for step_type, steps in step_phase.items():
                    self.logger.info(f"Executing pipeline of step type '{step_type}' ")
                    for step in steps:
                        for step_function_name,  step_function_params in step.items():
                            self.logger.info(f"Executing step '{step_function_name}' with params '{step_function_params}'")
                            step_function = self._build_step(step_type,step_function_name)
                            step_function(**step_function_params)
            end_time = time.time()
            self._save_pipeline_tracking(start_time,end_time,"completed")
        except Exception:
            self._save_pipeline_tracking(start_time,status="fail",fail_log=traceback.format_exc())
            self.logger.error(traceback.format_exc())
            raise

