#!/usr/bin/env python
from controllers.injector import Injector
from controllers.ml_step_repo import MLStepRepo

import mlflow

class MLPipeline(Injector):
    def __init__(self, rabbit_message):
        super().__init__()
        self._ml_steps = MLStepRepo()
        self._message = rabbit_message
        self._pipeline = rabbit_message['pipeline']

    def _run_step(self):
        pass
    
    def run_pipeline(self):
        pass

    def demo_pipeline(self):
        df = self._ml_steps.load_data_from_file(
            path=self._pipeline['load_data_from_file']['path'])
        data_is_valid = self._ml_steps.validate_data(df)

        if(not data_is_valid):
            return
            
        df = self._ml_steps.generate_features(df)
        model = self._ml_steps.autoML(df)
        model_is_valid = self._ml_steps.validate_model(model,df)

        if(not model_is_valid):
            return

        self._ml_steps.deploy_model()

