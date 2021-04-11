#!/usr/bin/env python
from controllers.injector import Injector
from controllers.ml_step_repo import MLStepRepo

import mlflow

class MLPipeline(Injector):
    def __init__(self, rabbit_message):
        super().__init__()
        self._run_params = rabbit_message
        self._pipeline = rabbit_message['pipeline']
        self._ml_steps = MLStepRepo(self._run_params)
        self._message = rabbit_message
        
    def _build_step(self):
        pass
    
    def run_pipeline(self):
        pass

    def demo_pipeline(self):
        df = self._ml_steps.load_data_from_file(
            **self._pipeline['load_data_from_file'])
        data_is_valid = self._ml_steps.validate_data(df)

        if(not data_is_valid):
            return
            
        model = self._ml_steps.autoML(df,**self._pipeline['autoML'])
        model_is_valid = self._ml_steps.validate_model(model,df)

        if(not model_is_valid):
            return

        self._ml_steps.register_model(**self._pipeline['register_model'])

