#!/usr/bin/env python
import pika
import pickle
import pymongo
import sys
import json
import os
import logging
import traceback

from app.controllers.queue import Queue
from app.controllers.inference_worker import InferenceWorker

def callback(ch, method, properties, body):
    logging.info(" [x] Received %r" % body.decode())
    message = json.loads(body)
    worker = InferenceWorker(
        inference_type=message.get('inference_type','standard'),
        model_version=message.get('model_version',0),
        model_name=message['model_name']
    )
    data = message['data']
    worker.predict(data)
    logging.info(f"Finished!")

def main():
    queue = Queue()
    queue.start_consuming_messages(callback)    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit(1)