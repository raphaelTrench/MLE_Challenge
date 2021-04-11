#!/usr/bin/env python
import pika
import pickle
import pymongo
import sys
import json
import os
import logging
import traceback

from controllers.queue import Queue
from controllers.ml_pipeline import MLPipeline

def callback(ch, method, properties, body):
    logging.info(" [x] Received %r" % body.decode())
    message = json.loads(body)
    pipeline = MLPipeline(message)
    pipeline.demo_pipeline()
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