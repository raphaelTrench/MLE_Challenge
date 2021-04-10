#!/usr/bin/env python
import pika
import pickle
import pymongo
import sys
import json
import os
import logging

from controllers.queue import Queue
from controllers.ml_pipeline import MLPipeline 

logging.basicConfig(level=20)

def callback(ch, method, properties, body):
    logging.info(" [x] Received %r" % body.decode())
    message = json.loads(body)

    pipeline = MLPipeline(body)
    pipeline.demo_pipeline()

    logging.info(f"Finished!")

def main():
    queue = Queue()
    queue.start_consuming_messages(callback)    

if __name__ == '__main__':
    program_name = os.path.splitext(os.path.basename(__file__))[0]
    try:
        main()
    except Exception as e:
        logging.error(e)
        sys.exit(1)