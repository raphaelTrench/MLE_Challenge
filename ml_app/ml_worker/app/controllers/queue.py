#!/usr/bin/env python
import pika
import pickle
import pymongo
import sys
import os
import logging

from controllers.injector import Injector

class Queue(Injector):
    def __init__(self):
        super().__init__()
        self.rabbitMQHost = os.environ.get("RABBITMQ_HOST")
        self.dbHost = os.environ.get("DB_HOST")
        self.queueName = os.environ.get("WORKER_QUEUE_NAME")
        self.heartBeatTimeOut = int(os.environ.get("HEART_BEAT_TIMEOUT"))
        self.blockedConnectionTimeOut = int(os.environ.get("BLOCKED_CONNECTION_TIMEOUT"))

        self._channel = self._get_channel()

    def _get_channel(self):
        connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host= self.rabbitMQHost,
            heartbeat=self.heartBeatTimeOut,
            blocked_connection_timeout=self.blockedConnectionTimeOut))
        channel = connection.channel()

        return channel

    def start_consuming_messages(self,callback_function):
        logging.error("****************************")
        
        logging.info("model server - receiver")
        self._channel.queue_declare(
            queue=self.queueName,
            durable=True        
        )
        
        self._channel.queue_declare(
            queue=f'{self.queueName}-dlq', durable=True)

        self._channel.basic_publish(
            exchange='',
             routing_key=f'{self.queueName}-dlq',
              body='Hello World!')

        logging.info(' [*] Waiting for messages. To exit press CTRL+C')
        self._channel.basic_qos(prefetch_count=1)
        self._channel.basic_consume(
            queue=self.queueName,
            auto_ack=True,
            on_message_callback=callback_function)
        self._channel.start_consuming()