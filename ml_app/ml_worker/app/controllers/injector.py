
#!/usr/bin/env python
import logging
import os
import pymongo

logging.basicConfig(level=20)

class Injector():
    def __init__(self):
        self.db_client = self._get_db_client()
        self.logger = logging.getLogger()

    def _get_db_client(self):
        mongo_client = pymongo.MongoClient(
            f"mongodb://{os.environ['DB_HOST']}:27017")

        return mongo_client