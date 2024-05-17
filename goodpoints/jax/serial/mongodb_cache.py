'''
MongoDB cache. See `dummy_cache.py` for the documentation of each function.
'''

from goodpoints.jax.serial.dummy_cache import DummyCache

from pymongo import MongoClient
import time
import numpy as np
import h5py
from pathlib import Path
import logging
from datetime import datetime

from goodpoints.jax.serial.h5 import save_dict_h5


class MongoDBCache(DummyCache):
    def __init__(self, uri, db_name, collection_name, *,
                 h5_folder,
                 record={},
                 non_key_record={},
                 retry_interval=5):
        '''
        Args:
            uri: A string representing the URI of the MongoDB server.
            db_name: A string representing the name of the database.
            collection_name: A string representing the name of the collection.
            h5_folder:
                A string representing the path to the folder where the
                cached results are stored.
            record: Initial record to be used for caching.
            non_key_record: Non-key records to be stored in the cache.
            retry_interval:
                The interval in seconds to retry when the cache is busy.
        '''
        super().__init__()

        client = MongoClient(uri)
        self.db = client[db_name]
        self.collection = self.db[collection_name]
        self.h5_folder = Path(h5_folder)
        self.h5_folder.mkdir(parents=True, exist_ok=True)
        self.retry_interval = retry_interval

        self.cur_record = record.copy()
        self.non_key_record = non_key_record.copy()
        self.result_dict = {}

    def nonblocking_advance(self, stage, new_record):
        self.cur_record[stage] = new_record

    def blocking_advance(self, stage, new_record, exec_fn):
        self.cur_record[stage] = new_record

        while True:
            update_result = self.collection.update_one(
                {**self.cur_record,
                 '_stage': stage},
                {
                    '$setOnInsert': {
                        '_status': 'busy',
                        '_created_at': datetime.now(),
                        **self.non_key_record,
                    },
                },
                upsert=True,
            )
            if update_result.matched_count == 0:
                upserted_id = update_result.upserted_id
                logging.info(f'[{stage}] Inserting a new document '
                             f'({upserted_id}) '
                             'and marking it as busy.')
                try:
                    result_dict = exec_fn()
                except Exception as e:
                    logging.error(f'[{stage}] Exception occurred: {e}')
                    # Remove busy record.
                    self.collection.delete_one({'_id': upserted_id})
                    raise

                h5_path = f'{upserted_id}.h5'
                save_dict_h5(result_dict, self.h5_folder / h5_path)

                self.collection.update_one(
                    {'_id': upserted_id},
                    {
                        '$set': {
                            '_h5_path': str(h5_path),
                            '_status': 'finished',
                        }},
                )
                logging.info(f'[{stage}] Committed document ({upserted_id}).')
                return h5py.File(self.h5_folder / h5_path, 'r')

            while True:
                doc = self.collection.find_one({
                    **self.cur_record,
                    '_stage': stage,
                })
                if doc is None:
                    logging.info(f'[{stage}] Cannot locate cached document! '
                                 f'Trying to insert a new document and get '
                                 f'busy...')
                    break
                if doc['_status'] == 'busy':
                    logging.info(f'[{stage}] Located busy cached document'
                                 f'({doc["_id"]}). Waiting for '
                                 f'{self.retry_interval}s...')
                    time.sleep(self.retry_interval)
                elif doc['_status'] == 'finished':
                    logging.info(f'[{stage}] Located finished cached document'
                                 f'({doc["_id"]}).')
                    return h5py.File(self.h5_folder / doc['_h5_path'], 'r')

    def finalize(self, result_dict):
        update_result = self.collection.update_one(
            {**self.cur_record,
             '_stage': 'final'},
            {
                '$set': {
                    '_status': 'finished',
                    '_created_at': datetime.now(),
                    **self.non_key_record,
                    '_result': result_dict,
                },
            },
            upsert=True,
        )
        if update_result.matched_count == 0:
            upserted_id = update_result.upserted_id
            logging.info(f'[final] Inserting a final document '
                         f'({upserted_id}).')
        else:
            logging.info(f'[final] Located existing final document.'
                         ' Overwriting it.')
