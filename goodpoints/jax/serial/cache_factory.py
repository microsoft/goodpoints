'''
Factory function for creating caches.
'''

from pathlib import Path


def make_cache(name, cfg, record, non_key_record, cache_dir):
    if name == 'mongodb':
        from goodpoints.jax.serial.mongodb_cache import MongoDBCache
        return MongoDBCache(
            uri=cfg['uri'],
            db_name=cfg['db_name'],
            collection_name=cfg['collection_name'],
            h5_folder=Path(cache_dir) / cfg['collection_name'],
            record=record,
            non_key_record=non_key_record,
            retry_interval=cfg['retry_interval']
        )
    elif name == 'dummy':
        from goodpoints.jax.serial.dummy_cache import DummyCache
        return DummyCache()
    else:
        raise ValueError(f'Unknown cache type: {name}')
