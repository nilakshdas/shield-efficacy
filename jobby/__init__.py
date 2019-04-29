import datetime as _dt
from pymongo import MongoClient as _MC


class JobbyJob(object):
    def __init__(self, db_url, args, namespace='job'):
        assert type(args) is dict

        self._namespace = namespace
        self._db_url = db_url
        self._args = args
        self._out = dict()
        self._started = None
        self._ended = None

    def __enter__(self):
        self._started = _dt.datetime.now()
        return self

    def __exit__(self, *args):
        self._ended = _dt.datetime.now()

        doc = {
            'started': self._started,
            'ended': self._ended,
            'args': self._args,
            'out': self._out
        }

        try:
            with _MC(self._db_url) as db_client:
                db = db_client.jobby
                collection = db[self._namespace]
                collection.insert_one(doc)
        except Exception as e:
            print(e)

    def update_output(self, **kwargs):
        self._out.update(kwargs)