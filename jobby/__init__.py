import datetime as _dt
import os as _os

from envbash import load_envbash as _load_env
from pymongo import MongoClient as _MC

from .utils import get_env as _get_env


_load_env(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), 
        '.env'))


class JobbyJob(object):
    def __init__(self, args, namespace='job'):
        assert type(args) is dict

        self._db_url = _get_env('JOBBY_DATABASE_URL')
        self._namespace = namespace
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
