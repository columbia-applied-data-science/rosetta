"""
Classes for streaming tokens/info from databases
"""
import abc

try:
    import MySQLdb
    import MySQLdb.cursors
    HAS_MYSQLDB = True
except ImportError:
    HAS_MYSQLDB = False


from streamers import BaseStreamer
import pymongo

from .. import common
from rosetta.text import text_processors


class DBStreamer(BaseStreamer):
    """
    Database streamer base class
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
            self, db_setup, tokenizer=None, tokenizer_func=None):
        """
        Parameters
        ----------
        db_setup: A dictionary containing parameters needed to connect to, and
            query the database.  The required parameters are documented in each
            subclass, but at minimum you will need information about the host,
            username/password, and the query that will be executed.  The query
            must return a 'text' field in its dictionary.
        tokenizer : Subclass of BaseTokenizer
            Should have a text_to_token_list method.  Try using MakeTokenizer
            to convert a function to a valid tokenizer.
        tokenizer_func : Function
            Transforms a string (representing one file) to a list of strings
            (the 'tokens').
        """
        self.db_setup = db_setup
        self.tokenizer = tokenizer
        self.tokenizer_func = tokenizer_func
        self.cursor = None

        assert (tokenizer is None) or (tokenizer_func is None)
        if tokenizer_func:
            self.tokenizer = text_processors.MakeTokenizer(tokenizer_func)

    @abc.abstractmethod
    def connect(self):
        """
        Open connection to database.
        sets the classes cursor object.
        """
        return

    @abc.abstractmethod
    def disconnect(self):
        """
        Close connection to database
        """
        return

    @abc.abstractmethod
    def iterate_over_query(self):
        """
        Return an iterator over query result.
        We suggest that the entire query result not be returned and that
        iteration is controlled on server side, but this method does not
        guarantee that.  This method must return a dictionary, which at
        least has the key 'text' in it, containing the next to be tokenized.
        """
        return

    def record_stream(self):
        for info in self.iterate_over_query():
            yield info

    def info_stream(self):
        """
        Yields a dict from self.executing the query as well as "tokens".
        """
        for info in self.record_stream():
            info['tokens'] = self.tokenizer.text_to_token_list(info['text'])
            yield info


class MySQLStreamer(DBStreamer):
    """
    Subclass of DBStreamer to connect to a MySQL database and iterate over
    query results.  db_setup is expected to be a dictionary containing
    host, user, password, database, and query.  The query itself must return
    a column named text.

    Example:
        db_setup = {}
        db_setup['host'] = 'hostname'
        db_setup['user'] = 'username'
        db_setup['password'] = 'password'
        db_setup['database'] = 'database'
        db_setup['query'] = 'select
                                id as doc_id,
                                body as text
                             from tablename
                             where length(body) > 100'

        my_tokenizer = TokenizerBasic()
        stream = MySQLStreamer(db_setup=db_setup, tokenizer=my_tokenizer)

        for text in stream.info_stream(cache_list=['doc_id']):
            print text['doc_id'], text['tokens']
    """
    def __init__(self, *args, **kwargs):
        if not HAS_MYSQLDB:
            raise ImportError("MySQLdb was not importable, therefore\
                MySQLStreamer cannot be used.")
        super(MySQLStreamer, self).__init__(*args, **kwargs)

    def connect(self):
        try:
            _host = self.db_setup['host']
            _user = self.db_setup['user']
            _password = self.db_setup['password']
            _db = self.db_setup['database']
        except:
            raise common.BadDataError("MySQLStreamer expects db_setup to have \
                        host, user, password, and database fields")
        connection = MySQLdb.connect(
            host=_host, user=_user,
            passwd=_password, db=_db,
            cursorclass=MySQLdb.cursors.SSDictCursor)
        self.cursor = connection.cursor()

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        self.cursor = None

    def iterate_over_query(self):
        if not self.cursor:
            self.connect()
        try:
            _query = self.db_setup['query']
        except:
            raise common.BadDataError("MySQLStreamer expects db_setup \
                                      to have a query field")
        self.cursor.execute(_query)
        for result in self.cursor:
            if 'text' not in result:
                raise common.BadDataError("The query must return a text field")
            yield result


class MongoStreamer(DBStreamer):
    """
    Subclass of DBStreamer to connect to a Mongo database and iterate over
    query results.  db_setup is expected to be a dictionary containing
    host, database, collection, query, and text_key.  Additionally an optional
    limit parameter is allowed.
    The query itself must return a column named text_key which is passed on
    as 'text' to the iterator.
    In addition, because it is difficult to rename mongo fields (similar
    to the SQL 'AS' syntax), we allow a translation dictionary to be
    passed in, which translates keys in the mongo dictionary result names
    k to be passed into the result as v for key value pairs {k : v}.
    Currently we don't deal with nested documents.

    Example:

        db_setup = {}
        db_setup['host'] = 'localhost'
        db_setup['database'] = 'places'
        db_setup['collection'] = 'opentable'
        db_setup['query'] = {}
        db_setup['limit'] = 5
        db_setup['text_key'] = 'desc'
        db_setup['translations'] = {'_id' : 'doc_id'}

        # In this example, we assume that the collection has a field named
        # desc, holding the text to be analyzed, and a field named _id which
        # will be translated to doc_id and stored in the cache.

        my_tokenizer = TokenizerBasic()
        stream = MongoStreamer(db_setup=db_setup, tokenizer=my_tokenizer)

        for text in stream.info_stream(cache_list=['doc_id']):
            print text['doc_id'], text['tokens']
    """
    def connect(self):
        try:
            _host = self.db_setup['host']
            _db = self.db_setup['database']
            _col = self.db_setup['collection']
            if 'port' in self.db_setup:
                _port = self.db_setup['port']
            else:
                _port = None
        except:
            raise common.BadDataError("MongoStreamer expects db_setup to have \
                        host and database fields")

        client = pymongo.MongoClient(_host, _port)
        db = client[_db]
        col = db[_col]
        self.cursor = col

    def disconnect(self):
        self.cursor = None

    def iterate_over_query(self):
        if not self.cursor:
            self.connect()
        try:
            _query = self.db_setup['query']
            _text_key = self.db_setup['text_key']
            if 'limit' in self.db_setup:
                _limit = self.db_setup['limit']
            else:
                _limit = None
            if 'translations' in self.db_setup:
                _translate = self.db_setup['translations']
            else:
                _translate = None
        except:
            raise common.BadDataError("MySQLStreamer expects db_setup \
                                      to have a query and text_key field")

        results = self.cursor.find(_query)

        if _limit:
            results = results.limit(_limit)

        for result in results:
            if _text_key not in result:
                raise common.BadDataError("The query must return the \
                                           specified text field")
            result['text'] = result[_text_key]
            if _translate:
                for k, v in _translate.items():
                    result[v] = result[k]
            yield result
