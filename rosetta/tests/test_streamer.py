import os
import unittest

from StringIO import StringIO
from scipy import sparse

from rosetta import TokenizerBasic
from rosetta.text.streamers import TextFileStreamer, TextIterStreamer
from rosetta.text.database_streamers import MySQLStreamer, MongoStreamer

try:
    import MySQLdb
    import MySQLdb.cursors
    HAS_MYSQLDB = True
except ImportError:
    HAS_MYSQLDB = False


class TestTextFileStreamer(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath('./rosetta/tests')
        self.testdata_path = os.path.join(self.test_path, 'temp')

        # create some temp files to work with
        self.doc1 = os.path.join(self.testdata_path, 'doc1.txt')
        self.doc2 = os.path.join(self.testdata_path, 'doc2.txt')
        with open(self.doc1, 'w') as f:
            f.write('doomed to failure\n')
        with open(self.doc2, 'w') as f:
            f.write('set for success\n')
        self.tokenizer = TokenizerBasic()

    def test_info_stream(self):
        stream = TextFileStreamer(path_list=[self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        text_benchmark = ['doomed to failure\n', 'set for success\n']

        token_result = []
        text_result = []
        for each in stream.info_stream():
            token_result.append(each['tokens'])
            text_result.append(each['text'])

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(text_benchmark, text_result)

    def test_token_stream(self):
        stream = TextFileStreamer(path_list=[self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        id_benchmark = ['doc1', 'doc2']
        token_result = []
        for each in stream.token_stream(cache_list=['doc_id']):
            token_result.append(each)

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(id_benchmark, stream.__dict__['doc_id_cache'])

    def test_to_vw(self):
        stream = TextFileStreamer(path_list=[self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)
        result = StringIO()
        stream.to_vw(result)

        benchmark = " 1 doc1| failure:1 doomed:1\n 1 doc2| set:1 success:1\n"
        self.assertEqual(benchmark, result.getvalue())

    def test_to_scipyspare(self):
        stream = TextFileStreamer(path_list=[self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)

        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())

    def tearDown(self):
        os.remove(self.doc1)
        os.remove(self.doc2)


class TestTextIterStreamer(unittest.TestCase):
    def setUp(self):
        self.text_iter = [{'text': 'doomed to failure', 'doc_id': 'a'},
                          {'text': 'set for success', 'doc_id': '1'}]
        self.tokenizer = TokenizerBasic()
        self.test_path = os.path.abspath('./rosetta/tests')
        self.temp_vw_path = os.path.join(self.test_path, 'temp', 'test.vw')

    def test_info_stream(self):
        stream = TextIterStreamer(text_iter=self.text_iter,
                                  tokenizer=self.tokenizer)
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        text_benchmark = ['doomed to failure', 'set for success']
        token_result = []
        text_result = []
        for each in stream.info_stream():
            token_result.append(each['tokens'])
            text_result.append(each['text'])

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(text_benchmark, text_result)

    def test_token_stream(self):
        stream = TextIterStreamer(text_iter=self.text_iter,
                                  tokenizer=self.tokenizer)
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        id_benchmark = ['a', '1']
        token_result = []
        for each in stream.token_stream(cache_list=['doc_id']):
            token_result.append(each)

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(id_benchmark, stream.__dict__['doc_id_cache'])

    def test_to_vw(self):
        stream = TextIterStreamer(text_iter=self.text_iter,
                                  tokenizer=self.tokenizer)
        stream.to_vw(open(self.temp_vw_path, 'w'))
        result = open(self.temp_vw_path).read()
        benchmark = " 1 a| failure:1 doomed:1\n 1 1| set:1 success:1\n"
        self.assertEqual(benchmark, result)

    def test_to_scipysparse(self):
        stream = TextIterStreamer(text_iter=self.text_iter,
                                  tokenizer=self.tokenizer)

        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())

    def tearDown(self):
        os.remove(self.temp_vw_path) if (
            os.path.exists(self.temp_vw_path)) else None


class TestMySQLStreamer(unittest.TestCase):
    def setUp(self):
        self.query_result = [{'text': 'doomed to failure', 'doc_id': 'a'},
                             {'text': 'set for success', 'doc_id': '1'}]
        self.test_path = os.path.abspath('./rosetta/tests')
        self.temp_vw_path = os.path.join(self.test_path, 'temp', 'test.vw')

        class MockCursor(object):
            def __init__(self, my_iter):
                self.my_iter = my_iter

            def __iter__(self):
                for item in self.my_iter:
                    yield item

            def execute(self, query):
                return None
        self.mock_cursor = MockCursor(self.query_result)
        self.db_setup = {}
        self.db_setup['host'] = 'hostname'
        self.db_setup['user'] = 'username'
        self.db_setup['password'] = 'password'
        self.db_setup['database'] = 'database'
        self.db_setup['query'] = 'select something'
        self.tokenizer = TokenizerBasic()

    @unittest.skipUnless(HAS_MYSQLDB, "requires MySQLdb")
    def test_info_stream(self):
        stream = MySQLStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        text_benchmark = ['doomed to failure', 'set for success']
        token_result = []
        text_result = []
        for each in stream.info_stream():
            token_result.append(each['tokens'])
            text_result.append(each['text'])

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(text_benchmark, text_result)

    @unittest.skipUnless(HAS_MYSQLDB, "requires MySQLdb")
    def test_token_stream(self):
        stream = MySQLStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        id_benchmark = ['a', '1']
        token_result = []
        for each in stream.token_stream(cache_list=['doc_id']):
            token_result.append(each)

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(id_benchmark, stream.__dict__['doc_id_cache'])

    @unittest.skipUnless(HAS_MYSQLDB, "requires MySQLdb")
    def test_to_vw(self):
        stream = MySQLStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        stream.to_vw(open(self.temp_vw_path, 'w'))
        result = open(self.temp_vw_path).read()

        benchmark = " 1 a| failure:1 doomed:1\n 1 1| set:1 success:1\n"
        self.assertEqual(benchmark, result)

    @unittest.skipUnless(HAS_MYSQLDB, "requires MySQLdb")
    def test_to_scipyspare(self):
        stream = MySQLStreamer(self.db_setup,
                               tokenizer=self.tokenizer)

        stream.cursor = self.mock_cursor
        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())

    def tearDown(self):
        os.remove(self.temp_vw_path) if (
            os.path.exists(self.temp_vw_path)) else None


class TestMongoStreamer(unittest.TestCase):
    def setUp(self):
        self.query_result = [{'text': 'doomed to failure', '_id': 'a'},
                             {'text': 'set for success', '_id': '1'}]

        class MockCursor(object):
            def __init__(self, my_iter):
                self.my_iter = my_iter

            def find(self, query):
                for item in self.my_iter:
                    yield item

            def execute(self):
                pass

        self.mock_cursor = MockCursor(self.query_result)
        self.db_setup = {}
        self.db_setup['host'] = 'hostname'
        self.db_setup['user'] = 'username'
        self.db_setup['password'] = 'password'
        self.db_setup['database'] = 'database'
        self.db_setup['query'] = 'select something'
        self.db_setup['text_key'] = 'text'
        self.db_setup['translations'] = {'_id': 'doc_id'}
        self.tokenizer = TokenizerBasic()
        self.test_path = os.path.abspath('./rosetta/tests')
        self.temp_vw_path = os.path.join(self.test_path, 'temp', 'test.vw')

    def test_info_stream(self):
        stream = MongoStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        text_benchmark = ['doomed to failure', 'set for success']
        token_result = []
        text_result = []
        for each in stream.info_stream():
            token_result.append(each['tokens'])
            text_result.append(each['text'])

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(text_benchmark, text_result)

    def test_token_stream(self):
        stream = MongoStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        token_benchmark = [['doomed', 'failure'],
                           ['set', 'success']]
        id_benchmark = ['a', '1']
        token_result = []
        for each in stream.token_stream(cache_list=['doc_id']):
            token_result.append(each)

        self.assertEqual(token_benchmark, token_result)
        self.assertEqual(id_benchmark, stream.__dict__['doc_id_cache'])

    def test_to_vw(self):
        stream = MongoStreamer(self.db_setup,
                               tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        stream.to_vw(open(self.temp_vw_path, 'w'))
        result = open(self.temp_vw_path).read()

        benchmark = " 1 a| failure:1 doomed:1\n 1 1| set:1 success:1\n"
        self.assertEqual(benchmark, result)

    def test_to_scipyspare(self):
        stream = MongoStreamer(self.db_setup,
                               tokenizer=self.tokenizer)

        stream.cursor = self.mock_cursor
        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())

    def tearDown(self):
        os.remove(self.temp_vw_path) if (
            os.path.exists(self.temp_vw_path)) else None
