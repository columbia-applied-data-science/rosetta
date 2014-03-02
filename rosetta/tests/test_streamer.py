import unittest
from StringIO import StringIO

from scipy import sparse

from rosetta import TokenizerBasic
from rosetta.text.streamers import TextIterStreamer
from rosetta.common import DocIDError, TokenError


class TestTextIterStreamer(unittest.TestCase):
    def setUp(self):
        self.text_iter = [{'text': 'doomed to failure', 'doc_id': 'a'},
                          {'text': 'set for success', 'doc_id': '1'}]
        self.tokenizer = TokenizerBasic()

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
        result = StringIO()
        stream.to_vw(result, cache_list=['doc_id'])

        benchmark = " 1 a| failure:1 doomed:1\n 1 1| set:1 success:1\n"
        self.assertEqual(benchmark, result.getvalue())

    def test_to_scipyspare(self):
        stream = TextIterStreamer(text_iter=self.text_iter,
                                  tokenizer=self.tokenizer)

        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())


from rosetta.text.streamers import MySQLStreamer


class TestMySQLStreamer(unittest.TestCase):
    def setUp(self):
        self.query_result = [{'text': 'doomed to failure', 'doc_id': 'a'},
                {'text': 'set for success', 'doc_id': '1'}]
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

    def test_to_vw(self):
        stream = MySQLStreamer(self.db_setup,
                                  tokenizer=self.tokenizer)
        stream.cursor = self.mock_cursor
        result = StringIO()
        stream.to_vw(result, cache_list=['doc_id'])

        benchmark = " 1 a| failure:1 doomed:1\n 1 1| set:1 success:1\n"
        self.assertEqual(benchmark, result.getvalue())

    def test_to_scipyspare(self):
        stream = MySQLStreamer(self.db_setup,
                                  tokenizer=self.tokenizer)

        stream.cursor = self.mock_cursor
        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])

        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())

