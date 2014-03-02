import os
import unittest

from StringIO import StringIO
from scipy import sparse

from rosetta import TokenizerBasic
from rosetta.text.streamers import TextFileStreamer
from rosetta.common import DocIDError, TokenError


class TestTextFileStreamer(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath('./rosetta/tests')
        self.testdata_path = os.path.join(self.test_path, 'temp')
        ###create some temp files to work with
        self.doc1 = os.path.join(self.testdata_path, 'doc1.txt')
        self.doc2 = os.path.join(self.testdata_path, 'doc2.txt')
        with open(self.doc1, 'w') as f:
            f.write('doomed to failure\n')
        with open(self.doc2, 'w') as f:
            f.write('set for success\n')
        self.tokenizer = TokenizerBasic()


    def test_info_stream(self):
        stream = TextFileStreamer(path_list = [self.doc1, self.doc2],
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
        stream = TextFileStreamer(path_list = [self.doc1, self.doc2],
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
        stream = TextFileStreamer(path_list = [self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)
        result = StringIO()
        stream.to_vw(result)

        benchmark = " 1 doc1| failure:1 doomed:1\n 1 doc2| set:1 success:1\n"
        self.assertEqual(benchmark, result.getvalue())

    def test_to_scipyspare(self):
        stream = TextFileStreamer(path_list = [self.doc1, self.doc2],
                                  tokenizer=self.tokenizer)
        
        result = stream.to_scipysparse()
        benchmark = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])
        
        compare = result.toarray() == benchmark.toarray()
        self.assertTrue(compare.all())


    def tearDown(self):
        os.remove(self.doc1)
        os.remove(self.doc2)

