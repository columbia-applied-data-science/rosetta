import unittest
from StringIO import StringIO
from collections import Counter, OrderedDict

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from rosetta.text import text_processors, vw_helpers, nlp


class TestWordTokenizers(unittest.TestCase):
    """
    """
    def setUp(self):
        self.text = 'Is this really going to work? not sure, but maybe. ok'
        self.word_tokenize = nlp.word_tokenize
        self.bigram_tokenize = nlp.bigram_tokenize

    def test_word_tokenize(self):
        benchmark = [
            'Is', 'this', 'really', 'going', 'to', 'work', 'not', 'sure',
            'but', 'maybe', 'ok']
        result = self.word_tokenize(self.text)
        self.assertEqual(result, benchmark)

    def test_bigram_tokenize(self):
        benchmark = [
            ('Is', 'this'), ('this', 'really'), ('really', 'going'),
            ('going', 'to'), ('to', 'work'), ('not', 'sure'), ('but', 'maybe')]
        result = self.bigram_tokenize(self.text)
        self.assertEqual(result, benchmark)


class TestTokenizerBasic(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Tokenizer = text_processors.TokenizerBasic

    def test_text_to_counter(self):
        text = "Hi there's:alot,of | food hi"
        result = self.Tokenizer().text_to_counter(text)
        benchmark = Counter(["hi", "there's", "alot", "food", "hi"])
        self.assertEqual(result, benchmark)


class TestSparseFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = text_processors.SparseFormatter()

    def test_parse_feature_str(self):
        feature_str = ' hi:1 bye:2.2 what:3 is:'
        feature_values = self.formatter._parse_feature_str(feature_str)
        benchmark = {'hi': 1, 'bye': 2.2, 'what': 3, 'is': 1}
        self.assertEqual(feature_values, benchmark)


class TestVWFormatter(unittest.TestCase):
    """
    """
    def setUp(self):
        self.formatter = text_processors.VWFormatter()

    def test_get_sstr_01(self):
        doc_id = 'myname'
        feature_values = OrderedDict([('hello', 1), ('dude', 3)])
        importance = 1
        result = self.formatter.get_sstr(
            feature_values=feature_values, doc_id=doc_id,
            importance=importance)
        benchmark = " 1 %s| hello:1 dude:3" % doc_id
        self.assertEqual(result, benchmark)

    def test_write_dict_01(self):
        record_str = " 3.2 doc_id1| hello:1 bye:2"
        result = self.formatter.sstr_to_dict(record_str)
        benchmark = {
            'importance': 3.2, 'doc_id': 'doc_id1',
            'feature_values': {'hello': 1, 'bye': 2}}
        self.assertEqual(result, benchmark)


class TestVWHelpers(unittest.TestCase):
    def setUp(self):
        #self.varinfo_path = 'files/varinfo'
        self.varinfo_file = StringIO(
            'FeatureName                                                      '
            '\t   HashVal   MinVal   MaxVal    Weight   RelScore\n^bcc        '
            '\t     77964     0.00     1.00   +0.2789    100.00%\n^illiquids  '
            '\t     83330     5.00     2.00   -0.1786     64.05%\n')
        self.topics_file_1 = StringIO(
            "Version 7.3\nlabel: 11\n"
            "0 1.1 2.2\n"
            "1 1.11 2.22")
        self.num_topics_1 = 2
        self.predictions_file_1 = StringIO(
            "0.0 0.0 doc1\n"
            "0.0 0.0 doc2\n"
            "1.1 2.2 doc1\n"
            "1.11 2.22 doc2")
        self.start_line_1 = 2

    def test_parse_varinfo_01(self):
        result = vw_helpers.parse_varinfo(self.varinfo_file)
        benchmark = pd.DataFrame(
            {
                'feature_name': ['bcc', 'illiquids'],
                'hash_val': [77964, 83330],
                'max_val': [1., 2.],
                'min_val': [0., 5.],
                'rel_score': [1., 0.6405],
                'weight': [0.2789, -0.1786]}).set_index('hash_val')
        assert_frame_equal(result, benchmark)

    def test_parse_lda_topics_01(self):
        result = vw_helpers.parse_lda_topics(
            self.topics_file_1, self.num_topics_1, normalize=False)
        benchmark = pd.DataFrame(
            {
                'hash_val': [0, 1], 'topic_0': [1.1, 1.11],
                'topic_1': [2.2, 2.22]}).set_index('hash_val')
        assert_frame_equal(result, benchmark)

    def test_parse_lda_predictions_01(self):
        result = vw_helpers.parse_lda_predictions(
            self.predictions_file_1, self.num_topics_1, self.start_line_1,
            normalize=False)
        benchmark = pd.DataFrame(
            {'doc_id': ['doc1', 'doc2'], 'topic_0': [1.1, 1.11],
                'topic_1': [2.2, 2.22]}).set_index('doc_id')
        assert_frame_equal(result, benchmark)

    def test_find_start_line_lda_predictions(self):
        result = vw_helpers.find_start_line_lda_predictions(
            self.predictions_file_1, self.num_topics_1)
        self.assertEqual(result, 2)


class TestLDAResults(unittest.TestCase):
    def setUp(self):
        self.outfile = StringIO()

        formatter = text_processors.VWFormatter()
        sff = text_processors.SFileFilter(
            formatter, bit_precision=8, verbose=False)
        sff.id2token = {0: 'w0', 1: 'w1'}
        sfile = StringIO(" 1 doc1| w0:1 w1:2\n 1 doc2| w0:3 w1:4")
        sff.load_sfile(sfile)

        topics_file_1 = StringIO(
            "Version 7.3\nlabel: 11\n"
            "0 1 2\n"
            "1 3 4")
        num_topics_1 = 2
        predictions_file_1 = StringIO(
            "0.0 0.0 doc1\n"
            "0.0 0.0 doc2\n"
            "1 2 doc1\n"
            "39 58 doc2")
        self.lda = vw_helpers.LDAResults(
            topics_file_1, predictions_file_1, num_topics_1, sff)

    def test_print_topics_1(self):
        self.lda.print_topics(num_words=2, outfile=self.outfile)
        result = self.outfile.getvalue()
        benchmark = (
            u'========== Printing top 2 tokens in every topic==========\n-----'
            '-------------------------\nTopic name: topic_0.  P[topic_0] = 0.4'
            '000\n       topic_0  doc_freq\ntoken                   \nw1      '
            '  0.75         2\nw0        0.25         2\n\n-------------------'
            '-----------\nTopic name: topic_1.  P[topic_1] = 0.6000\n        t'
            'opic_1  doc_freq\ntoken                    \nw1     0.66666'
            '7         2\nw0     0.333333         2\n')
        self.assertEqual(result, benchmark)

    def test_set_probabilities_marginals(self):
        pr_doc = pd.Series({'doc1': 3./(3+39+58), 'doc2': (39.+58)/(3+39+58)})
        assert_series_equal(self.lda.pr_doc, pr_doc)

        pr_topic = pd.Series({'topic_0': 4./10, 'topic_1': 6./10})
        assert_series_equal(self.lda.pr_topic, pr_topic)

        # Use the topics file for the token marginals
        # Should be almost equal to results obtained with the predictions file
        pr_token = pd.Series({'w0': 3./10, 'w1': 7./10})
        assert_series_equal(self.lda.pr_token, pr_token)

    def test_prob_1(self):
        result = self.lda.prob_token_topic(token='w0', c_token=['w1'])
        benchmark = pd.DataFrame(
            {'topic_0': [np.nan], 'topic_1': [np.nan]}, index=['w0'])
        benchmark.index.name = 'token'
        assert_frame_equal(result, benchmark)

    def test_prob_2(self):
        result = self.lda.prob_token_topic(c_token=['w1'])
        benchmark = pd.DataFrame(
            {'topic_0': [3/7.], 'topic_1': [4/7.]}, index=['w1'])
        benchmark.index.name = 'token'
        assert_frame_equal(result, benchmark)

    def test_prob_3(self):
        result = self.lda.prob_token_topic(topic=['topic_0'], token=['w0'])
        benchmark = pd.DataFrame({'topic_0': [1/10.]}, index=['w0'])
        benchmark.index.name = 'token'
        assert_frame_equal(result, benchmark)

    def test_prob_4(self):
        result = self.lda.prob_token_topic(c_topic=['topic_0'])
        benchmark = pd.DataFrame({'topic_0': [1/4., 3/4.]}, index=['w0', 'w1'])
        benchmark.index.name = 'token'
        assert_frame_equal(result, benchmark)

    def test_prob_5(self):
        result = self.lda.prob_token_topic(token=['w0'], c_topic=['topic_0'])
        benchmark = pd.DataFrame({'topic_0': [1/4.]}, index=['w0'])
        benchmark.index.name = 'token'
        assert_frame_equal(result, benchmark)

    def test_prob_6(self):
        result = self.lda.prob_doc_topic(doc=['doc1'], c_topic=['topic_0'])
        benchmark = pd.DataFrame({'topic_0': [1/40.]}, index=['doc1'])
        benchmark.index.name = 'doc'
        assert_frame_equal(result, benchmark)

    def test_prob_7(self):
        result = self.lda.prob_doc_topic(
            doc=['doc1', 'doc2'], c_topic=['topic_0'])
        benchmark = pd.DataFrame(
            {'topic_0': [1/40., 39/40.]}, index=['doc1', 'doc2'])
        benchmark.index.name = 'doc'
        assert_frame_equal(result, benchmark)

    def test_predict_1(self):
        tokenized_text = ['w0', 'w1', 'w0']
        result = self.lda.predict(tokenized_text)
        benchmark = pd.Series({'topic_0': 0.111111 , 'topic_1': 0.888889})
        assert_series_equal(result, benchmark)

    def test_predict_2(self):
        tokenized_text = ['w0', 'w1', 'w0', 'extratoken']
        result = self.lda.predict(tokenized_text)
        benchmark = pd.Series({'topic_0': 0.111111 , 'topic_1': 0.888889})
        assert_series_equal(result, benchmark)

    def test_cosine_similarity_1(self):
        frame = self.lda.pr_topic_g_doc
        result = self.lda.cosine_similarity(frame, frame)
        assert_allclose(np.diag(result.values), 1)

    def test_cosine_similarity_2(self):
        topics = ['topic_0', 'topic_1']
        frame1 = pd.DataFrame({'doc1': [1, 0], 'doc2': [0, 1]}, index=topics)
        frame2 = pd.DataFrame({'doc3': [1, 0]}, index=topics)
        result = self.lda.cosine_similarity(frame1, frame2)
        benchmark = pd.DataFrame({'doc3': [1, 0]}, index=['doc1', 'doc2'])
        assert_frame_equal(result, benchmark.astype(float))

    def test_cosine_similarity_3(self):
        topics = ['topic_0', 'topic_1', 'topic_3']
        frame1 = pd.DataFrame(
            {'doc1': [0.5, 0.5, 0], 'doc2': [0, 0.5, 0.5]}, index=topics)
        frame2 = pd.DataFrame({'doc3': [0.5, 0, 0.5]}, index=topics)
        result = self.lda.cosine_similarity(frame1, frame2)
        benchmark = pd.DataFrame({'doc3': [0.5, 0.5]}, index=['doc1', 'doc2'])
        assert_frame_equal(result, benchmark.astype(float))

    def test_cosine_similarity_4(self):
        topics = ['topic_0', 'topic_1']
        frame1 = pd.DataFrame({'doc1': [1, 0], 'doc2': [0, 1]}, index=topics)
        frame2 = pd.Series({'topic_0': 1, 'topic_1': 0})
        frame2.name = 'doc3'
        result = self.lda.cosine_similarity(frame1, frame2)
        benchmark = pd.DataFrame({'doc3': [1, 0]}, index=['doc1', 'doc2'])
        assert_frame_equal(result, benchmark.astype(float))

    def test_repr(self):
        result = self.lda.__repr__()
        benchmark = 'LDAResults for 2 topics, 2 docs, 2 topics, 2 tokens'
        self.assertEqual(result, benchmark)

    def tearDown(self):
        self.outfile.close()


class TestSFileFilter(unittest.TestCase):
    def setUp(self):
        self.outfile = StringIO()
        formatter = text_processors.VWFormatter()
        self.sff = text_processors.SFileFilter(
            formatter, bit_precision=20, verbose=False)
        self.hash_fun = self.sff._get_hash_fun()

    @property
    def sfile_1(self):
        return StringIO(
            " 1 doc1| word1:1 word2:2\n"
            " 1 doc2| word1:1.1 word3:2")

    def test_load_sfile_fwd_1(self):
        token2id, token_score, doc_freq, num_docs = (
            self.sff._load_sfile_fwd(self.sfile_1))
        self.assertEqual(num_docs, 2)
        self.assertEqual(len(token2id), 3)
        self.assertEqual(token_score, {'word1': 2.1, 'word2': 2, 'word3': 2})
        self.assertEqual(doc_freq, {'word1': 2, 'word2': 1, 'word3': 1})

    def test_set_id2token_1(self):
        # No collisions
        self.sff.token2id = {'one': 1, 'two': 2}
        self.sff.set_id2token()
        benchmark = {1: 'one', 2: 'two'}
        self.assertEqual(self.sff.id2token, benchmark)

    def test_load_sfile_rev_2(self):
        # One collision, both '0' and '100' map to 0
        self.sff.token2id = {0: 1, 1: 2, 2: 3, 3: 1}
        self.sff.set_id2token(seed=1976)
        self.assertEqual(self.sff.id2token, {1: 3, 893658: 0, 3: 2, 2: 1})

    def test_resolve_collisions(self):
        sff = self.sff
        sff.token2id = {'a': 1, 'b': 2, 'c': 3, 'd': 1, 'e': 1}
        sff._resolve_collisions()
        # Check that the dicts are inverses of each other
        token2id_rev = {v: k for k, v in sff.token2id.iteritems()}
        for tok, id_val in sff.token2id.iteritems():
            self.assertEqual(tok, token2id_rev[id_val])

    def check_keys(self, sff, benchmark_key_list):
        all_keys = [
            sff.token2id.keys(), sff.token_score.keys(),
            sff.doc_freq.keys()]

        for keys in all_keys:
            self.assertEqual(set(keys), set(benchmark_key_list))

    def test_load_sfile_1(self):
        self.sff.load_sfile(self.sfile_1)
        self.check_keys(self.sff, ['word1', 'word2', 'word3'])

    def test_filter_tokens(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_tokens('word1')
        self.check_keys(self.sff, ['word2', 'word3'])

    def test_filter_extremes_1(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_extremes(doc_freq_min=2)
        self.check_keys(self.sff, ['word1'])

    def test_filter_extremes_2(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_extremes(doc_freq_max=1)
        self.check_keys(self.sff, ['word2', 'word3'])

    def test_filter_extremes_3(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_extremes(doc_fraction_max=0.5)
        self.check_keys(self.sff, ['word2', 'word3'])

    def test_filter_extremes_4(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_extremes(doc_fraction_min=0.8)
        self.check_keys(self.sff, ['word1'])

    def test_filter_extremes_5(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_extremes(token_score_quantile_max=0.4)
        self.check_keys(self.sff, ['word2', 'word3'])

    def test_filter_sfile_1(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_tokens('word1')
        self.sff.filter_sfile(self.sfile_1, self.outfile)
        result = self.outfile.getvalue()
        benchmark = (
            " 1 doc1| %d:2\n"
            " 1 doc2| %d:2\n" %
            (self.hash_fun('word2'), self.hash_fun('word3')))
        self.assertEqual(result, benchmark)

    def test_filter_sfile_2(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_sfile(
            self.sfile_1, self.outfile, doc_id_list=['doc1'])
        result = self.outfile.getvalue()
        benchmark = (
            " 1 doc1| %d:1 %s:2\n" %
            (self.hash_fun('word1'), self.hash_fun('word2')))
        self.assertEqual(result, benchmark)

    def test_filter_sfile_3(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_tokens('word1')
        self.sff.filter_sfile(
            self.sfile_1, self.outfile, doc_id_list=['doc1'])
        result = self.outfile.getvalue()
        benchmark = (" 1 doc1| %s:2\n" % (self.hash_fun('word2')))
        self.assertEqual(result, benchmark)

    def test_filter_sfile_4(self):
        self.sff.load_sfile(self.sfile_1)
        self.sff.filter_sfile(
            self.sfile_1, self.outfile, doc_id_list=['doc1', 'unseen'],
            enforce_all_doc_id=False)
        result = self.outfile.getvalue()
        benchmark = (
            " 1 doc1| %d:1 %s:2\n" %
            (self.hash_fun('word1'), self.hash_fun('word2')))
        self.assertEqual(result, benchmark)

    def test_filter_sfile_5(self):
        self.sff.load_sfile(self.sfile_1)
        with self.assertRaises(AssertionError) as cm:
            self.sff.filter_sfile(
                self.sfile_1, self.outfile, doc_id_list=['doc1', 'unseen'])

    def test_compactify_1(self):
        self.sff.token2id = {'a': 1, 'b': 100, 'c': 1000}
        self.sff.compactify()
        self.assertEqual(
            self.sff.vocab_size - 1, max(self.sff.token2id.values()))

    def test_compactify_2(self):
        self.sff.token2id = {'a': 1, 'b': 100, 'c': 1000}
        self.sff.set_id2token()
        self.sff.compactify()
        self.assertEqual(
            self.sff.vocab_size - 1, max(self.sff.token2id.values()))

    def tearDown(self):
        self.outfile.close()
