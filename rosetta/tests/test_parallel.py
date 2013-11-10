import unittest
from functools import partial

import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np

from rosetta.parallel import parallel_easy, pandas_easy


# A couple functions for testing parallel easy
# Must be defined outside of the test class for some reason.
def _abfunc(x, a, b=1):
    return x * a * b
abfunc = partial(_abfunc, 2, 3)


def frame_to_series(frame):
    x = frame.iloc[0, 0]
    return pd.Series([x] * len(frame.columns), index=frame.columns)


def rightmax(mylist):
    return [max(mylist[i: i+2]) for i in range(len(mylist))]


def leftmax(mylist):
    for i in range(len(mylist)):
        if i == 0:
            result = [mylist[0]]
        else:
            result.append(max(mylist[i - 1: i+1]))

    return result


class TestBase(unittest.TestCase):
    """
    Tests the parallel_easy module.
    """
    def setUp(self):
        self.numbers = range(5)
        self.benchmark = [0, 6, 12, 18, 24]

    def test_map_easy_1job(self):
        result = parallel_easy.map_easy(abfunc, self.numbers, 1)
        self.assertEqual(result, self.benchmark)

    def test_map_easy_3job(self):
        result = parallel_easy.map_easy(abfunc, self.numbers, 3)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_1job(self):
        result_iterator = parallel_easy.imap_easy(abfunc, self.numbers, 1, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_3job(self):
        result_iterator = parallel_easy.imap_easy(abfunc, self.numbers, 3, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_n_jobs_wrap_positive(self):
        """
        For n_jobs positive, the wrap should return n_jobs.
        """
        for n_jobs in range(1, 5):
            result = parallel_easy._n_jobs_wrap(n_jobs)
            self.assertEqual(result, n_jobs)

    def test_n_jobs_wrap_zero(self):
        """
        For n_jobs zero, the wrap should raise a ValueError
        """
        self.assertRaises(ValueError, parallel_easy._n_jobs_wrap, 0)


class TestMapEasyPaddedBlock(unittest.TestCase):
    """
    Tests the parallel_easy.map_easy_padded_blocks function.
    """
    def setUp(self):
        #self.numbers_1 = [
        #    0, 0, 2, -1, 4, 2, 6, 7, 6, 9, 12, 11, 11, 14, 55, 55, 44, 33, 33]
        self.numbers_10 = np.random.randint(0, 5, 10)
        self.numbers_101 = np.random.randint(0, 5, 101)
        self.numbers_51 = np.random.randint(0, 5, 101)
        #self.numbers_1 = [0, 1, 2, 0, 3, 2, 4, 3, 2, 3, 3]
        self.n_jobs = 1

    def lefttest(self, numbers, buffer_len, blocksize):
        result = parallel_easy.map_easy_padded_blocks(
            leftmax, numbers, self.n_jobs, buffer_len, blocksize=blocksize)
        benchmark = leftmax(numbers)
        self.assertEqual(result, benchmark)

    def righttest(self, numbers, buffer_len, blocksize):
        result = parallel_easy.map_easy_padded_blocks(
            rightmax, numbers, self.n_jobs, buffer_len, blocksize=blocksize)
        benchmark = rightmax(numbers)
        self.assertEqual(result, benchmark)

    def test_map_easy_padded_blocks_14(self):
        buffer_len = 1
        blocksize = 4
        self.lefttest(self.numbers_10, buffer_len, blocksize)
        self.lefttest(self.numbers_101, buffer_len, blocksize)
        self.lefttest(self.numbers_51, buffer_len, blocksize)
        self.righttest(self.numbers_10, buffer_len, blocksize)
        self.righttest(self.numbers_101, buffer_len, blocksize)
        self.righttest(self.numbers_51, buffer_len, blocksize)

    def test_map_easy_padded_blocks_24(self):
        buffer_len = 2
        blocksize = 4
        self.lefttest(self.numbers_10, buffer_len, blocksize)
        self.lefttest(self.numbers_101, buffer_len, blocksize)
        self.lefttest(self.numbers_51, buffer_len, blocksize)
        self.righttest(self.numbers_10, buffer_len, blocksize)
        self.righttest(self.numbers_101, buffer_len, blocksize)
        self.righttest(self.numbers_51, buffer_len, blocksize)

    def test_map_easy_padded_blocks_37(self):
        buffer_len = 3
        blocksize = 7
        self.lefttest(self.numbers_101, buffer_len, blocksize)
        self.lefttest(self.numbers_51, buffer_len, blocksize)
        self.righttest(self.numbers_101, buffer_len, blocksize)
        self.righttest(self.numbers_51, buffer_len, blocksize)

    def test_map_easy_padded_blocks_17(self):
        buffer_len = 1
        blocksize = 7
        self.lefttest(self.numbers_10, buffer_len, blocksize)
        self.lefttest(self.numbers_101, buffer_len, blocksize)
        self.lefttest(self.numbers_51, buffer_len, blocksize)
        self.righttest(self.numbers_10, buffer_len, blocksize)
        self.righttest(self.numbers_101, buffer_len, blocksize)
        self.righttest(self.numbers_51, buffer_len, blocksize)


class TestPandasEasy(unittest.TestCase):
    """
    Tests the pandas_easy module.
    """
    def setUp(self):
        pass

    def test_groupby_to_scalar_to_series_1(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        benchmark = df.groupby('a').apply(max)
        result = pandas_easy.groupby_to_scalar_to_series(df, max, 1, by='a')
        assert_series_equal(result, benchmark)

    def test_groupby_to_scalar_to_series_2(self):
        s = pd.Series([1, 2, 3, 4])
        labels = ['a', 'a', 'b', 'b']
        benchmark = s.groupby(labels).apply(max)
        result = pandas_easy.groupby_to_scalar_to_series(
            s, max, 1, by=labels)
        assert_series_equal(result, benchmark)

    def test_groupby_to_series_to_frame_1(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        labels = ['g1', 'g1', 'g2']
        benchmark = df.groupby(labels).mean()
        result = pandas_easy.groupby_to_series_to_frame(
            df, np.mean, 1, use_apply=True, by=labels)
        assert_frame_equal(result, benchmark)

    def test_groupby_to_series_to_frame_2(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        labels = ['g1', 'g1', 'g2']
        benchmark = df.groupby(labels).apply(frame_to_series)
        result = pandas_easy.groupby_to_series_to_frame(
            df, frame_to_series, 1, use_apply=False, by=labels)
        assert_frame_equal(result, benchmark)
