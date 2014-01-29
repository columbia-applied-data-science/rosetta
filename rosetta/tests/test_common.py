"""
tests for common, common_abc, etc...
"""
import os
import sys
import unittest
from StringIO import StringIO
from numpy.testing import assert_allclose
from collections import defaultdict

from rosetta import common


class TestCommon(unittest.TestCase):
    """
    Tests the common.py module
    """
    def setUp(self):
        self.outfile = StringIO()
        #for testing file_to_txt

    def test_get_list_from_filerows(self):
        infile = StringIO("1\n2\n#3\n\n5")
        result = common.get_list_from_filerows(infile)
        self.assertEqual(result, ['1', '2', '5'])

    def test_write_list_to_filerows(self):
        common.write_list_to_filerows(self.outfile, ['a', 'b'])
        self.assertEqual(self.outfile.getvalue(), 'a\nb\n')

    def test_smart_open_1(self):
        with common.smart_open(sys.stdout, 'w') as f:
            self.assertTrue(isinstance(f, file))

    def test_smart_open_2(self):
        with common.smart_open(StringIO(), 'w') as f:
            self.assertTrue(isinstance(f, StringIO))

    def test_compose_1(self):
        def fun(x):
            return 2 * x
        result = common.compose(fun)(1)
        self.assertEqual(result, 2)

    def test_compose_2(self):
        def fun(x):
            return 2 * x
        result = common.compose(fun, fun)(1)
        self.assertEqual(result, 4)

    def tearDown(self):
        self.outfile.close()


class TestNestedDicts(unittest.TestCase):
    def test_levels1(self):
        ddict = common.nested_defaultdict(int, levels=1)
        self.assertTrue(isinstance(ddict, defaultdict))
        self.assertEqual(ddict['key1'], 0)

    def test_levels2(self):
        ddict = common.nested_defaultdict(int, levels=2)
        self.assertTrue(isinstance(ddict, defaultdict))
        self.assertTrue(isinstance(ddict['k1'], defaultdict))
        self.assertEqual(ddict['k1']['k2'], 0)
        benchmark = {'k1': {'k2': 0}}
        self.assertEqual(ddict, benchmark)

    def test_levels3(self):
        ddict = common.nested_defaultdict(int, levels=3)
        ddict['a1']['a2']['a3'] = 1
        ddict['a1']['b2'] = 2
        benchmark = {'a1': {'a2': {'a3': 1}}}
        benchmark['a1']['b2'] = 2
        self.assertEqual(ddict, benchmark)

    def test_nested_keysearch_1(self):
        adict = {'a': 1}
        self.assertTrue(common.nested_keysearch(adict, ['a']))
        self.assertFalse(common.nested_keysearch(adict, ['A']))

    def test_nested_keysearch_2(self):
        adict = {'a': {'b': 1}}
        self.assertTrue(common.nested_keysearch(adict, ['a', 'b']))
        self.assertFalse(common.nested_keysearch(adict, ['a', 'B']))
        self.assertFalse(common.nested_keysearch(adict, ['A', 'B']))

