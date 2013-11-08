"""
tests for common, common_abc, etc...
"""
import unittest
from StringIO import StringIO
import sys
from numpy.testing import assert_allclose
from datetime import datetime
import copy

from dspy import common

class TestCommon(unittest.TestCase):
    """
    Tests the common.py module
    """
    def setUp(self):
        self.outfile = StringIO()

    def test_get_list_from_filerows(self):
        infile = StringIO("1\n2\n#3\n\n5")
        result = common.get_list_from_filerows(infile)
        self.assertEqual(result, ['1', '2', '5'])

    def test_write_list_to_filerows(self):
        common.write_list_to_filerows(self.outfile, ['a', 'b'])
        self.assertEqual(self.outfile.getvalue(), 'a\nb\n')

    def test_get_outfile(self):
        result = common.get_outfile(None)
        self.assertTrue(isinstance(result, file))

    def tearDown(self):
        self.outfile.close()
