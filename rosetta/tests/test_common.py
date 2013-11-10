"""
tests for common, common_abc, etc...
"""
import sys
import unittest
from StringIO import StringIO
from numpy.testing import assert_allclose

from rosetta import common


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

    def test_smart_open_1(self):
        with common.smart_open(sys.stdout, 'w') as f:
            self.assertTrue(isinstance(f, file))

    def test_smart_open_2(self):
        with common.smart_open(StringIO(), 'w') as f:
            self.assertTrue(isinstance(f, StringIO))

    def tearDown(self):
        self.outfile.close()
