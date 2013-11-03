import unittest
from StringIO import StringIO
import sys
from numpy.testing import assert_allclose
from datetime import datetime
import copy

import jrl_utils.src.common as common
from jrl_utils.src.common import BadDataError
import generic.src.generic_filter as generic_filter


"""
To run, from the tests/ directory, type:
$ python -m unittest test_all

OR for verbose output
$ python -m unittest -v test_all

OR to run only the methods in TestFilter
$ python -m unittest test_utils.TestFilter

OR to run only the TestFilter.test_generic_filter method
$ python -m unittest test_utils.TestFilter.test_generic_filter
"""


class TestFilter(unittest.TestCase):
    """
    Tests the implementation (but not the interface) of generic_filter.py
    """
    def setUp(self):
        self.outfile = StringIO()

    def test_generic_filter(self):
        instr = "ian,1,11\r\ndaniel,2,22\r\nchang,3,33"
        infile = StringIO(instr)
        generic_filter.generic_filter(infile, self.outfile)
        result = self.outfile.getvalue()
        benchmark = instr + '\r\n'
        self.assertEqual(result, benchmark)

    def test_modify_row(self):
        row = {'a': 1, 'b': 2}
        generic_filter._modify_row(row)
        benchmark = copy.copy(row)  # For now there is no modification
        self.assertEqual(row, benchmark)

    def test_pop_header(self):
        instr = 'h1,h2\na,b'
        infile = StringIO(instr)
        header = generic_filter._popheader(infile, ',')
        self.assertEqual(header, ['h1', 'h2'])

    def tearDown(self):
        self.outfile.close()
