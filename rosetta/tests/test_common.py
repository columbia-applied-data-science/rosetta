"""
tests for common, common_abc, etc...
"""
import os
import sys
import unittest
import subprocess
from StringIO import StringIO
from numpy.testing import assert_allclose

from rosetta import common


class TestCommon(unittest.TestCase):
    """
    Tests the common.py module
    """
    def setUp(self):
        self.outfile = StringIO()
        #for testing file_to_txt
        self.test_path = os.path.abspath('./rosetta/tests')
        self.testdata_path = os.path.join(self.test_path, 'data')
        self.testtemp_path = os.path.join(self.test_path, 'temp')
        self.testpdf_path = os.path.join(self.testdata_path, 'test.pdf')
        self.testdoc_path = os.path.join(self.testdata_path, 'test.doc')
        self.testdocx_path = os.path.join(self.testdata_path, 'test.docx')
        self.testtxt_path = os.path.join(self.testdata_path, 'test.txt')

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

    def test_file_to_txt(self):

        if cmd_exists('pdftotext'):
            common.file_to_txt(self.testpdf_path, self.testtemp_path)
            temppdf_path = os.path.join(self.testtemp_path, 'test.txt')
            with open(temppdf_path) as f:
                self.assertTrue(isinstance(f, file))
            os.system('rm %s'%os.path.join(self.testtemp_path, 'test.txt'))
        else:
            sys.stdout.write('Please install unix utility pdftotext')

        if cmd_exists('catdoc'):
            common.file_to_txt(self.testdoc_path, self.testtemp_path)
            tempdoc_path = os.path.join(self.testtemp_path, 'test.txt')
            with open(tempdoc_path) as f:
                self.assertTrue(isinstance(f, file))
            os.system('rm %s'%os.path.join(self.testtemp_path, 'test.txt'))
        else:
            sys.stdout.write('Please install unix utility catdoc')

        common.file_to_txt(self.testpdf_path, self.testtemp_path)
        tempdocx_path = os.path.join(self.testtemp_path, 'test.txt')
        with open(tempdocx_path) as f:
            self.assertTrue(isinstance(f, file))
        os.system('rm %s'%os.path.join(self.testtemp_path, 'test.txt'))


        common.file_to_txt(self.testtxt_path, self.testtemp_path)
        temptxt_path = os.path.join(self.testtemp_path, 'test.txt')
        with open(temptxt_path) as f:
            self.assertTrue(isinstance(f, file))
        os.system('rm %s'%os.path.join(self.testtemp_path, 'test.txt'))


    def tearDown(self):
        self.outfile.close()


def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0
