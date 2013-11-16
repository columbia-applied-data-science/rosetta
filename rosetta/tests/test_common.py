"""
tests for common, common_abc, etc...
"""
import os
import sys
import unittest
from StringIO import StringIO
from numpy.testing import assert_allclose

from rosetta import common, tests


class TestCommon(unittest.TestCase):
    """
    Tests the common.py module
    """
    def setUp(self):
        self.outfile = StringIO()
        #for testing file_to_txt
        self.test_path = os.path.split(tests.__file__)[0]
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

    @unittest.skip("Skipping test until linux problem fixed")
    def test_file_to_txt(self):
        #import pdb; pdb.set_trace()

        common.file_to_txt(self.testpdf_path, self.testtemp_path)
        temppdf_path = os.path.join(self.testtemp_path, 'test.pdf')
        with open(temppdf_path, 'w') as f:
            self.assertTrue(isinstance(f, file))
        
        common.file_to_txt(self.testdoc_path, self.testtemp_path)
        tempdoc_path = os.path.join(self.testtemp_path, 'test.doc')
        with open(tempdoc_path, 'w') as f:
            self.assertTrue(isinstance(f, file))
        
        common.file_to_txt(self.testpdf_path, self.testtemp_path)
        tempdocx_path = os.path.join(self.testtemp_path, 'test.docx')
        with open(tempdocx_path, 'w') as f:
            self.assertTrue(isinstance(f, file))

        common.file_to_txt(self.testtxt_path, self.testtemp_path)
        temptxt_path = os.path.join(self.testtemp_path, 'test.txt')
        with open(temptxt_path, 'w') as f:
            self.assertTrue(isinstance(f, file))

    def tearDown(self):
        self.outfile.close()
        os.system('rm %s'%os.path.join(self.testtemp_path, 'test*'))



