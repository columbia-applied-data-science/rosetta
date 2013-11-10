import unittest
from StringIO import StringIO
from numpy.testing import assert_allclose

from rosetta.cmd import concat_csv, join_csv, subsample, cut, row_filter


"""
To run, from the rosetta/ directory, type:
    make test
"""


class TestCut(unittest.TestCase):
    """
    Tests the implementation (but not the interface) of cut.py
    """
    def setUp(self):
        self.outfile = StringIO()

        commastring = \
            "name,age,weight\r\nian,1,11\r\ndaniel,2,22\r\nchang,3,33"
        self.commafile = StringIO(commastring)

        pipestring = \
            "name|age|weight\r\nian|1|11\r\ndaniel|2|22\r\nchang|3|33"
        self.pipefile = StringIO(pipestring)

    def test_cut_file_keepname(self):
        cut.cut_file(self.commafile, self.outfile, keep_list=['name'])
        result = self.outfile.getvalue()
        self.assertEqual('name\r\nian\r\ndaniel\r\nchang\r\n', result)

    def test_cut_file_removename(self):
        cut.cut_file(self.commafile, self.outfile, remove_list=['name'])
        result = self.outfile.getvalue()
        self.assertEqual('age,weight\r\n1,11\r\n2,22\r\n3,33\r\n', result)

    def test_cut_file_keepnameage(self):
        cut.cut_file(self.commafile, self.outfile, keep_list=['name', 'age'])
        result = self.outfile.getvalue()
        self.assertEqual(
            'name,age\r\nian,1\r\ndaniel,2\r\nchang,3\r\n', result)

    def test_cut_file_keepagename(self):
        cut.cut_file(self.commafile, self.outfile, keep_list=['age', 'name'])
        result = self.outfile.getvalue()
        self.assertEqual(
            'age,name\r\n1,ian\r\n2,daniel\r\n3,chang\r\n', result)

    def test_cut_file_keepagename_pipe(self):
        cut.cut_file(
            self.pipefile, self.outfile, keep_list=['age', 'name'],
            delimiter='|')
        result = self.outfile.getvalue()
        self.assertEqual(
            'age|name\r\n1|ian\r\n2|daniel\r\n3|chang\r\n', result)

    def tearDown(self):
        self.outfile.close()


class TestRowFilter(unittest.TestCase):
    """
    Tests the implementation (but not the interface) of row_filter.py
    """
    def setUp(self):
        self.outfile = StringIO()
        self.infile = StringIO(
            "course|enrollment\n" "algebra|1\n" "analysis|2\n")

    def test_contains_1(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'a', '|')
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n" "analysis|2\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_contains_2(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'alg', '|')
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_not_contains(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'not_contains', 'alg', '|')
        benchmark = 'course|enrollment\r\n' "analysis|2\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_equals_1(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'algebra', '|')
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_not_equals_1(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'not_equals', 'analysis', '|')
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def tearDown(self):
        self.outfile.close()


class TestSubsample(unittest.TestCase):
    """
    Tests the subsampler
    """
    def setUp(self):
        self.outfile = StringIO()
        self.commafile = StringIO(
            'name,age,weight\nian,1,11\ndaniel,2,22\nchang,3,33')
        self.pipefile = StringIO(
            'name|age|weight\nian|1|11\ndaniel|2|22\nchang|3|33')
        self.longfile = StringIO(
            'name,age,weight\nian,1,11\ndaniel,2,22\nian,1b,11b\nchang,3,33'
            '\ndaniel,2b,22b\nchang,3b,33b')
        self.seed = 1234

    def test_r0p0_comma(self):
        subsample.subsample(
            self.commafile, self.outfile, subsample_rate=0.0, seed=self.seed)
        result = self.outfile.getvalue()
        benchmark = 'name,age,weight\r\n'
        self.assertEqual(result, benchmark)

    def test_r0p5_comma(self):
        subsample.subsample(
            self.commafile, self.outfile, subsample_rate=0.5, seed=self.seed)
        result = self.outfile.getvalue()
        benchmark = 'name,age,weight\r\nian,1,11\r\nchang,3,33\r\n'
        self.assertEqual(result, benchmark)

    def test_r0p5_pipe(self):
        subsample.subsample(
            self.pipefile, self.outfile, subsample_rate=0.5, seed=self.seed)
        result = self.outfile.getvalue()
        benchmark = 'name|age|weight\r\nian|1|11\r\nchang|3|33\r\n'
        self.assertEqual(result, benchmark)

    def test_r0p5_keyname_comma(self):
        subsample.subsample(
            self.longfile, self.outfile, subsample_rate=0.5,
            key_column='name', seed=self.seed)
        result = self.outfile.getvalue()
        benchmark = 'name,age,weight\r\nian,1,11\r\nian,1b,11b\r\nchang,3,33'\
            '\r\nchang,3b,33b\r\n'
        self.assertEqual(result, benchmark)

    def tearDown(self):
        self.outfile.close()


class TestConcatCSV(unittest.TestCase):
    """
    Tests concat_csv.py
    """
    def setUp(self):
        self.outfile = StringIO()
        self.file1 = StringIO("name,age\n" + "ian,1\n" + "daniel,2")
        self.file2 = StringIO("name,height\n" + "ian,11\n" + "daniel,22")

    def test_concat_header_index(self):
        paths = [self.file1, self.file2]
        sep = ','
        index = True
        header = True
        axis = 1
        concat_csv._concat(self.outfile, paths, sep, index, header, axis)
        result = self.outfile.getvalue()
        benchmark = "name,age,height\n" + "ian,1,11\n" + "daniel,2,22\n"
        self.assertEqual(result, benchmark)

    def test_concat_header(self):
        paths = [self.file1, self.file2]
        sep = ','
        index = False
        header = True
        axis = 1
        concat_csv._concat(self.outfile, paths, sep, index, header, axis)
        result = self.outfile.getvalue()
        benchmark = (
            "name,age,name,height\n" + "ian,1,ian,11\n"
            + "daniel,2,daniel,22\n")
        self.assertEqual(result, benchmark)

    def test_concat_index(self):
        paths = [self.file1, self.file2]
        sep = ','
        index = True
        header = False
        axis = 1
        concat_csv._concat(self.outfile, paths, sep, index, header, axis)
        result = self.outfile.getvalue()
        benchmark = "ian,1,11\n" + "daniel,2,22\n"
        self.assertEqual(result, benchmark)

    def test_concat(self):
        paths = [self.file1, self.file2]
        sep = ','
        index = False
        header = False
        axis = 1
        concat_csv._concat(self.outfile, paths, sep, index, header, axis)
        result = self.outfile.getvalue()
        benchmark = "ian,1,ian,11\n" + "daniel,2,daniel,22\n"
        self.assertEqual(result, benchmark)

    def tearDown(self):
        self.outfile.close()


class TestJoinCSV(unittest.TestCase):
    """
    Tests join_csv.py
    """
    def setUp(self):
        self.outfile = StringIO()
        self.file1 = StringIO("name,age\n" + "ian,1\n" + "daniel,2")
        self.file2 = StringIO("name,height\n" + "ian,11\n" + "joe,22")
        self.file3 = StringIO("myname,weight\n" + "ian,111\n" + "joe,222")
        self.file4 = StringIO("name,job\n" + "ian,tacobell\n" + "joe,scrub")

    def test_outer_12_null_fill(self):
        files = [self.file1, self.file2]
        sep = ','
        index = ['name']
        how = ['outer']
        parse_dates = False
        null_fill = ['age,100', 'height,200']
        join_csv._join(
            self.outfile, files, sep, index, how, null_fill, parse_dates)
        result = self.outfile.getvalue()
        benchmark = (
            'name,age,height\ndaniel,2.0,200.0\nian,1.0,11.0\n'
            'joe,100.0,22.0\n')
        self.assertEqual(result, benchmark)

    def test_outer_14_null_fill(self):
        files = [self.file1, self.file4]
        sep = ','
        index = ['name']
        how = ['outer']
        parse_dates = False
        null_fill = ['job,unemployed']
        join_csv._join(
            self.outfile, files, sep, index, how, null_fill, parse_dates)
        result = self.outfile.getvalue()
        benchmark = (
            'name,age,job\n' 'daniel,2.0,unemployed\n' 'ian,1.0,tacobell\n'
            'joe,,scrub\n')
        self.assertEqual(result, benchmark)

    def test_inner_12(self):
        files = [self.file1, self.file2]
        sep = ','
        index = ['name']
        how = ['inner']
        parse_dates = False
        null_fill = []
        join_csv._join(
            self.outfile, files, sep, index, how, null_fill, parse_dates)
        result = self.outfile.getvalue()
        benchmark = 'name,age,height\nian,1,11\n'
        self.assertEqual(result, benchmark)

    def test_inner_123(self):
        files = [self.file1, self.file2, self.file3]
        sep = ','
        index = ['name', 'name', 'myname']
        how = ['inner']
        parse_dates = False
        null_fill = []
        join_csv._join(
            self.outfile, files, sep, index, how, null_fill, parse_dates)
        result = self.outfile.getvalue()
        benchmark = ',age,height,weight\n' + 'ian,1,11,111\n'
        self.assertEqual(result, benchmark)

    def test_left_12(self):
        files = [self.file1, self.file2]
        sep = ','
        index = ['name', 'name']
        how = ['left']
        parse_dates = False
        null_fill = []
        join_csv._join(
            self.outfile, files, sep, index, how, null_fill, parse_dates)
        result = self.outfile.getvalue()
        benchmark = 'name,age,height\n' + 'ian,1,11.0\n' + 'daniel,2,\n'
        self.assertEqual(result, benchmark)

    def tearDown(self):
        self.outfile.close()
