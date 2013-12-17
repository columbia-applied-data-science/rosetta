import unittest
from StringIO import StringIO
from numpy.testing import assert_allclose

from rosetta.cmd import concat_csv, join_csv, subsample, cut, row_filter, \
    groupby_reduce


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
            self.infile, self.outfile, 'course', 'contains', 'a', '|', False, 
            False)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n" "analysis|2\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_contains_2(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'alg', '|', False,
            False)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_contains_ignore_case(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'ALG', '|', False,
            True)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_not_contains(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'alg', '|', True,
            False)
        benchmark = 'course|enrollment\r\n' "analysis|2\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_equals(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'contains', 'algebra', '|',
            False, False)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_not_equals(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'equals', 'analysis', '|',
            True, False)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_regex(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'regex', '^alg.[a-z]ra$', '|',
            False, False)
        benchmark = 'course|enrollment\r\n' "algebra|1\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_regex_does_not_match(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'regex', '^alg.[a-z]ra$', '|',
            True, False)
        benchmark = 'course|enrollment\r\n' "analysis|2\r\n"
        self.assertEqual(self.outfile.getvalue(), benchmark)

    def test_regex_ignore_case(self):
        row_filter.filter_file(
            self.infile, self.outfile, 'course', 'regex', '^Alg.[a-z]Ra$', '|',
            False, True)
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


class TestGroupbyReduce(unittest.TestCase):
    """
    Tests groupby_reduce.
    """
    def setUp(self):
        self.outfile = StringIO()
        self.ss_1 = groupby_reduce.SmartStore(['count', 'sum', 'mean'])
        self.ss_2 = groupby_reduce.SmartStore(['count', 'sum'])
        self.ss_3 = groupby_reduce.SmartStore(['count'])
        self.ss_4 = groupby_reduce.SmartStore([])

        self.infile_1 = StringIO(
            "name,cit,age,other\n"
            "ian,us,1,2\n"
            "ian,us,111,2\n"
            "ian,uk,11,\n"
            "dan,fr,2,7\n"
            "dan,uk,22,")

    def test_ss_add_11(self):
        self.ss_1.add('a', 0.1)
        self.assertEqual(self.ss_1.sums, {'a': 0.1})
        self.assertEqual(self.ss_1.counts, {'a': 1})

    def test_ss_12(self):
        self.ss_1.add('a', 0.1)
        self.ss_1.add('a', 0.1)
        self.assertEqual(self.ss_1.sums, {'a': 0.2})
        self.assertEqual(self.ss_1.counts, {'a': 2})

    def test_ss_13(self):
        self.ss_1.add('a', 0.1)
        self.ss_1.add('b', 0.5)
        self.ss_1.add('a', 0.1)
        self.assertEqual(self.ss_1.sums, {'a': 0.2, 'b': 0.5})
        self.assertEqual(self.ss_1.counts, {'a': 2, 'b': 1})

    def test_ss_23(self):
        self.ss_2.add('a', 0.1)
        self.ss_2.add('b', 0.5)
        self.ss_2.add('a', 0.1)
        self.assertEqual(self.ss_2.sums, {'a': 0.2, 'b': 0.5})
        self.assertEqual(self.ss_2.counts, {'a': 2, 'b': 1})

    def test_ss_31(self):
        self.ss_3.add('a', 0.1)
        self.ss_3.add('a', 0.1)
        self.ss_3.add('b', 0.2)
        self.assertEqual(self.ss_3.counts, {'a': 2, 'b': 1})
        self.assertEqual(self.ss_3.sums, {})

    def test_gr_1(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name'],
            'age', ['count', 'mean', 'sum'])
        results = self.outfile.getvalue()
        self.assertEqual(
            results,
            'name|count|mean|sum\r\ndan|2|12.0|24.0\r\nian|3|41.0|'
            '123.0\r\n')

    def test_gr_2(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name', 'cit'],
            'age', ['count', 'mean', 'sum'])
        results = self.outfile.getvalue()
        self.assertEqual(
            results,
            'name,cit|count|mean|sum\r\ndan,uk|1|22.0|22.0\r\nian,uk|1'
            '|11.0|11.0\r\nian,us|2|56.0|112.0\r\ndan,fr|1|2.0|2.0\r\n'
            )

    def test_gr_3(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name'],
            'name', [])
        results = self.outfile.getvalue()
        self.assertEqual(results, 'name\r\ndan\r\nian\r\n')

    def test_gr_4(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name'],
            None, [])
        results = self.outfile.getvalue()
        self.assertEqual(results, 'name\r\ndan\r\nian\r\n')

    def test_gr_4(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name', 'other'],
            'age', ['sum'])
        results = self.outfile.getvalue()
        self.assertEqual(
            results, 'name,other|sum\r\nian,2|112.0\r\ndan,7|2.0\r\n')

    def test_gr_5(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name', 'cit'],
            'other', ['sum'])
        results = self.outfile.getvalue()
        self.assertEqual(
            results, 'name,cit|sum\r\ndan,fr|7.0\r\nian,us|4.0\r\n')

    def test_gr_6(self):
        groupby_reduce.groupby_reduce(
            self.infile_1, self.outfile, ',', ['name', 'cit'],
            'age', ['mean'])
        results = self.outfile.getvalue()
        self.assertEqual(
            results,
            'name,cit|mean\r\ndan,uk|22.0\r\nian,uk|11.0\r\nian,us|56.0\r\ndan'
            ',fr|2.0\r\n')
