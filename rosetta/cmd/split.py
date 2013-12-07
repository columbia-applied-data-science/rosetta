#! python
"""
Splits a file or stdin into multiple files.  Assumes a header.

Allows grouping by a key column and then sending identical groups to
the same file, possibly keeping only one member of each group.  To do
this, the data must first be sorted by the key column.

Automatically names output files by appending e.g. _split-0-40, _split-1-60
for a 40/60 split filename.
"""
from optparse import OptionParser
import sys
import csv
import scipy as sp
from itertools import groupby

from random import choice


#### Global variables

def main():
    r"""
    DESCRIPTION
    -----------
    Splits a file or stdin into multiple files.  Assumes a header.

    Allows grouping by a key column and then sending identical groups to
    the same file, possibly keeping only one member of each group.  To do
    this, the data must first be sorted by the key column.

    Automatically names output files by appending e.g. _split-0-40, _split-1-60
    for a 40/60 split filename.

    EXAMPLES
    --------
    $ split.py data.csv
    will produce the files data_split-0-50.csv and data_split-1-50.csv

    $ split.py -r 60/20/20 data.csv
    will produce a 60/20/20 split

    $ split.py --ratio=60/40 -k -s name dataset.csv
    will produce a 60/40 split and keep only one record from each group, with
    the "name" column determining the groups.
    """
    usage = "usage: %prog [options] dataset"
    usage += '\n'+main.__doc__
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-r", "--ratio",
        help="Ratio for the split, e.g. 50/50 [default: %default] ",
        action="store", dest='ratio', default='50/50')
    parser.add_option(
        "-b", "--base_name",
        help="Base name to use for the outfile rather than the dataset "
        "filename.  [default: %default] ",
        action="store", dest='base_name', default=None)
    parser.add_option(
        "-k", "--key_column",
        help="String (in the header) defining the key column. "
        "[default: %default].  Needed only if you want to use the "
        "grouping functionality.",
        action="store", dest='key_column', default=None)
    parser.add_option(
        "-k", "--keeprandomone",
        help="Keep only one record from each group.  Choose the record "
        "randomly. [default: %default] ",
        action="store_true", dest='keeprandomone', default=False)

    (options, args) = parser.parse_args()

    ### Parse args
    assert len(args) <= 1
    if args:
        file_to_read = args[0]
    else:
        file_to_read = 'stdin'

    if not options.base_name:
        options.base_name = file_to_read

    ### Call the function that does the real work
    split(file_to_read, **options.__dict__)


def split(
    file_to_read, base_name=None, ratio='50/50', key_column=None,
    keeprandomone=False):
    """
    Write later, if module interface is needed.
    """
    ## Get the reader
    if file_to_read == 'stdin':
        reader = csv.reader(sys.stdin)
    else:
        f = open(file_to_read, 'r')
        reader = csv.reader(f, delimiter=',')
    ## Get the monkeys (writers)
    # The csv writer holder
    monkeys = Monkeys(base_name, ratio)

    ## Extract, modify, and write the header
    header = reader.next()
    # If we're using a file to select variables, then replace header
    monkeys.writerow_all(header)

    ## Define the function that gets the key for each group
    # If it is None, then groupby returns each line unchanged.
    if key_column:
        key_idx = header.index(key_column)
        keyfunc = lambda row: row[key_idx]
    else:
        keyfunc = None

    ## Iterate through the file and print a selection of rows
    for key, group in groupby(reader, key=keyfunc):
        current_monkey = monkeys.select(sp.rand())
        process_group(
            group, current_monkey, keeprandomone=keeprandomone)

    ## Close the reader
    if file_to_read != 'stdin':
        f.close()


class Monkeys(object):
    """
    Holds a list of csv writers (monkeys).  Gives access to group writing.
    Files (to be written to) are open in 'w' mode until this Monkey's instance
    is deleted.
    """
    def __init__(self, base_name, split_ratio):
        base_name_nocsv = base_name.replace('.csv', '')
        # The fractions that we use to split files
        split_ints = split_ratio.split('/')
        self.split_nums = [0.01*int(p) for p in split_ints]
        assert sum(self.split_nums) == 1, "Split ratio must add up to 1"
        self.num_monkeys = len(self.split_nums)

        self.monkey_list = []
        self.file_list = []
        for m in xrange(len(self.split_nums)):
            file_name = base_name_nocsv + '_split-' + str(m) + '-' \
                + split_ints[m] +'.csv'
            f = open(file_name, 'w')
            writer = csv.writer(f, delimiter=',')
            self.monkey_list.append(writer)
            self.file_list.append(f)

    def __del__(self):
        if hasattr(self, 'file_list'):
            for f in self.file_list:
                f.close()

    def select(self, u):
        split_sum = 0.0
        for m, num in enumerate(self.split_nums):
            split_sum += num
            if split_sum > u:
                return self.monkey_list[m]

    def writerow_all(self, row):
        for m in self.monkey_list:
            m.writerow(row)


def process_group(group, writer, keeprandomone=False, keepfirstone=False):
    """
    Process each group, taking appropriate printing action.

    Parameters
    ----------
    group : itertools.grouper
    writer : csv.writer
    keeprandomone : Boolean, optional
        If True, write only one (randomly chosen) member of the group
    keepfirstone : Boolean, optional
        If True, keep only the first record in each group
    """
    group_list = list(group)
    if keeprandomone:
        writer.writerow(choice(group_list))
    elif keepfirstone:
        writer.writerow(group_list[0])
    else:
        writer.writerows(group_list)


if __name__ == '__main__':
    main()
