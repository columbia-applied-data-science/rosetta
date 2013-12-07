#!/usr/bin/env python
"""
Streaming version of groupby then reduce.  Prints a tab delimited report with
header e.g.

name    size
smith   1000
jones   543
krasner     2
langmore    1


Note
----
Requires storing keys and reduced values in memory.  Use --lastkey_limit to
protect memory.
"""
import argparse
import csv
import sys


def main():
    epilog = r"""
    Assumes the first row is a header.
    By default reads from stdin, writes to stdout.

    EXAMPLES
    ---------
    Count the number of unique firstnames in census.csv
    $ groupby_reduce.py -k firstname -r size census.csv

    Count the number of people with different first/last name combinations
    $ groupby_reduce.py -k firstname,lastname -r size census.csv

    Find the total revenue each market sector
    $ groupby_reduce.py -k sector -r sum  marketdata.csv
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
        help='Convert this file.  If not specified, read from stdin.')
    parser.add_argument(
        '-o', '--outfile', default=sys.stdout, type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')

    parser.add_argument(
        "-d", "--delimiter", default=',',
        help="Use DELIMITER as the column delimiter.  "
        " For tabs use one of -d t  -d tab -d \\t -d '\\t'"
        "[default: %(default)s]")
    parser.add_argument(
        "-k", "--key_columns", required=True,
        help="Group according to key_columns.  String or comma-delimited list."
        )
    parser.add_argument(
        "-r", "--reducer", default='size',
        help="Either 'size', 'sum', 'mean'.  [default: %(default)s].")
    parser.add_argument(
        "-l", "--lastkey_limit", type=int,
        help="If you have one key, e.g. --key_columns=k1, only store "
        "LASTKEY_LIMIT unique values for it.  In the case of two "
        "key_columns, say --key_columns=k1,k2 , only store the "
        "LASTKEY_LIMIT number of unique values of k2 for every unique k1 (so "
        "all unique values of k1 are stored).") 
    # Parse args
    args = parser.parse_args()

    # Deal with tabs
    if args.delimiter in ['t', '\\t', '\t', 'tab']:
        args.delimiter = '\t'

    # Convert key_columns into a list
    key_columns = args.key_columns.split(',')

    ## Call the function that does the real work
    groupby_reduce(
        args.infile, args.outfile, args.delimiter, key_columns, args.reducer,
        args.lastkey_limit)


def groupby_reduce(
    infile, outfile, delimiter, key_columns, reducer, lastkey_limit):
    """
    Write later, if module interface is needed.
    """
    reader = csv.DictReader(infile, delimiter=delimiter)

    row_store = RowStore(key_columns, reducer, lastkey_limit)

    for row in reader:
        row_store.add(row)


class RowStore(object):
    """
    To store info from every row.
    """
    def __init__(self, key_columns, reducer, lastkey_limit):
        self.key_columns = key_columns
        self.reducer = reducer
        self.lastkey_limit = lastkey_limit

        self.sums = {}
        self.counts = {}

    def add(self, row):
        values = [row[k] for k in key_columns]
        pass

    @property
    def mean(self, key):
        pass


if __name__ == '__main__':
    main()

