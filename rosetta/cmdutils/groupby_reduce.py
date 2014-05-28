#!/usr/bin/env python
"""
Streaming implementation of groupby then reduce.  Prints a delimited report
with header e.g.

name|count|sum|mean
smith|1000|239|0.239
jones|100|532|5.32

Note
----
Works in O(N) time but stores unique keys and reduced values in memory.  If you
have K unique keys, O(K) storage is required.
Assumes the first row is a header.  Skips rows with missing entries in either
key_columns or reduce_column.
"""
import argparse
import csv
import sys
from collections import defaultdict

from rosetta import common


def main():
    epilog = r"""

    EXAMPLES
    ---------
    Count the number of unique firstnames in census.csv
    $ groupby_reduce.py -k firstname -r count census.csv

    Count the number of people with different first/last name combinations
    $ groupby_reduce.py -k firstname,lastname -r count census.csv

    Find the total revenue each market sector
    $ groupby_reduce.py -k sector -r sum  marketdata.csv
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    iogroup = parser.add_argument_group('I/O Options')
    iogroup.add_argument(
        'infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
        help='Convert this file.  If not specified, read from stdin.')
    iogroup.add_argument(
        '-o', '--outfile', default=sys.stdout, type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')
    iogroup.add_argument(
        "-d", "--delimiter", default=',',
        help="Use DELIMITER as the column delimiter.  "
        " For tabs use one of -d t  -d tab -d \\t -d '\\t'"
        "[default: %(default)s]")

    columnspecs = parser.add_argument_group('Column choices')
    columnspecs.add_argument(
        "-k", "--key_columns", required=True,
        help="Group according to key_columns.  String or comma-delimited list."
        )
    columnspecs.add_argument(
        "-r", "--reduce_column", default=None,
        help="Column to use for reductions.  If you're only counting then this"
        " defaults to key_columns[0].")

    red = parser.add_argument_group('Reducers')
    red.add_argument(
        "-c", "--count", action='store_true', default=False,
        help="Count the number of non-missing entries in reduce_column for "
        "each group.")
    red.add_argument(
        "-s", "--sum", action='store_true', default=False,
        help="Sum the values in non-missing entries in reduce_column for each"
        " group.")
    red.add_argument(
        "-m", "--mean", action='store_true', default=False,
        help="Get mean value of non-missing entries in reduce_column for each"
        " group.")
    # Parse args
    args = parser.parse_args()

    # Deal with tabs
    if args.delimiter in ['t', '\\t', '\t', 'tab']:
        args.delimiter = '\t'

    # Convert key_columns into a list
    key_columns = args.key_columns.split(',')

    # Put the list of reductions together
    reductions = []
    if args.count:
        reductions.append('count')
    if args.mean:
        reductions.append('mean')
    if args.sum:
        reductions.append('sum')

    ## Call the function that does the real work
    groupby_reduce(
        args.infile, args.outfile, args.delimiter, key_columns,
        args.reduce_column, reductions)


def groupby_reduce(
    infile, outfile, delimiter, key_columns, reduce_column, reductions):
    """
    Write later, if module interface is needed.  For now see main for docs.
    """
    if reductions == ['count'] and reduce_column is None:
        reduce_column = key_columns[0]
    elif reductions:
        assert reduce_column, "If reducing, must supply reduce_column"

    reader = csv.DictReader(infile, delimiter=delimiter)
    keyname = ','.join(key_columns)
    writer = csv.DictWriter(
        outfile, delimiter='|', fieldnames=[keyname] + reductions)
    writer.writeheader()

    store = SmartStore(reductions)

    # Populate store
    for row in reader:
        # Only add keys that are not ''
        keys = [row[k] for k in key_columns if row[k]]
        val = row[reduce_column] if reduce_column else 'NA'
        # Only add rows with entries in key_columns and reduce_column.
        if val and (len(keys) == len(key_columns)):
            store.add(','.join(keys), val)

    # Write results
    for row in store.iterresults():
        row[keyname] = row.pop('key')
        writer.writerow(row)


class SmartStore(object):
    """
    To store needed info from every row, and nothing more.
    """
    def __init__(self, reductions):
        """
        Parameters
        ----------
        reductions : String or list of strings
            Reductions to use.  Determines data stored to some extent.
        """
        if isinstance(reductions, basestring):
            reductions = [reductions]

        self.reductions = reductions

        # To hold the values of final key (floats) and count of occurences.
        # E.g. self.counts = {'key1': {'key2': 55}}
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key, val):
        """
        Add a new key,val pair to appropriate dictionaries.

        Parameters
        ----------
        key : Hashable
        val : Anything
        """
        # Always count the unique values...we don't have to print this.
        self.counts[key] += 1

        # Update sum
        if ('mean' in self.reductions) or ('sum' in self.reductions):
            val = float(val)
            self.sums[key] += val

    def iterresults(self):
        """
        Returns iterator over results.
        """
        for k in self.counts.iterkeys():
            row = {'key': k}

            if 'count' in self.reductions:
                row['count'] = self.counts[k]
            if 'sum' in self.reductions:
                row['sum'] = self.sums[k]
            if 'mean' in self.reductions:
                row['mean'] = self.sums[k] / float(self.counts[k])

            yield row


if __name__ == '__main__':
    main()

