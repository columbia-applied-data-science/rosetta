#!/usr/bin/env python
"""
Streaming version of groupby then reduce.  Prints a tab delimited report with
header e.g.

name    count   sum     mean
smith   1000    239     0.239
jones   100     532     5.32


Note
----
Requires storing keys and reduced values in memory.  Use --lastkey_limit to
protect memory.
"""
import argparse
import csv
import sys
from collections import defaultdict

from rosetta import common


def main():
    epilog = r"""
    Assumes the first row is a header.
    Skips rows with missing entries in either key_columns or reduce_column.
    By default reads from stdin, writes to stdout.

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
        "-r", "--reduce_column", default=None,
        help="Column to use for reductions")
    parser.add_argument(
        "-c", "--count", action='store_true', default=False,
        help="Count the number of non-missing entries in reduce_column for "
        "each group")
    parser.add_argument(
        "-s", "--sum", action='store_true', default=False,
        help="Sum the values in non-missing entries in reduce_column for each"
        " group")
    parser.add_argument(
        "-m", "--mean", action='store_true', default=False,
        help="Get mean value of non-missing entries in reduce_column for each"
        " group")
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
        args.reduce_column, args.lastkey_limit, reductions)


def groupby_reduce(
    infile, outfile, delimiter, key_columns, reduce_column, reductions,
    lastkey_limit):
    """
    Write later, if module interface is needed.
    """
    reader = csv.DictReader(infile, delimiter=delimiter)

    row_store = NestedStore(reductions, lastkey_limit, len(self.key_columns))

    for row in reader:
        # Only add keys that are not ''
        keys = [row[k] for k in self.key_columns if k]
        val = row[reduce_column]
        # Only add rows with entries in key_columns and reduce_column.
        if val and (len(keys) == len(self.key_columns)):
            row_store.add(keys, val)


class NestedStore(object):
    """
    To store info from every row.
    """
    def __init__(self, reductions, lastkey_limit, num_keys):
        """
        Parameters
        ----------
        See the CLI documentation.
        """
        self.reductions = reductions
        self.lastkey_limit = lastkey_limit
        self.num_keys = num_keys

        # To hold the values of final key (floats) and count of occurences.
        # E.g. self.counts = {'key1': {'key2': 55}}
        self.sums = common.nested_defaultdict(float, levels=num_keys)
        self.counts = common.nested_defaultdict(int, levels=num_keys)

    def add(self, keys, val):
        # This recurses through self.counts and finds the lowest level dict.
        # We will do things such as subdict[keys[-1]] += 1
        c_subdict = reduce(lambda d, k: d[k], keys[: -1], self.counts)

        # If the c_subdict already has too many keys, and this is a new key,
        # return.
        if self.lastkey_limit and (len(c_subdict) > self.lastkey_limit):
            if not common.nested_keysearch(self.counts, keys):
                return

        # Always count the unique values...we don't have to print this.
        # Since c_subdict is a defaultdict, this may add a new key.
        c_subdict[keys[-1]] += 1

        # Update sum
        if ('mean' in self.reductions) or ('sum' in self.reductions):
            val = float(val)
            s_subdict = reduce(lambda d, k: d[k], keys[: -1], self.sums)
            s_subdict[keys[-1]] += val

    def results(self):
        """
        Returns dictionary of results.
        """
        keys = self.counts.keys()
        results = {'count': {}}
        self._add_items(self.counts, results['count'])

        if 'sum' in self.reductions or 'mean' in self.reductions:
            results['sum'] = {}
            self._add_items(self.sums, results['sum'])

        if 'mean' in self.reductions:
            results['mean'] = {}
            for k in results['count']:
                count = results['count'][k]
                sum_ = results['sum'][k]
                results['mean'][k] = float(sum_) / count

        return results

    def _add_items(self, ndict, items):
        for k in ndict:
            v = ndict[k]
            if isinstance(v, dict) or isinstance(v, defaultdict):
                self._add_items(v, items)
            else:
                items[k] = v

    @property
    def mean(self, key):
        pass


if __name__ == '__main__':
    main()

