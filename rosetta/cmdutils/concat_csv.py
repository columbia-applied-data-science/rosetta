#!/usr/bin/env python
"""
Concat a list of csv files in an "outer join" style.

From pandas, uses DataFrame.from_csv, DataFrame.to_csv, concat to do
reads/writes/joins.  Except noted below, the default arguments are used.
"""

import argparse
import sys

import pandas as pd


def _cli():
    # Text to display after help
    epilog = """
    EXAMPLES

    Concat two files, each with a header and index, redirect output to newfile
    $ concat_csv.py --index --header file1 file2 > newfile

    Concat two files, write result to newfile
    $ concat_csv.py --index --header -o newfile file1 file2

    Concat all files in mydir/, write result to stdout.
    $ concat_csv.py  mydir/*
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'paths', nargs='*', help='Concat files in this space separated list')
    parser.add_argument(
        '-o', '--outfile', default=sys.stdout,
        type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')
    parser.add_argument(
        '-s', '--sep', default=',',
        help='Delimiter to use.  Regular expressions are accepted.'
        '  [default: %(default)s]')

    parser.add_argument(
        '--index', action='store_true', default=False,
        help='Flag to set if files have an index (leftmost column).'
        ' [default: %(default)s].')
    parser.add_argument(
        '--header', action='store_true', default=False,
        help='Flag to set if files have headers (in top row).  '
        '[default: %(default)s]')

    parser.add_argument(
        '-a', '--axis', type=int, default=0,
        help='Axes along which to concatenate')

    # Parse and check args
    args = parser.parse_args()

    # Call the module interface
    _concat(
        args.outfile, args.paths, args.sep, args.index, args.header, args.axis)


def _concat(outfile, paths, sep, index, header, axis):
    # Read
    index_col = 0 if index else False
    header_row = 0 if header else False
    kwargs = {'sep': sep, 'index_col': index_col, 'header': header_row}
    frames = pd.concat(
        (pd.DataFrame.from_csv(p, **kwargs) for p in paths), axis=axis)

    # Write
    kwargs = {'sep': sep, 'index': index, 'header': header}

    frames.to_csv(outfile, **kwargs)


if __name__ == '__main__':
    _cli()
