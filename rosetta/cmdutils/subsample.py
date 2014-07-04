#!/usr/bin/env python
"""
Subsample files or stdin and write to stdout.  Optionally subsample in the
space of different values of a KEY_COLUMN.  When doing this, every time a
new key value appears, decide whether or not to keep all rows containing
this value.
"""
import argparse
import csv
import sys
from numpy.random import rand
from numpy.random import seed as randomseed


def main():
    epilog = r"""

    Assumes the first row is a header.

    EXAMPLES
    ---------
    Subsample a comma delimited dataset and redirect output to a new file
    $ subsample.py data.csv > subsampled_data.csv

    Subsample, keeping only 10% of rows
    $ subsample.py -r 0.1 data.csv

    Subsample, keeping 10% of different values in the 'height' column
    $ subsample.py -r 0.1 -k height data.csv
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
        "-r", "--subsample_rate", type=float, default=0.01,
        help="Subsample subsample_rate, 0 <= r <= 1.  E.g. r = 0.1 keeps 10 "
        "percent of rows. [default: %(default)s] ")
    parser.add_argument(
        "-d", "--delimiter", default=',',
        help="Use DELIMITER as the column delimiter.  "
        " For tabs use one of -d t  -d tab -d \\t -d '\\t'"
        "[default: %(default)s]")
    parser.add_argument(
        "-k", "--key_column",
        help="Subsample in the space of values of key_column.  ")

    parser.add_argument(
        "-s", "--seed", type=int,
        help="Integer to seed the random number generator with. ")

    # Parse args
    args = parser.parse_args()

    # Deal with tabs
    if args.delimiter in ['t', '\\t', '\t', 'tab']:
        args.delimiter = '\t'

    ## Call the function that does the real work
    subsample(
        args.infile, args.outfile, args.subsample_rate, args.delimiter,
        args.key_column, args.seed)


def subsample(
    infile, outfile, subsample_rate=0.01, delimiter=',', key_column=None,
    seed=None):
    """
    Write later, if module interface is needed.
    """
    ## Seed the random number generator for deterministic results
    if seed:
        randomseed(seed)

    ## Get the csv reader and writer.  Use these to read/write the files.
    reader = csv.DictReader(infile, delimiter=delimiter)
    writer = csv.DictWriter(
        outfile, delimiter=delimiter, fieldnames=reader.fieldnames)
    writer.writeheader()

    ## Iterate through the file and print a selection of rows
    if key_column:
        _subsample_using_keys(reader, writer, subsample_rate, key_column)
    else:
        _subsample_without_keys(reader, writer, subsample_rate)


def _subsample_without_keys(reader, writer, subsample_rate):
    for row in reader:
        if subsample_rate > rand():
            writer.writerow(row)


def _subsample_using_keys(reader, writer, subsample_rate, key_column):
    """
    Iterate through reader, for every new value in key_column, decide whether
    or not to print ALL rows with that value.
    """
    keys_to_use = set()
    keys_to_not_use = set()

    for row in reader:
        key_value = row[key_column]

        if key_value in keys_to_use:
            writer.writerow(row)
        elif key_value not in keys_to_not_use:
            if subsample_rate > rand():
                keys_to_use.add(key_value)
                writer.writerow(row)
            else:
                keys_to_not_use.add(key_value)


if __name__ == '__main__':
    main()
