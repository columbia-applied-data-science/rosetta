#!/usr/bin/env python
from optparse import OptionParser
import sys
import csv
from numpy.random import rand
from numpy.random import seed as randomseed

from .. import common


def main():
    r"""
    DESCRIPTION
    -----------
    Subsample files or stdin and write to stdout.  Optionally subsample in the
    space of different values of a KEY_COLUMN.  When doing this, every time a
    new key value appears, decide whether or not to keep all rows containing
    this value.


    NOTES
    -----
    Assumes the first row is a header.


    EXAMPLES
    ---------
    Subsample a comma delimited dataset and redirect output to a new file
    $ python subsample.py data.csv > subsampled_data.csv

    Subsample, keeping only 10% of rows
    $ python subsample.py -r 0.1 data.csv

    Subsample, keeping 10% of different values in the 'height' column
    $ python subsample.py -r 0.1 -k height data.csv
    """
    usage = "usage: %prog [options] dataset"
    usage += '\n'+main.__doc__
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-r", "--subsample_rate",
        help="Subsample subsample_rate, 0 <= r <= 1.  E.g. r = 0.1 keeps 10% "
        "of rows. [default: %default] ",
        action="store", dest='subsample_rate', type=float, default=0.01)
    parser.add_option(
        "-d", "--delimiter",
        help="Use DELIMITER as the column delimiter.  [default: %default]",
        action="store", dest='delimiter', default=',')
    parser.add_option(
        "-k", "--key_column",
        help="Subsample in the space of values of key_column.  "
        "[default: %default]",
        action="store", dest="key_column", default=None)
    parser.add_option(
        "-s", "--seed",
        help="Integer to seed the random number generator with. "
        "[default: %default] ",
        action="store", dest='seed', type=int, default=None)
    parser.add_option(
        "-o", "--outfilename",
        help="Write to this file rather than stdout.  [default: %default]",
        action="store", dest='outfilename', default=None)

    (opt, args) = parser.parse_args()

    ### Parse args
    # Raise an exception if the length of args is greater than 1
    assert len(args) <= 1
    # If an argument is given, then it is the 'infilename'
    # If no arguments are given, set infilename equal to None
    infilename = args[0] if args else None

    ## Handle the options
    # Deal with tabs
    if opt.delimiter in ['t', '\\t', '\t', 'tab']:
        opt.delimiter = '\t'

    ## Get the infile/outfile
    infile, outfile = common.get_inout_files(infilename, opt.outfilename)

    ## Call the function that does the real work
    subsample(
        infile, outfile, opt.subsample_rate, opt.delimiter,
        opt.key_column, opt.seed)

    ## Close the files iff not stdin, stdout
    common.close_files(infile, outfile)


def subsample(infile, outfile, subsample_rate=0.01, delimiter=',',
    key_column=None, seed=None):
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


if __name__=='__main__':
    main()
