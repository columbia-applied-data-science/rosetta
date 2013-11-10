#!/usr/bin/env python
"""
Removes rows in csv file (or stdin) with header where columns meet
certain criteria.
"""
import argparse
import sys
import csv
# Set the limit to 1 billion columns
#csv.field_size_limit(10000000)


def _cli():
    epilog = r"""

    Examples
    ---------
    Keep rows in curriculum.csv where subject contains 'algebra'
    $ python row_filter.py -n subject -C algebra curriculum.csv

    Keep rows in curriculum.csv where subject doesn't contain 'algebra'
    $ python row_filter.py -n subject -c algebra curriculum.csv

    Keep rows in curriculum.csv where subject equals 'algebra'
    $ python row_filter.py -n subject -E algebra curriculum.csv

    Keep rows in curriculum.csv where subject doesn't equal 'algebra'
    $ python row_filter.py -n subject -e algebra curriculum.csv
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
        "-d", "--delimiter",
        help="Use DELIMITER as the column delimiter in infile."
        "  [default: %(default)s]", default=',')

    parser.add_argument(
        "-n", "--name", required=True, help="Name of the columm to filter on.")

    spec = parser.add_mutually_exclusive_group(required=True)
    spec.add_argument(
        "-C", "--contains",
        help="Column with name = NAME must contain CONTAINS else we kill that "
        "row. ")
    spec.add_argument(
        "-E", "--equals",
        help="Column with name = NAME must equal EQUALS else we kill that "
        "row. ")
    spec.add_argument(
        "-c", "--not_contains",
        help="Column with name = NAME must not contain NOTCONTAINS else we "
        "kill that row.")
    spec.add_argument(
        "-e", "--not_equals",
        help="Column with name = NAME must not equal NOTEQUALS else we kill "
        "that row. ")

    args = parser.parse_args()

    for mode in ['contains', 'equals', 'not_contains', 'not_equals']:
        if args.__dict__[mode]:
            match_str = args.__dict__[mode]
            break

    filter_file(
        args.infile, args.outfile, args.name, mode, match_str, args.delimiter)


def filter_file(infile, outfile, name, mode, match_str, delimiter):
    """
    Module interface.  See _cli for doc.  Add doc later if needed.
    """
    ## Get the csv reader and writer.  Use these to read/write the files.
    # reader.fieldnames gives you the header
    reader = csv.DictReader(infile, delimiter=delimiter)
    writer = csv.DictWriter(
        outfile, delimiter=delimiter, fieldnames=reader.fieldnames)
    writer.writeheader()

    mode_fun = {
        'contains': _check_contains, 'not_contains': _check_not_contains,
        'equals': _check_equals, 'not_equals': _check_not_equals}

    ## Iterate through the file, printing out lines
    for row in reader:
        if mode_fun[mode](row[name], match_str):
            writer.writerow(row)


def _check_contains(item, match_str):
    return match_str in item


def _check_not_contains(item, match_str):
    return not _check_contains(item, match_str)


def _check_equals(item, match_str):
    return match_str == item


def _check_not_equals(item, match_str):
    return not _check_equals(item, match_str)


if __name__ == '__main__':
    _cli()
