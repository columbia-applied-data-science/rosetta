#!/usr/bin/env python
"""
Removes rows in csv file (or stdin) with header where columns meet
certain criteria.
"""
import argparse
import sys
import csv
import re
# Set the limit to 1 billion columns
#csv.field_size_limit(10000000)


def _cli():
    epilog = r"""

    Examples
    ---------
    Keep rows in curriculum.csv where subject contains 'algebra'
    $ row_filter.py -n subject -c algebra curriculum.csv

    Keep rows in curriculum.csv where subject contains 'algebra' ignoring case
    $ row_filter.py -n subject -c algebra -i curriculum.csv

    Keep rows in curriculum.csv where subject doesn't contain 'algebra'
    $ row_filter.py -n subject -c algebra -v curriculum.csv

    Keep rows in curriculum.csv where subject equals 'algebra'
    $ row_filter.py -n subject -e algebra curriculum.csv

    Keep rows in curriculum.csv where subject doesn't equal 'algebra'
    $ row_filter.py -n subject -e algebra -v curriculum.csv

    Keep rows in curriculum.csv where subject matches regex 'myregex'
    $ row_filter.py -n subject -r myregex curriculum.csv
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
        help=r"Use DELIMITER as the column delimiter in infile."
        " For tabs use one of -d t  -d tab -d \\t -d '\\t'"
        "  [default: %(default)s]", default=',')

    parser.add_argument(
        "-n", "--name", required=True, help="Name of the columm to filter on.")

    spec = parser.add_mutually_exclusive_group(required=True)
    spec.add_argument(
        "-c", "--contains",
        help="Column with name = NAME must contain CONTAINS else we kill that "
        "row. ")
    spec.add_argument(
        "-e", "--equals",
        help="Column with name = NAME must equal EQUALS else we kill that "
        "row. ")
    spec.add_argument(
        "-r", "--regex",
        help="Column with name = NAME must match regex else we kill "
        "that row. ")

    parser.add_argument(
        "-v", "--invert", action='store_true', default=False,
        help="Invert the sense of searching, to get lines that do not meet "
        "criteria.")

    parser.add_argument(
        "-i", "--ignorecase", action='store_true', default=False,
        help="Ignore the case of searched elements")

    args = parser.parse_args()

    # Deal with tabs
    if args.delimiter in ['t', '\\t', '\t', 'tab']:
        args.delimiter = '\t'

    for mode in ['contains', 'equals', 'regex']:
        if args.__dict__[mode]:
            match_str = args.__dict__[mode]
            break

    filter_file(
        args.infile, args.outfile, args.name, mode, match_str, args.delimiter,
        args.invert, args.ignorecase)


def filter_file(infile, outfile, name, mode, match_str, delimiter, invert,
                ignorecase):
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
        'contains': _check_contains, 'equals': _check_equals,
        'regex': _check_regex}

    ## Iterate through the file, printing out lines
    for row in reader:
        if mode_fun[mode](row[name], match_str, ignorecase) != invert:
            writer.writerow(row)


def _check_contains(item, match_str, ignorecase):
    if ignorecase:
        return match_str.lower() in item.lower()
    else:
        return match_str in item


def _check_equals(item, match_str, ignorecase):
    if ignorecase:
        return match_str.lower() == item.lower()
    else:
        return match_str == item


def _check_regex(item, match_str, ignorecase):
    flags = re.IGNORECASE if ignorecase else 0
    return bool(re.search(match_str, item, flags=flags))


if __name__ == '__main__':
    _cli()
