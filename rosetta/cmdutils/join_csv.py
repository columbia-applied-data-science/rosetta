#!/usr/bin/env python
"""
Join a list of csv files along indices.  Allows you to specify the indices
for every file, the join type, and missing data fill values.

From pandas, uses DataFrame.from_csv, DataFrame.to_csv, DataFrame.join to do
reads/writes/joins.  Except noted below, the default arguments are used.
"""
import argparse
import sys
import re

import pandas as pd


def _cli():
    # Text to display after help
    epilog = """
    EXAMPLES

    Join three files, use 'doc_id' as index, use 'inner' then 'outer' joins.
    Write to stdout
    $ join_csv.py --index doc_id --how inner outer
        --files file1 file2 file3

    Fill missing values in the 'score' column with 0, and the 'age' column
    with 22.
    $ join_csv.py --index doc_id --how inner --null_fill score,0 age,22
        --files file1 file2 file3
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    required_grp = parser.add_argument_group("Required Arguments")
    required_grp.add_argument(
        '-f', '--files', nargs='*', type=argparse.FileType('r'), required=True,
        help='Concat files in this space separated list')
    required_grp.add_argument(
        '-i', '--index', nargs='*', required=True,
        help='Use this column name (or list of names) as the index or indices.'
        '  Required.')
    required_grp.add_argument(
        '--how', nargs='*', required=True,
        help="How to join files.  'inner', 'outer', 'left', 'right', or a list"
        " of these.  If a list, the files are joined in the order given in "
        "argument FILES, using the specifications in HOW.  Required.")

    optional_grp = parser.add_argument_group("Optional Arguments")
    optional_grp.add_argument(
        '-o', '--outfile', default=sys.stdout,
        type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')
    optional_grp.add_argument(
        '-s', '--sep', default=',',
        help='Delimiter to use.  Regular expressions are accepted.'
        '  [default: %(default)s]')
    optional_grp.add_argument(
        '--null_fill', nargs='*',
        help="After all joins, fill null values using this.  Specified as a "
        "list of pairs, 'name1,value1 name2,value2...', meaning replace  "
        "missing in column with name with value (possibly as float).")
    optional_grp.add_argument(
        '--parse_dates', action='store_true', default=False,
        help="Parse dates.  [default: %(default)s]")

    # Parse and check args
    args = parser.parse_args()

    # Call the module interface
    _join(
        args.outfile, args.files, args.sep, args.index, args.how,
        args.null_fill, args.parse_dates)


def _join(outfile, files, sep, index, how, null_fill, parse_dates):
    """
    Could be made into a module interface.  For now, see _cli for doc.
    """
    # Check/modify args
    # After this call, null_fill will be [(name1, value1),...,(nameN, valueN)]
    null_fill = _parse_null_fill(null_fill)

    # Stretch how to the proper length
    # how, is not used the first time through, so put an undefined in
    if len(how) == 1:
        how = how * (len(files) - 1)
    else:
        assert len(how) == len(files) - 1, \
            "how must be length 1 or len(files) - 1"
    how = ['undefined'] + how

    # Stretch index to the proper length.
    if len(index) == 1:
        index = index * len(files)
    else:
        assert len(index) == len(files), \
            "index must be length 1 or len(files)"

    # Load and join frames
    loop_vars = zip(files, index, how)
    for i, (files_i, index_i, how_i) in enumerate(loop_vars):
        newframe = pd.DataFrame.from_csv(
            files_i, sep=sep, index_col=index_i, parse_dates=parse_dates)
        if i == 0:
            frames = newframe.copy()
        else:
            frames = frames.join(newframe, how=how_i)

    for name, value in null_fill:
        frames[name] = frames[name].fillna(value)

    # Write
    frames.to_csv(outfile, sep=sep)


def _parse_null_fill(null_fill):
    if null_fill is None:
        return []
    # If it is a simple item
    new_null_fill = []
    for fill_pair in null_fill:
        name, value = fill_pair.split(',')
        name = _format_null_fill_name(name)
        value = _format_null_fill_value(value)
        new_null_fill.append((name, value))

    return new_null_fill


def _format_null_fill_name(name):
    if re.sub('[\w_]', '', name) != '':
        raise ValueError("Item must be alpha/numeric/underscore")

    return name


def _format_null_fill_value(value):
    try:
        value = float(value)
    except ValueError:
        if re.sub('[\w_]', '', value) != '':
            raise ValueError("Item must be alpha/numeric/underscore")

    return value


if __name__ == '__main__':
    _cli()
