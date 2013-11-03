"""
Removes rows in csv file (or stdin) with header where columns meet
certain criteria.
"""
import argparse
import sys
import csv
# Set the limit to 1 billion columns
#csv.field_size_limit(10000000)

import dspy.common as common# import common


def _cli():
    epilog = r"""

    Examples
    ---------
    Keep rows in curriculum.csv where the subject contains the word 'algebra'
    $ python row_filter.py -n subject -C algebra curriculum.csv

    Keep rows in curriculum.csv where the subject doesn't contain the word 'algebra'
    $ python row_filter.py -n subject -c algebra curriculum.csv

    Keep rows in curriculum.csv where the subject equals the word 'algebra'
    $ python row_filter.py -n subject -E algebra curriculum.csv

    Keep rows in curriculum.csv where the subject doesn't equal the word 'algebra'
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
        help="Use DELIMITER as the column delimiter.  [default: %(default)s]",
        default=',')
    parser.add_option(
        "-n", "--name",
        help="Name of the columm to filter on.  [default: %default]",
        action="store", dest='name', default=None)
    parser.add_option(
        "-C", "--contains",
        help="Column with name = NAME must contain CONTAINS else we kill that row. "
        "[default: %default]", 
        action='store', dest='contains', default=None)
    parser.add_option(
        "-E", "--equals",
        help="Column with name = NAME must equal EQUALS else we kill that row. "
        "[default: %default]", 
        action='store', dest='equals', default=None)
    parser.add_option(
        "-e", "--notequals",
        help="Column with name = NAME must not equal NOTEQUALS else we kill that row. "
        "[default: %default]", 
        action='store', dest='notequals', default=None)
    parser.add_option(
        "-c", "--notcontains",
        help="Column with name = NAME must not contain NOTCONTAINS else we kill that row."
        "  [default: %default]", 
        action='store', dest='notcontains', default=None)
    parser.add_option(
        "-o", "--outfilename",
        help="Write to this file rather than stdout.  [default: %default]",
        action="store", dest='outfilename', default=None)

    (opt, args) = parser.parse_args()

    ### Parse args
    infilename = args[0] if args else None

    infile, outfile = common.get_inout_files(infilename, opt.outfilename, outmode='wb')

    column_filter(infile, outfile, opt.delimiter, opt)

    common.close_files(infile, outfile)


def column_filter(infile, outfile, delimiter, opt):
    """
    NOTE:  Written late at night after drinking...should be refactored!
    """
    ## Get the csv reader and writer.  Use these to read/write the files.
    # reader.fieldnames gives you the header
    reader = csv.DictReader(infile, delimiter=delimiter)
    writer = csv.DictWriter(outfile, delimiter=delimiter, fieldnames=reader.fieldnames)
    writer.writeheader()

    ## Iterate through the file, printing out lines 
    for row in reader:
        content = row[opt.name]
        if _shouldwrite(content, opt):
            writer.writerow(row)


def _shouldwrite(content, opt):
    if opt.equals and content:
        shouldwrite = content == opt.equals
    elif opt.contains and content:
        shouldwrite = opt.contains in content
    elif opt.notequals:
        if not content:
            shouldwrite = True
        else: 
            shouldwrite = content != opt.notequals
    elif opt.notcontains:
        if not content:
            shouldwrite = True
        else:
            shouldwrite = opt.notcontains not in content
    else:
        raise ValueError(
            "Unable to determine what to filter.  options = %s" % opt.__dict__)

    return shouldwrite


if __name__=='__main__':
    _cli()
