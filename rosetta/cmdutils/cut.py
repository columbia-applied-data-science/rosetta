#! python
"""
Reads a csv file or stdin, keeps/removes selected columns.
Prints to stdout or a file.
"""
import argparse
import sys
import csv
# Set the limit to 1 billion columns
#csv.field_size_limit(10000000)

from rosetta import common


def main():
    epilog = r"""

    Examples
    ---------
    Read a comma delimited csv file, data.csv, keep the 'name' column
    $ cut.py -k name test/commafile.csv

    Read a comma delimited csv file, data.csv, remove 'name', 'age' columns
    $ cut.py -r name,age test/commafile.csv
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

    specs = parser.add_mutually_exclusive_group(required=True)
    specs.add_argument(
        "-k", "--keep_list",
        help="Only keep columns in this (comma delimited) list.")
    specs.add_argument(
        "--keep_file",
        help="Only keep columns whose name appears in this "
        "(newline delimited) file. (# lines are comments)")
    specs.add_argument(
        "-r", "--remove_list",
        help="Remove columns in this (comma delimited) list.")
    specs.add_argument(
        "--remove_file",
        help="Remove columns whose name appears in this "
        "(newline delimited) file. (# lines are comments)")

    parser.add_argument(
        "-d", "--delimiter",
        help="Use DELIMITER as the column delimiter in infile."
        " For tabs use one of -d t  -d tab -d \\t -d '\\t'"
        "  [default: %(default)s]", default=',')

    args = parser.parse_args()

    ## Handle the options
    # These 4 (keep/remove options) are enforced as mutually exclusive by
    # argparse
    if args.keep_list:
        keep_list = args.keep_list.split(',')
    elif args.keep_file:
        keep_list = common.get_list_from_filerows(args.keep_file)
    else:
        keep_list = None

    if args.remove_list:
        remove_list = args.remove_list.split(',')
    elif args.remove_file:
        remove_list = common.get_list_from_filerows(args.remove_file)
    else:
        remove_list = None

    # Deal with tabs
    if args.delimiter in ['t', '\\t', '\t', 'tab']:
        args.delimiter = '\t'

    ## Call the function that does the real work
    cut_file(
        args.infile, args.outfile, delimiter=args.delimiter,
        keep_list=keep_list, remove_list=remove_list)


def cut_file(infile, outfile, delimiter=',', keep_list=None, remove_list=None):
    """
    Write later, if module interface is needed.
    """
    assert keep_list or remove_list
    ## Get the csv reader and writer.  Use these to read/write the files.
    reader = csv.reader(infile, delimiter=delimiter)
    writer = csv.writer(outfile, delimiter=delimiter)

    ## Extract the first row of the file
    header = reader.next()

    ## Get and write the new header
    if keep_list:
        new_header = keep_list
    elif remove_list:
        new_header = [item for item in header if item not in set(remove_list)]

    writer.writerow(new_header)

    ## Get the indices in the file that we will keep
    indices_to_keep = [header.index(item) for item in new_header]

    ## Iterate through the file, printing out lines
    for row in reader:
        try:
            new_row = [row[i] for i in indices_to_keep]
            writer.writerow(new_row)
        except IndexError:
            sys.stderr.write(
                "ERROR: Could not write row:\n%s\n" % delimiter.join(row))
            raise


if __name__ == '__main__':
    main()
