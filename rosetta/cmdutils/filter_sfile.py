#! python
"""
Filters an sfile, reading from a file or stdin, writing to a file or stdout.
"""
import argparse
import sys

from rosetta.text.text_processors import SFileFilter

def _cli():
    # Text to display after help
    epilog = r"""
    EXAMPLES

    Read from stdin, write to stdout and pipe to vw
    head myfile.vw | filter_sfile.py -s saved_sfile_filter.pkl \
        | vw --lda 5
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
        '-s', '--sfile_filter', required=True,
        help='Load a pickled sfile_filter from this path')

    # Parse and check args
    args = parser.parse_args()

    # Call the module interface
    do_filter(args.infile, args.outfile, args.sfile_filter)


def do_filter(infile, outfile, sfile_filter):
    sfile_filter = SFileFilter.load(sfile_filter)
    sfile_filter.filter_sfile(infile, outfile)


if __name__ == '__main__':
    _cli()
