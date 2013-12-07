#! python
"""
Converts files into newline separated lists of tokens.
Tokens are represented in Vowpal Wabbit format.
"""
import argparse
from functools import partial
import sys
from random import shuffle

from rosetta.text import filefilter, text_processors
from rosetta.common_abc import SaveLoad

from rosetta.parallel.parallel_easy import imap_easy


def _cli():
    # Text to display after help
    epilog = """
    EXAMPLES

    Convert file1 and file2 to vw format, redirect to my_vw_file
    $ files_to_vw.py file1 file2 > my_vw_file

    Convert all files in mydir/ to vw format
    $ files_to_vw.py  --base_path=mydir

    Convert the first 10 files in mydir/ to vw format
    $ find mydir/ -type f | head | files_to_vw.py

    The supported Vowpal Wabbit format is
    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...
    See: https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io_grp = parser.add_argument_group('I/O group')
    io_grp.add_argument(
        '--base_path', dest='base_path',
        help='Walk this directory for documents.')
    io_grp.add_argument(
        'paths', nargs='*',
        help='Convert files in this space separated list.  If not specified,'
        ' use base_path or read paths from stdin.')
    io_grp.add_argument(
        '-o', '--outfile', dest='outfile', default=sys.stdout,
        type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')
    io_grp.add_argument(
        '--doc_id_level', default=1, type=int,
        help='Form the record doc_id using items this far back in the path'
        ' e.g. if doc_id_level == 2, and path = mydata/1234/3.txt, then we '
        'will have doc_id = 1234_3.  [default: %(default)s]')
    io_grp.add_argument(
        '--no_shuffle', action='store_true', default=False,
        help="Unless this flag is given, paths denoted by --base_path will be "
        "read in random order.")

    tok_grp = parser.add_mutually_exclusive_group(required=False)
    tok_grp.add_argument(
        '--tokenizer_pickle', help="Path to a pickled Tokenizer to load/use")
    tok_grp.add_argument(
        '--tokenizer_type', default='basic',
        help="Use TOKENIZER_TYPE to tokenize the raw text.  Currently "
        "supported:  'basic' .  Use the default tokenizer_type if neither "
        "tokenizer_type or tokenizer_pickle is specified.  "
        "[default: %(default)s]")

    perf_grp = parser.add_argument_group('Performance group')
    perf_grp.add_argument(
        '--n_jobs', help="Use n_jobs to tokenize files.",
        type=int, default=1)
    perf_grp.add_argument(
        '--chunksize', type=int, default=1000,
        help="Have workers process CHUNKSIZE files at a time.  "
        "[default: %(default)s]")

    # Parse and check args
    args = parser.parse_args()

    # If tokenizer_pickle is specified, override type (note:  Can't give both).
    # If neither is specified, then use type == 'basic', which is the default.
    if args.tokenizer_pickle is not None:
        args.tokenizer_type = None

    if args.base_path:
        assert args.paths == []
    elif args.paths == []:
        args.paths = sys.stdin

    # Call the module interface
    tokenize(
        args.outfile, args.paths, args.base_path, args.no_shuffle,
        args.tokenizer_type, args.tokenizer_pickle, args.doc_id_level,
        args.n_jobs, args.chunksize)


def tokenize(
    outfile, paths, base_path, no_shuffle, tokenizer_type, tokenizer_pickle,
    doc_id_level, n_jobs, chunksize):
    """
    Write later if module interface is needed. See _cli for the documentation.
    """
    assert (paths == []) or (base_path is None)

    if base_path:
        paths = filefilter.get_paths(base_path, file_type='*', get_iter=True)
        if no_shuffle is False:
            paths = list(paths)
            shuffle(paths)

    if tokenizer_pickle is not None:
        tokenizer = SaveLoad.load(tokenizer_pickle)
    else:
        tokenizer_dict = {'basic': text_processors.TokenizerBasic}
        tokenizer = tokenizer_dict[tokenizer_type]()

    formatter = text_processors.VWFormatter()

    func = partial(_tokenize_one, tokenizer, formatter, doc_id_level)

    results_iterator = imap_easy(func, paths, n_jobs, chunksize)

    for result in results_iterator:
        outfile.write(result + '\n')


def _tokenize_one(tokenizer, formatter, doc_id_level, path):
    """
    Tokenize file contained in path.  Return results in a sparse format.
    """
    # If path comes from find (and a pipe to stdin), there will be newlines.
    path = path.strip()
    with open(path, 'r') as f:
        text = f.read()

    feature_values = tokenizer.text_to_counter(text)

    # Format
    doc_id = filefilter.path_to_newname(path, name_level=doc_id_level)
    tok_sstr = formatter.get_sstr(feature_values, importance=1, doc_id=doc_id)

    return tok_sstr


if __name__ == '__main__':
    _cli()
