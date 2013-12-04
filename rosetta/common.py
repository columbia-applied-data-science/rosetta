"""
Common functions/classes for dataprep.
"""
import numpy as np
import cPickle
import itertools
import os
import re
import sys
import shutil
import subprocess

from docx import opendocx, getdocumenttext
from unidecode import unidecode

###############################################################################
# Decorators
###############################################################################


def lazyprop(fn):
    """
    Use as a decorator to get lazily evaluated properties.
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


###############################################################################
# Wrappers for opening/closing files
###############################################################################


class smart_open(object):
    """Context manager that opens a filename and closes it on exit, but does
    nothing for file-like objects.
    """
    def __init__(self, filename, *args):
        """
        The exact same call structure as the built-in function 'open'

        Parmeters
        ---------
        filename : filepath, buffer, or StringIO
        args : Optional args
            First arg will be 'mode', e.g. 'r', 'rb', 'w'
            Second arg will be 'buffering', read the docs for open
        """
        if isinstance(filename, basestring):
            self.fh = open(filename, *args)
            self.closing = True
        else:
            self.fh = filename
            self.closing = False

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closing:
            self.fh.close()

        return False


###############################################################################
# Functions to read special file formats
###############################################################################

def get_list_from_filerows(infile):
    """
    Returns a list generated from rows of a file.

    Parameters
    ----------
    infile : File buffer or path
        Lines starting with # are comments
        Blank lines and leading/trailing whitespace are ignored
        Other lines will be converted to a string and appended to a
        list.
    """
    with smart_open(infile, 'rb') as f:
        kpv_list = []
        for line in f:
            # Strip whitespace
            line = line.strip()
            # Skip empty lines
            if len(line) > 0:
                # If the line isn't a comment
                # Append the content to the list
                if line[0] != '#':
                    kpv_list.append(line.rstrip('\n'))

    return kpv_list


def write_list_to_filerows(outfile, mylist):
    """
    The inverse of get_list_from_filerows.

    Parameters
    ----------
    outfile : filepath or buffer
    mylist : List
    """
    with smart_open(outfile, 'wb') as f:
        for item in mylist:
            f.write(str(item) + '\n')


def pickleme(obj, pkl_file, protocol=2):
    """
    Save obj to disk using cPickle.

    Parameters
    ----------
    obj : Serializable Python object
    pkl_file : filepath or buffer
        File to store obj to
    protocol : 0, 1, or 2
        2 is fastest
    """
    with smart_open(pkl_file, 'w') as f:
        cPickle.dump(obj, f, protocol=protocol)


def unpickleme(pkl_file):
    """
    Returns unpickled version of object.

    Parameters
    ----------
    pkl_file : filepath or buffer
        We will attempt to unpickle this file.
    """
    with smart_open(pkl_file, 'r') as f:
        return cPickle.load(f)


def get_structured_array(listoflists, schema, dropmissing=False):
    """
    Uses schema to convert listoflists to a structured array.

    Parameters
    ----------
    listoflists : List of lists
    schema : List of tuples
        E.g. [(var1, type1),...,(varK, typeK)]
    dropmissing : Boolean
        If True, drop rows that contain missing values
    """
    ## First convert listoflists to a list of tuples...
    # TODO : This CAN'T actually be necessary..find another way
    if dropmissing:
        tuple_list = [tuple(row) for row in listoflists if '' not in row]
    else:
        tuple_list = [tuple(row) for row in listoflists]

    return np.array(tuple_list, schema)


###############################################################################
# Custom Exceptions
###############################################################################


class BadDataError(Exception):
    """
    Dummy class that is exactly like the Exception class.  Used to make sure
    people are raising the intended exception, rather than some other wierd
    one.
    """
    pass


class TokenError(Exception):
    """
    Raise when tokens are passed to a method/function and you don't know how
    to deal with them.
    """
    pass


class ConfigurationSyntaxError(Exception):
    """
    Dummy class that is exactly like the Exception class.
    Used to deal with syntax issues config files.
    """
    pass


class DocIDError(Exception):
    pass


###############################################################################
# Functions for printing objects
###############################################################################


def printdict(d, max_print_len=None):
    s = ''
    for key, value in d.iteritems():
        s += str(key) + ': ' + str(value) + '\n'
    if max_print_len:
        print s[:max_print_len]
    else:
        print s


def print_dicts(dicts, prepend_str=''):
    for key, value in dicts.iteritems():
        if isinstance(value, dict):
            print prepend_str + key
            next_prepend_str = prepend_str + '  '
            print_dicts(value, next_prepend_str)
        else:
            print "%s%s = %.5f" % (prepend_str, key, value)


###############################################################################
# Functions for converting various format files to .txt
###############################################################################
def file_to_txt(file_path, dst_dir):
    """
    Takes a file path and writes the file in txt format to dst_dir.
    If file is alread .txt, then simply copies the file.

    Notes
    -----
    Currently only support pdf, txt, doc and docx.

    """
    try:
        file_path = _filepath_clean_copy(file_path)
    except IOError:
        sys.stdout.write('unable to clean file_name %s \n'%file_path)
    file_name = os.path.split(file_path)[1]
    name, ext = os.path.splitext(file_name)
    ext = re.sub(r'\.', '', ext)
    try:
        out = eval('_%s_to_txt'%ext)(file_path, dst_dir) #calls one of the _to_txt()
        if out: sys.stdout.write('unable to process file %s'%file_path)
    except NameError:
        sys.stdout.write('file type %s not supported, skipping %s \n'%(ext,
            file_name))
        pass

def _filepath_clean_copy(file_path):
    """
    creates a copy of the file with chars which need to be escapes
    replaced with a '_';

    Returns
    -------
    file_name : str
        clean file name

    """
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    if re.search(r'[,\s|:\'\.]', name):
        clean_name = re.sub(r'[,\s|:\'\.]', '_', name)
        clean_file_name = clean_name + ext
        clean_file_path = os.path.join(dir_name, clean_file_name)
        shutil.copyfile(file_path, clean_file_path)
    else:
        clean_file_path = file_path
    return clean_file_path

def _txt_to_txt(file_path, dst_dir):
    """
    Simply copies the file to the target dir.
    """
    file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, file_name)
    return subprocess.call(['cp', file_path, file_dst])


def _pdf_to_txt(file_path, dst_dir):
    """
    Uses the pdftotxt unix util, with --layout option, to convert file_name
    to .txt and save in dst_dir

    Notes
    -----
    Download and install Xpdf from http://www.foolabs.com/xpdf/download.html
    Follow the instruciton in INSTALL - should work on most *nix systems.
    """
    file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.pdf$', '.txt', file_name))
    with open(file_dst, 'w') as f:
        return subprocess.call(["pdftotext",  "-layout", file_path], stdout=f)


def _doc_to_txt(file_path, dst_dir):
    """
    Uses catdoc unix util to convert file_name
    to .txt and save in dst_dir.

    Notes
    -----
    To install catdoc:
        apt-get catdoc on unix/linux
        brew install on mac
    """
    file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.doc$', '.txt', file_name))
    with open(file_dst, 'w') as f:
        return subprocess.call(["catdoc",  "-w", file_path], stdout=f)


def _docx_to_txt(file_path, dst_dir):
    """
    Uses the docx python module to extract text from a docx file and save
    to .txt in dst_dir.
    """
    file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.docx$', '.txt', file_name))
    doc = opendocx(file_path)
    txt = '\n'.join(getdocumenttext(doc))
    txt = unidecode(txt)
    with open(file_dst, 'w') as f:
        f.write(txt)
    return 0



###############################################################################
# String type operations
###############################################################################


###############################################################################
# Misc.
###############################################################################

def grouper(iterable, chunksize, fillvalue=None):
    """
    Group iterable into chunks of length n, with fillvalue for the (possibly)
    smaller last chunk.

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    Parameters
    ----------
    iterable : Iterable
    chunksie : Integer
    fillvalue : Anything
        Fill missing values with this
    """
    args = [iter(iterable)] * chunksize

    return itertools.izip_longest(fillvalue=fillvalue, *args)
