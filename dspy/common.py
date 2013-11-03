"""
Common functions/classes for dataprep.
"""
import numpy as np
import sys
import cPickle
from StringIO import StringIO
import itertools


################################################################################
# Decorators
################################################################################


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


################################################################################
# Wrappers for opening/closing files
################################################################################


def get_inout_files(infilename, outfilename, inmode='rb', outmode='wb'):
    """
    Gets infile, and outfile, which are opened versions of infilename,
    outfilename.

    Parameters
    ----------
    infilename : String
        Name of file to read.  If None, we will read from stdin
    outfilename : String
        Name of file to write.  If None, we will write to stdout
    outmode : String
        Mode to open file in

    Returns
    -------
    The tuple (infile, outfile)

    Examples
    --------
    >>> infile, outfile = get_inout_files(infilename, outfilename)
    >>> myfunction(infile, outfile,...)
    >>> close_files(infile, outfile)
    """
    infile = get_infile(infilename, inmode=inmode)
    outfile = get_outfile(outfilename, outmode=outmode)

    return infile, outfile


def close_files(infile, outfile):
    """
    Closes the files if and only if they are not equal to sys.stdin, sys.stdout

    Parameters
    ----------
    infile : Open file buffer
    outfile : Open file buffer

    Examples
    --------
    >>> infile, outfile = get_inout_files(infilename, outfilename)
    >>> myfunction(infile, outfile,...)
    >>> close_files(infile, outfile)
    """
    close_infile(infile)
    close_outfile(outfile)


def close_infile(infile):
    """
    Closes infile if and only if it is not equal to sys.stdin.  Use with get_infile.
    
    Parameters
    ----------
    infile : Open file buffer

    Examples
    --------
    >>> infile = get_infile(infilename)
    >>> myfunction(infile,...)
    >>> close_infile(infile)
    """
    if infile != sys.stdin:
        infile.close()


def close_outfile(outfile):
    """
    Closes outfile if and only if it is not equal to sys.stdout.  Use with get_outfile.

    Examples
    --------
    >>> outfile = get_infile(outfilename)
    >>> myfunction(outfile,...)
    >>> close_outfile(outfile)
    """
    if outfile != sys.stdout:
        outfile.close()


def get_infile(infilename, inmode='rb'):
    """
    Gets infile, which is an opened version of infilename.

    Parameters
    ----------
    infilename : String
        Name of file to read.  If None, we will read from stdin

    Returns
    -------
    infile

    Examples
    --------
    >>> infile = get_infile(infilename)
    >>> myfunction(infile,...)
    >>> close_infile(infile)
    """
    if infilename:
        infile = open(infilename, inmode)
    else:
        infile = sys.stdin

    return infile


def get_outfile(outfilename, outmode='wb', default=sys.stdout):
    """
    Open outfilename in outmode.

    Parameters
    ----------
    outfilename : String
        Name of file to open and return.
        If None, return the kwarg 'default'
    outmode : String
        Mode to open file in
    default : File buffer
       The value to return if outfilename is None 

    Returns
    -------
    outfile

    Examples
    --------
    >>> outfile = get_outfile(outfilename)
    >>> myfunction(outfile,...)
    >>> close_outfile(outfile)
    """
    if isinstance(outfilename, str):
        outfile = open(outfilename, outmode)
    elif outfilename is None:
        outfile = default
    else:
        raise ValueError(
            "Argument outfilename is of type %s. Not handled." % outfilename)

    return outfile


def openfile_wrap(filename, mode):
    """
    If filename is a string, returns an opened version of filename.
    If filename is a file buffer, then passthrough.

    Parameters
    ----------
    filename : String or file buffer
    mode : String
        mode to open the file in

    Returns
    -------
    opened_file : Opened file buffer
    was_path : Boolean
        If True, then filename was a string (and thus was opened here, and so
        you better remember to close it elsewhere)

    Examples
    --------
    >>> infile, was_path = openfile_wrap(infilename, 'r')
    >>> myfunction(infile,...)
    >>> if was_path:
    >>>     infile.close()
    """
    if isinstance(filename, str):
        was_path = True
        opened_file = open(filename, mode)
    elif isinstance(filename, file) or isinstance(filename, StringIO):
        was_path = False
        opened_file = filename
    else:
        raise Exception("Could not work with %s" % filename)

    return opened_file, was_path


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


################################################################################
# Functions to read special file formats
################################################################################

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
    f, was_path = openfile_wrap(infile, 'r')

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

    if was_path:
        f.close()

    return kpv_list


def write_list_to_filerows(outfile, mylist):
    """
    The inverse of get_list_from_filerows.

    Parameters
    ----------
    mylist : List
    """
    f, was_path = openfile_wrap(outfile, 'w')

    for item in mylist:
        f.write(str(item) + '\n')

    if was_path:
        f.close()


def pickleme(obj, filename, protocol=2):
    """
    Save obj to disk using cPickle.

    Parameters
    ----------
    obj : Serializable Python object
    filename : String
        Name of file to store obj to
    protocol : 0, 1, or 2
        2 is fastest
    """
    with open(filename, 'w') as f:
        cPickle.dump(obj, f, protocol=protocol)


def unpickleme(filename):
    """
    Returns unpickled version of object.

    Parameters
    ----------
    filename : String
        We will attempt to unpickle this file.
    """
    with open(filename, 'r') as f:
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
        tuple_list = [tuple(row) for row in loan_list if '' not in row]
    else:
        tuple_list = [tuple(row) for row in loan_list]

    return np.array(tuple_list, schema)


################################################################################
# Custom Exceptions
################################################################################


class BadDataError(Exception):
    """
    Dummy class that is exactly like the Exception class.  Used to make sure
    people are raising the intended exception, rather than some other wierd
    one.
    """
    pass


class ConfigurationSyntaxError(Exception):
    """
    Dummy class that is exactly like the Exception class. 
    Used to deal with syntax issues config files.
    """
    pass


################################################################################
# Functions for printing objects
################################################################################


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
            print "%s%s = %.5f"%(prepend_str, key, value)


################################################################################
# String type operations
################################################################################


################################################################################
# Misc.
################################################################################

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


