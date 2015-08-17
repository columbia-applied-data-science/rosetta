"""
Common functions/classes for dataprep.
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
try:
    import cPickle
except ImportError:
    import pickle as cPickle
try:
    from itertools import izip_longest
except ImportError:
    from itertools import zip_longest as izip_longest
import functools

from collections import defaultdict


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
        print(s[:max_print_len])
    else:
        print(s)


###############################################################################
# Custom data structures
###############################################################################

def nested_defaultdict(default_factory, levels=1):
    """
    Creates nested defaultdicts with the lowest level having default factory.

    Parameters
    ----------
    default_factory : Callable
        Called without arguments to produce a new value when a key is not
        present.
    levels : Positive Integer
        The number of nesting levels to use.  If levels == 1, this is just
        an ordinary defaultdict.

    Examples
    --------
    >>> mydict = nested_defaultdict(int, levels=2)
    >>> mydict['columbia']['undergrads'] += 1
    """
    if not isinstance(levels, int) or (levels < 1):
        raise ValueError("levels =%s, should be a postitive integer" % levels)

    def nestone():
        """Used in place of a lambda to allow pickling"""
        return nested_defaultdict(default_factory, levels - 1)

    if levels == 1:
        return defaultdict(default_factory)
    else:
        return defaultdict(nestone)


def nested_keysearch(ndict, key_list):
    """
    Returns True if ndict[key_list[0]][key_list[1]]...[key_list[-1]] exists.

    Parameters
    ----------
    ndict : Nested dictionary
        E.g. {'a': {'b': 2}}
    key_list : List of strings
    """
    if isinstance(key_list, basestring):
        key_list = [key_list]

    first_key = key_list[0]

    if len(key_list) == 1:
        return first_key in ndict
    else:
        if first_key in ndict:
            return nested_keysearch(ndict[first_key], key_list[1: ])


###############################################################################
# String type operations
###############################################################################


###############################################################################
# Functional
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

    return izip_longest(fillvalue=fillvalue, *args)


def compose(*functions):
    def compose2(f, g):
        def f_g(x):
            return f(g(x))
        return f_g
    return functools.reduce(compose2, functions)
