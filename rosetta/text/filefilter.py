"""
Contains a collection of function that clean, decode and move files around.
"""
import os
import re
import subprocess

from fnmatch import fnmatch

from ..common import lazyprop


def get_paths(
    base_path, relative=False, get_iter=False, limit=None, filters=None):
    """
    Crawls subdirectories and returns an iterator over paths to files that
    match the file_type.

    Parameters
    ----------
    base_path : String
        Path to the directory that will be crawled
    relative : Boolean
        If True, get paths relative to base_path
        If False, get absolute paths
    get_iter : Boolean
        If True, return an iterator over paths rather than a list.
    filters : list of tuples
        List of (file_filter, **kwargs_dict); each function takes path dict of
        kwargs and return bool
    """
    path_iter = _get_paths_iter(
        base_path, relative=relative, limit=limit, filters=filters)

    if get_iter:
        return path_iter
    else:
        return [path for path in path_iter]


def _get_paths_iter(base_path, relative=False, limit=None, filters=None):
    counter = 0
    if filters is None:
        filters=[]
    for path, subdirs, files in os.walk(base_path, followlinks=True):
        for name in files:
            if all([f[0](path, **f[1]) for f in filters]):
                if relative:
                    path = path.replace(base_path, "")
                    if path.startswith('/'):
                        path = path[1:]
                if counter == limit:
                    raise StopIteration

                yield os.path.join(path, name)
                counter += 1

def _fpmatch_filter(path, pattern="*"):
    """
    Filepath matching with shell style patters. 

    Notes
    -----
    path is cast to lower before patter match.
    """
    return fnmatch(path.lower(), pattern)

def _wc_filter(path, option, count, max_min):
    """
    Filters files by stats from the wc unix utility. 

    Parameters
    ----------
    path : str
        path to file
    option : str
        wc option, one of 'c' 'l' 'm' or 'w'
    count : int
    max_min : str
        specifies upper or lower bound for count compared to wc output

    Returns
    -------
    bool

    Notes
    -----
    example: _wc_filter(path, 'l', 100, 'max') will return True for files 
    with at most 100 lines
    """
    assert option in ['c', 'l', 'm', 'w'], 'wc option must be one of c,l,m,w'
    assert max_min in ['max', 'min'], 'max min must be either "max" or "min"'
    wc_stat = subprocess.check_output(
            ['wc', '-%s'%wc['option'], path]).split()[0]
    wc_stat = int(wc_stat)
    if max_min=='min':
        return wc_stat>=wc['count']
    elif max_min=='max':
        return wc_stat<=wc['count']


def path_to_name(path, strip_ext=True):
    """
    Takes one path and returns the filename, excluding the extension.
    """
    head, name = os.path.split(path)
    if strip_ext:
        name, ext = os.path.splitext(name)

    return name


def path_to_newname(path, name_level=1):
    """
    Takes one path and returns a new name, combining the directory structure
    with the filename.

    Parameters
    ----------
    path : String
    name_level : Integer
        Form the name using items this far back in the path.  E.g. if
        path = mydata/1234/3.txt and name_level == 2, then name = 1234_3

    Returns
    -------
    name : String
    """
    name_plus_ext = path.split('/')[-name_level:]
    name, ext = os.path.splitext('_'.join(name_plus_ext))

    return name


class PathFinder(object):
    """
    Find and access paths in a directory tree.
    """
    def __init__(
        self, text_base_path=None, file_type='*', name_strip=r'\..*',
        limit=None):
        """
        Parameters
        ----------
        text_base_path : String
            Base path that will be crawled to find paths.
        file_type : String
            Glob expression filtering the file type
        name_strip : String (Regex)
            To convert filenames to doc_id, we strip this pattern
            Default pattern r'\..*' strips everything after the first period
        limit : Integer
            Limit the paths returned to this number
        """
        self.text_base_path = text_base_path
        self.file_type = file_type
        self.name_strip = name_strip
        self.limit = limit

    @lazyprop
    def paths(self):
        """
        Get all paths that we will use.
        """
        if self.text_base_path:
            paths = get_paths(
                self.text_base_path, self.file_type, limit=self.limit)
        else:
            paths = None

        return paths

    @lazyprop
    def doc_id(self):
        """
        Get doc_id corresponding to all paths.
        """
        regex = re.compile(self.name_strip)
        doc_id = [
            regex.sub('', path_to_name(p, strip_ext=False))
            for p in self.paths]

        return doc_id

    @lazyprop
    def _doc_id_to_path(self):
        """
        Build the dictionary mapping doc_id to path.  doc_id is based on
        the filename.
        """
        return dict(zip(self.doc_id, self.paths))

    def __getitem__(self, identifiers):
        """
        self[identifiers] returns a list of paths corresponding to identifiers.
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        return [self._doc_id_to_path[str(doc_id)] for doc_id in identifiers]
