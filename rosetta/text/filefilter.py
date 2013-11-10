"""
Contains a collection of function that clean, decode and move files around.
"""
from fnmatch import fnmatch
import os
import re

from ..common import lazyprop


def get_paths(
    base_path, file_type="*", relative=False, get_iter=False, limit=None):
    """
    Crawls subdirectories and returns an iterator over paths to files that
    match the file_type.

    Parameters
    ----------
    base_path : String
        Path to the directory that will be crawled
    file_type : String
        String to filter files with.  E.g. '*.txt'.  Note that the filenames
        will be converted to lowercase before this comparison.
    relative : Boolean
        If True, get paths relative to base_path
        If False, get absolute paths
    get_iter : Boolean
        If True, return an iterator over paths rather than a list.
    """
    path_iter = _get_paths_iter(
        base_path, file_type=file_type, relative=relative, limit=limit)

    if get_iter:
        return path_iter
    else:
        return [path for path in path_iter]


def _get_paths_iter(base_path, file_type="*", relative=False, limit=None):
    counter = 0
    for path, subdirs, files in os.walk(base_path, followlinks=True):
        for name in files:
            if fnmatch(name.lower(), file_type):
                if relative:
                    path = path.replace(base_path, "")
                    if path.startswith('/'):
                        path = path[1:]
                if counter == limit:
                    raise StopIteration

                yield os.path.join(path, name)
                counter += 1


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
