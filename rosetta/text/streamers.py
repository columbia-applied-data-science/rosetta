"""
Classes for streaming tokens/info from files/sparse files etc...
"""
from collections import Counter
from random import shuffle
import re
from functools import partial
import abc
import sys
import os
from scipy import sparse

from rosetta.parallel.parallel_easy import imap_easy, parallel_apply

from .. import common
from ..common import lazyprop, smart_open, DocIDError
from . import filefilter, text_processors


class BaseStreamer(object):
    """
    Base class...don't use this directly.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def info_stream(self, **kwargs):
        """
        Abstract method.  All derived classes will implement to
        return an interator over the text documents with processing
        as appropriate.
        """
        return

    @abc.abstractmethod
    def record_stream(self, **kwargs):
        return

    def single_stream(self, item, cache_list=None, **kwargs):
        """
        Stream a single item from source.

        Parameters
        ----------
        item : String
            The single item to pull from info and stream.
        cache_list : List of strings
            Cache these items on every iteration
        kwargs : Keyword args
            Passed on to self.info_stream
        """
        cache_list = [] if cache_list is None else cache_list

        # Initialize the cached items as attributes
        for cache_item in cache_list:
            self.__dict__[cache_item + '_cache'] = []

        # Iterate through self.info_stream and pull off required information.
        if kwargs:
            stream = self.info_stream(**kwargs)
        else:
            stream = self.info_stream()
        for i, info in enumerate(stream):
            for cache_item in cache_list:
                self.__dict__[cache_item + '_cache'].append(info[cache_item])

            yield info[item]

    def token_stream(self, cache_list=None, **kwargs):
        """
        Returns an iterator over tokens with possible caching of other info.

        Parameters
        ----------
        cache_list : List of strings.
            Cache these items as they appear
            E.g. self.token_stream('doc_id', 'tokens') caches
            info['doc_id'] and info['tokens'] (assuming both are available).
        kwargs : Keyword args
            Passed on to self.info_stream
        """
        return self.single_stream('tokens', cache_list=cache_list, **kwargs)

    def to_vw(self, out_stream=sys.stdout, n_jobs=1, raise_on_bad_id=True,
              cache_list=None):
        """
        Write our filestream to a VW (Vowpal Wabbit) formatted file.

        Parameters
        ----------
        out_stream :  stream, buffer or open file object
        n_jobs : Integer
            number of CPU jobs
        cache_list : List of strings.
            Cache these items as they appear
            E.g. self.token_stream('doc_id', 'tokens') caches
            info['doc_id'] and info['tokens'] (assuming both are available).
        Notes:
        -----
        self.info_stream must have a 'doc_id' field, as this is used to index
        the lines in the vw file.
        """
        assert self.tokenizer, 'tokenizer must be defined to use .to_vw()'
        cache_list = [] if cache_list is None else cache_list
        # Initialize the cached items as attributes
        for cache_item in cache_list:
            self.__dict__[cache_item + '_cache'] = []
        formatter = text_processors.VWFormatter()
        func = partial(_to_sstr, tokenizer=self.tokenizer, formatter=formatter,
                       raise_on_bad_id=raise_on_bad_id, streamer=self,
                       cache_list=cache_list)
        parallel_apply(func, self.record_stream(), n_jobs,
                       sep='\n', out_stream=out_stream)

    def to_scipysparse(self, cache_list=None, **kwargs):
        """
        Returns a scipy sparse matrix representing the collection of documents
        as a bag of words, one (sparse) row per document.
        Translation from column to token are stored into the cache

        Parameters
        ----------
        cache_list : List of strings.
            Cache these items as they appear
            E.g. self.token_stream('doc_id', 'tokens') caches
            info['doc_id'] and info['tokens'] (assuming both are available).
        kwargs : Keyword args
            Passed on to self.info_stream
        """
        token_col = dict()
        n_col = -1
        row_ind = list()
        col_ind = list()
        data = list()

        # iterate through each document's tokens list and build up the
        # sparse matrix row by row.
        for row, tokens in enumerate(self.token_stream(cache_list, **kwargs)):
            doc_counts = Counter(tokens)
            for token, count in doc_counts.items():
                if token not in token_col:
                    n_col += 1
                    token_col[token] = n_col
                row_ind.append(row)
                col_ind.append(token_col[token])
                data.append(count)

        self.__dict__['token_col_map'] = token_col
        return sparse.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(row + 1, n_col + 1))


class VWStreamer(BaseStreamer):
    """
    For streaming from a single VW file.  Since the VW file format does not
    preserve token order, all tokens are unordered.
    """
    def __init__(self, sfile=None, cache_sfile=False,
                 limit=None, shuffle=False):
        """
        Parameters
        ----------
        sfile : File path or buffer
            Points to a sparse (VW) formatted file.
        cache_sfile : Boolean
            If True, cache the sfile in memory.  CAREFUL!!!
        limit : Integer
            Only return this many results
        shuffle : Boolean
            If True, shuffle paths once (and only once) before streaming
        """
        self.sfile = sfile
        self.cache_sfile = cache_sfile
        self.limit = limit
        self.shuffle = shuffle

        self.formatter = text_processors.VWFormatter()

        if cache_sfile:
            self.source = self._cached_stream
            self._init_cached_stream()
        else:
            assert not shuffle, "Can only shuffle a cached stream"
            self.source = self._sfile_stream

    def _init_cached_stream(self):
        records = {}
        for record_dict in self._sfile_stream():
            doc_id = record_dict['doc_id']
            records[doc_id] = record_dict

        # Set self.records and self.doc_id
        self.records = records
        doc_id = records.keys()
        if self.shuffle:
            shuffle(doc_id)
        self.doc_id = doc_id

    def _cached_stream(self, doc_id=None):
        records = self.records

        if doc_id is None:
            for i, doc in enumerate(self.doc_id):
                record_dict = self.records[doc]
                if i == self.limit:
                    raise StopIteration
                yield record_dict
        else:
            if (self.limit is not None) and self.cache_sfile:
                raise ValueError(
                    "Cannot use both self.limit and doc_id with cached stream")
            for doc in doc_id:
                yield records[doc]

    def _sfile_stream(self, doc_id=None):
        """
        Stream record_dict from an sfile that sits on disk.
        """
        # Open file if path.  If buffer or StringIO, passthrough.
        with smart_open(self.sfile, 'rb') as infile:
            if doc_id is not None:
                doc_id = set(doc_id)

            for i, line in enumerate(infile):
                if i == self.limit:
                    raise StopIteration

                record_dict = self.formatter.sstr_to_dict(line)
                if doc_id is not None:
                    if record_dict['doc_id'] not in doc_id:
                        continue
                yield record_dict

    def record_stream(self, doc_id=None):
        """
        Returns an iterator over record dicts.

        Parameters
        ----------
        doc_id : Iterable over Strings
            Return record dicts iff doc_id in doc_id
        """
        source = self.source(doc_id=doc_id)

        for record_dict in source:
            yield record_dict

    def info_stream(self, doc_id=None):
        """
        Returns an iterator over info dicts.

        Parameters
        ----------
        doc_id : Iterable over Strings
            Return info dicts iff doc_id in doc_id
        """

        # Read record_dict and convert to info by adding tokens
        for record_dict in self.record_stream(doc_id):
            record_dict['tokens'] = self.formatter._dict_to_tokens(record_dict)

            yield record_dict


class TextFileStreamer(BaseStreamer):
    """
    For streaming from text files.
    """
    def __init__(
            self, text_base_path=None, path_list=None, file_type='*',
            name_strip=r'\..*', tokenizer=None, tokenizer_func=None, limit=None,
            shuffle=True):
        """
        Parameters
        ----------
        text_base_path : string or None
            Base path to dir containing files.
        path_list : list or None
            explicit list of paths to be used
        file_type : String
            String to filter files with.  E.g. '*.txt'.
            Note that the filenames will be converted to lowercase before
            this comparison.
        name_strip : raw string
            Regex to strip doc_id.
        tokenizer : Subclass of BaseTokenizer
            Should have a text_to_token_list method.  Try using MakeTokenizer
            to convert a function to a valid tokenizer.
        tokenizer_func : Function
            Transforms a string (representing one file) to a list of strings
            (the 'tokens').
        limit : int or None
            Limit for number of docs processed.
        shuffle : Boolean
            If True, shuffle paths once (and only once) before streaming
        """
        self.text_base_path = text_base_path
        self.path_list = path_list
        self.file_type = file_type
        self.name_strip = name_strip
        self.limit = limit
        self.tokenizer = tokenizer
        self.tokenizer_func = tokenizer_func
        self.shuffle = shuffle
        assert (text_base_path is None) or (path_list is None)
        assert (tokenizer is None) or (tokenizer_func is None)
        if tokenizer_func:
            self.tokenizer = text_processors.MakeTokenizer(tokenizer_func)

    @lazyprop
    def paths(self):
        """
        Get all paths that we will use.
        """
        if self.text_base_path:
            paths = filefilter.get_paths(
                self.text_base_path, file_type=self.file_type)
            if self.shuffle:
                shuffle(paths)
            if self.limit:
                paths = paths[: self.limit]
        else:
            paths = self.path_list

        return paths

    @lazyprop
    def doc_id(self):
        """
        Get doc_id corresponding to all paths.
        """
        regex = re.compile(self.name_strip)
        doc_id = [
            regex.sub('', filefilter.path_to_name(p, strip_ext=False))
            for p in self.paths]

        return doc_id

    @lazyprop
    def _doc_id_to_path(self):
        """
        Build the dictionary mapping doc_id to path.  doc_id is based on
        the filename.
        """
        return dict(zip(self.doc_id, self.paths))

    @lazyprop
    def file_stat(self):
        """
        Builds a dictionary of os.stats file info. Currently included
        last modification time & last access time (seconds since epoch),
        size (in bytes).
        """
        return [self._file_stat(p) for p in self.paths]

    def _file_stat(self, path):
        """
        Retrieves os.stats info for file.
        """
        return {'mtime': os.path.getmtime(path),
                'atime': os.path.getatime(path),
                'size': os.path.getsize(path)}

    def record_stream(self, paths=None, doc_id=None, limit=None):
        return self.info_stream(paths, doc_id, limit)

    def info_stream(self, paths=None, doc_id=None, limit=None):
        """
        Returns an iterator over paths yielding dictionaries with information
        about the file contained within.

        Parameters
        ----------
        paths : list of strings
        doc_id : list of strings or ints
        limit : Integer
            Use limit in place of self.limit.
        """
        if limit is None:
            limit = self.limit

        if doc_id is not None:
            paths = [self._doc_id_to_path[str(doc)] for doc in doc_id]
        elif paths is None:
            paths = self.paths

        for index, onepath in enumerate(paths):
            if index == limit:
                raise StopIteration

            with open(onepath, 'r') as f:
                text = f.read()
                doc_id = re.sub(
                    self.name_strip, '', filefilter.path_to_name
                    (onepath, strip_ext=False))
                stat_dict = self._file_stat(onepath)
                info_dict = {'text': text, 'cached_path': onepath,
                             'doc_id': doc_id}
                info_dict.update(stat_dict)
                if self.tokenizer:
                    info_dict['tokens'] = (
                        self.tokenizer.text_to_token_list(text))

            yield info_dict

    def to_vw(self, outfile, n_jobs=1, chunksize=1000, raise_on_bad_id=True):
        """
        Write our filestream to a VW (Vowpal Wabbit) formatted file.

        Parameters
        ----------
        outfile : filepath or buffer
        n_jobs : Integer
            Use n_jobs different jobs to do the processing.  Set = 4 for 4
            jobs.  Set = -1 to use all available, -2 for all except 1,...
        chunksize : Integer
            Workers process this many jobs at once before pickling and sending
            results to master.  If this is too low, communication overhead
            will dominate.  If this is too high, jobs will not be distributed
            evenly.
        raise_on_bad_id : Boolean
            If True, raise DocIDError when the doc_id (formed by self) is not
            a valid VW "Tag".  I.e. contains :, |, ', or whitespace.
            If False, print warning.
        """
        # Note:  This is similar to rosetta/cmd/files_to_vw.py
        # This implementation is more complicated, due to the fact that a
        # streamer specifies the method to extract doc_id from a stream.
        # To be faithful to the streamer, we must therefore use the streamer
        # to stream the files.  This requires a combination of imap_easy and
        # a chunker.
        #
        # Create an iterator over chunks of paths
        path_group_iter = common.grouper(self.paths, chunksize)

        formatter = text_processors.VWFormatter()

        func = partial(_group_to_sstr, self, formatter, raise_on_bad_id)
        # Process one group at a time...set imap_easy chunksize arg to 1
        # since each group contains many paths.
        results_iterator = imap_easy(func, path_group_iter, n_jobs, 1)

        with smart_open(outfile, 'w') as open_outfile:
            for group_results in results_iterator:
                for sstr in group_results:
                    open_outfile.write(sstr + '\n')


class TextIterStreamer(BaseStreamer):
    """
    For streaming text.
    """
    def __init__(self, text_iter, tokenizer=None, tokenizer_func=None):
        """
        Parameters
        ----------
        text_iter : iterator or iterable of dictionaries
            Each dict must contain key:value "text": text_string,
            but can contain other metadata key:values for cacheing
            (ex/ see self.token_stream).
        tokenizer : Subclass of BaseTokenizer
            Should have a text_to_token_list method.  Try using MakeTokenizer
            to convert a function to a valid tokenizer.
        tokenizer_func : Function
            Transforms a string (representing one file) to a list of strings
            (the 'tokens').
        """
        self.text_iter = text_iter
        self.tokenizer = tokenizer
        self.tokenizer_func = tokenizer_func

        assert (tokenizer is None) or (tokenizer_func is None)
        if tokenizer_func:
            self.tokenizer = text_processors.MakeTokenizer(tokenizer_func)

    def record_stream(self):
        """
        Yields a dict from self.streamer
        """
        for info in self.text_iter:
            yield info

    def info_stream(self):
        """
        Yields a dict from self.streamer as well as "tokens".
        """
        for info in self.record_stream():
            info['tokens'] = self.tokenizer.text_to_token_list(info['text'])
            yield info


def _group_to_sstr(streamer, formatter, raise_on_bad_id, path_group):
    """
    Return a list of sstr's (sparse string representations).  One for every
    path in path_group.
    """
    # grouper might append None to the last group if this one is shorter
    path_group = (p for p in path_group if p is not None)

    group_results = []

    info_stream = streamer.info_stream(paths=path_group)
    for info_dict in info_stream:
        doc_id = info_dict['doc_id']
        tokens = info_dict['tokens']
        feature_values = Counter(tokens)
        try:
            tok_sstr = formatter.get_sstr(
                feature_values, importance=1, doc_id=doc_id)
        except DocIDError as e:
            msg = e.message + "\npath = %s\n" % info_dict['cached_path']
            if raise_on_bad_id:
                raise DocIDError(msg)
            else:
                msg = "WARNING: " + msg
                sys.stderr.write(msg)

        group_results.append(tok_sstr)

    return group_results


def _to_sstr(record_dict, tokenizer, formatter, raise_on_bad_id,
             streamer, cache_list=None):
    """
    Yield a list of sstr's (sparse string representations); takes yield
    instances of STREAMER.record_stream(), tokenizes the text, get 'doc_id'
    and returns sstr.
    """
    cache_list = [] if cache_list is None else cache_list

    doc_id = str(record_dict['doc_id'])
    tokens = tokenizer.text_to_token_list(record_dict['text'])
    feature_values = Counter(tokens)
    try:
        tok_sstr = formatter.get_sstr(
            feature_values, importance=1, doc_id=doc_id)
        for cache_item in cache_list:
            streamer.__dict__[cache_item + '_cache'].append(
                record_dict[cache_item])
    except DocIDError as e:
        msg = e.message + "\doc_id = %s\n" % record_dict['doc_id']
        if raise_on_bad_id:
            raise DocIDError(msg)
        else:
            msg = "WARNING: " + msg
            sys.stderr.write(msg)
    return tok_sstr
