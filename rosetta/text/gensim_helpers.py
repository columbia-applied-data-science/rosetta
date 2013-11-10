"""
Helper objects/functions specifically for use with Gensim.
"""
import pandas as pd
from gensim import corpora

from .. import common


class StreamerCorpus(object):
    """
    A "corpus type" object built with token streams and dictionaries.

    Depending on your method for streaming tokens, this could be slow...
    Before modeling, it's usually better to serialize this corpus using:

    self.to_corpus_plus(fname)
    or
    gensim.corpora.SvmLightCorpus.serialize(path, self)
    """
    def __init__(self, streamer, dictionary, doc_id=None, limit=None):
        """
        Stream token lists from pre-defined path lists.

        Parameters
        ----------
        streamer : Streamer compatible object.
            Method streamer.token_stream() returns a stream of lists of words.
        dictionary : gensim.corpora.Dictionary object
        doc_id : Iterable over strings
            Limit all streaming results to docs with these doc_ids
        limit : Integer
            Limit all streaming results to this many
        """
        self.streamer = streamer
        self.dictionary = dictionary
        self.doc_id = doc_id
        self.limit = limit

    def __iter__(self):
        """
        Returns an iterator of "corpus type" over text files.
        """
        token_stream = self.streamer.token_stream(
            doc_id=self.doc_id, limit=self.limit, cache_list=['doc_id'])

        for token_list in token_stream:
            yield self.dictionary.doc2bow(token_list)

    def serialize(self, fname):
        """
        Save to svmlight (plus) format, generating files:
        fname, fname.index, fname.doc_id
        """
        # Make the corpus and .index file
        corpora.SvmLightCorpus.serialize(fname, self)

        # Make the .doc_id file
        # Streamer cached the doc_id while streaming
        with open(fname + '.doc_id', 'w') as f:
            f.write('\n'.join(self.streamer.doc_id_cache))


class SvmLightPlusCorpus(corpora.SvmLightCorpus):
    """
    Extends gensim.corpora.SvmLightCorpus, providing methods to work with
    (e.g. filter by) doc_ids.
    """
    def __init__(self, fname, doc_id=None, doc_id_filter=None, limit=None):
        """
        Parameters
        ----------
        fname : Path
            Contains the .svmlight bag-of-words text file
        doc_id : Iterable
            Stream these doc_ids exactly, in the order given.
        doc_id_filter : Iterable
            Stream doc_ids in intersection of fname.doc_id and doc_id_filter
        limit : Integer
            Equivalent to initializing with the first limit rows of fname and
            fname.doc_id.
        """
        corpora.SvmLightCorpus.__init__(self, fname)

        self.limit = limit

        # All possible doc_id in the corpus
        self.doc_id_all = common.get_list_from_filerows(fname + '.doc_id')
        self.doc_id_all = self.doc_id_all[: limit]
        self.doc_id_all_set = set(self.doc_id_all)

        # Set self.doc_id
        if doc_id_filter is not None:
            assert doc_id is None, "Can't pass both doc_id and doc_id_filter"
            self.doc_id = [
                id for id in doc_id_filter if str(id) in self.doc_id_all_set]
        elif doc_id is not None:
            self.doc_id = doc_id
        else:
            self.doc_id = self.doc_id_all

    @property
    def doc_id(self):
        return self._doc_id

    @doc_id.setter
    def doc_id(self, iterable):
        # Called whenever you set self.doc_id = something
        self._doc_id = [str(id) for id in iterable]
        self.doc_id_set = set(self._doc_id)
        if not self.doc_id_set.issubset(self.doc_id_all_set):
            raise ValueError(
                "Attempt to set self.doc_id to values not contained in the"
                " corpus .doc_id file")

    def __iter__(self):
        """
        Returns a gensim-compatible corpus.

        Parameters
        ----------
        doc_id : Iterable over Strings
            Return info dicts iff doc_id in doc_id
        """
        base_iterable = corpora.SvmLightCorpus.__iter__(self)
        for i, row in enumerate(base_iterable):
            if i == self.limit:
                raise StopIteration

            if self.doc_id_all[i] in self.doc_id_set:
                yield row

    def serialize(self, fname, **kwargs):
        """
        Save to svmlight (plus) format, generating files:
        fname, fname.index, fname.doc_id

        Parameters
        ----------
        fname : String
            Path to save the bag-of-words file at
        kwargs : Additional keyword arguments
            Passed to SvmLightCorpus.serialize
        """
        # Make the corpus and .index file
        corpora.SvmLightCorpus.serialize(fname, self, **kwargs)

        # Make the .doc_id file
        with open(fname + '.doc_id', 'w') as f:
            f.write('\n'.join(self.streamer.doc_id))

    @classmethod
    def from_streamer_dict(
        self, streamer, dictionary, fname, doc_id=None, limit=None):
        """
        Initialize from a Streamer and gensim.corpora.dictionary, serializing
        the corpus (to disk) in SvmLightPlus format, then returning a
        SvmLightPlusCorpus.

        Parameters
        ----------
        streamer : Streamer compatible object.
            Method streamer.token_stream() returns a stream of lists of words.
        dictionary : gensim.corpora.Dictionary object
        fname : String
            Path to save the bag-of-words file at
        doc_id : Iterable over strings
            Limit all streaming results to docs with these doc_ids
        limit : Integer
            Limit all streaming results to this many

        Returns
        -------
        corpus : SvmLightCorpus
        """
        streamer_corpus = StreamerCorpus(
            streamer, dictionary, doc_id=doc_id, limit=limit)
        streamer_corpus.serialize(fname)

        return SvmLightPlusCorpus(fname, doc_id=doc_id, limit=limit)


def get_words_docfreq(dictionary):
    """
    Returns a df with token id, doc freq as columns and words as index.
    """
    id2token = dict(dictionary.items())
    words_df = pd.DataFrame(
        {id2token[tokenid]: [tokenid, docfreq]
         for tokenid, docfreq in dictionary.dfs.iteritems()},
        index=['tokenid', 'docfreq']).T
    words_df = words_df.sort_index(by='docfreq', ascending=False)

    return words_df


def get_topics_df(corpus, lda):
    """
    Creates a delimited file with doc_id and topics scores.
    """
    topics_df = pd.concat(
        (pd.Series(dict(doc)) for doc in lda[corpus]), axis=1).fillna(0).T
    topics_df = topics_df.rename(
        columns={i: 'topic_' + str(i) for i in topics_df.columns})

    if hasattr(corpus, 'doc_id'):
        topics_df.index = corpus.doc_id
        topics_df.index.name = 'doc_id'

    return topics_df
