import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from time import time
from gensim import corpora, models
from rosetta.text import streamers, gensim_helpers
from rosetta import common


class Topics(object):
    """
    Convenience wrapper for for the gensim LDA module.
    See http://radimrehurek.com/gensim/ for more details.
    """
    def __init__(
        self, text_base_path=None, limit=None, file_type='*.txt',
        shuffle=True, tokenizer=None, tokenizer_func=None, verbose=False):
        """
        Parameters
        ----------
        text_base_path : string or None
            Base path to dir containing files.  Used as the default source
            for dictionaries and corpus if these are not specified.
        limit : int or None
            Limit files read in text_base_path to this many.
        file_type : string
            File types to filter by.
        shuffle : Boolean
            If True, shuffle paths in base_path
        tokenizer : Subclass of BaseTokenizer
            Should have a text_to_token_list method.  Try using MakeTokenizer
            to convert a function to a valid tokenizer.
        tokenizer_func : function
            Takes in string (text) and returns list of strings.
        verbose : bool
        """
        self.verbose = verbose

        assert tokenizer or tokenizer_func, (
            'you must specify either tokenizer or tokenizer_func')

        if text_base_path:
            self.streamer = streamers.TextFileStreamer(
                text_base_path=text_base_path, file_type=file_type,
                tokenizer=tokenizer, tokenizer_func=tokenizer_func,
                limit=limit, shuffle=shuffle)

    def set_dictionary(
        self, doc_id=None, load_path=None, no_below=5, no_above=0.5,
        save_path=None):
        """
        Convert token stream into a dictionary, setting self.dictionary.

        Parameters
        ----------
        doc_id : List of doc_id
            Only use documents with these ids to build the dictionary
        load_path : string
            path to saved dictionary
        no_below : Integer
            Do not keep words with total count below no_below
        no_above : Real number in [0, 1]
            Do not keep words whose total count is more than no_above fraction
            of the total word count.
        save_path : string
            path to save dictionary
        """
        t0 = time()
        # Either load a pre-made dict, or create a new one using __init__
        # parameters.
        if load_path:
            dictionary = corpora.Dictionary.load(load_path)
        else:
            token_stream = self.streamer.token_stream(doc_id=doc_id)
            dictionary = corpora.Dictionary(token_stream)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        dictionary.compactify()

        if not load_path:
            build_time = (time() - t0) / 3600.
            self._print('Dictionary built in %.2f hours' % build_time)

        if save_path:
            dictionary.save(save_path)

        self.dictionary = dictionary

    def set_corpus(self, load_path=None, serialize_path=None, doc_id=None):
        """
        Creates a corpus and sets self.corpus

        Parameters
        ----------
        load_path : String
            Load an SvmLightPlusCorpus from here.
        serialize_path : String
            Create an SvmLightPlusCorpus using self.streamer and
            self.dictionary, then save it here.
        doc_id : List of strings
            Limit corpus building to documents with these ids
        """
        t0 = time()
        # Enforce one and only one of load_path, serialize_path
        load_nosave = (load_path is not None) and (serialize_path is None)
        noload_save = (load_path is None) and (serialize_path is not None)
        assert load_nosave or noload_save, (
            "Provide one and only one of load_path, serialize_path")

        # If you're loading, set streamer.doc_id_cache now
        # If you're streaming, self.streamer.doc_id_cache will be set when you
        # actually stream.
        if load_path:
            assert doc_id is None, "Can't filter by doc_id with loaded corpus"
            self.corpus = gensim_helpers.SvmLightPlusCorpus(
                load_path, doc_id=doc_id)
        else:
            self.corpus = gensim_helpers.SvmLightPlusCorpus.from_streamer_dict(
                self.streamer, self.dictionary, serialize_path, doc_id=doc_id)
            build_time = (time() - t0) / 3600.
            self._print('Corpus built %.2f hours' % build_time)

    def fit_lda(
        self, num_topics, alpha=None, eta=None, passes=1, chunksize=2000,
        update_every=1):
        """
        Buld the lda model on the current version of self.corpus.

        Parameters
        ----------
        num_topics : int
            number of topics
        alpha : list of floats, None
            hyperparameter vector for topic distribution
        eta : list of floats, None
            hyperparameter vector for word distribution
        passes : int
            number of passes for model build
        chunksize : int
        update_every ; int
        """
        self.num_topics = num_topics
        t0 = time()

        lda = models.LdaModel(self.corpus, id2word=self.dictionary,
            num_topics=num_topics, passes=passes, alpha=alpha, eta=eta,
            chunksize=chunksize, update_every=update_every)

        build_time = (time() - t0) / 3600.
        self._print('LDA built in %.2f hours' % build_time)
        self.lda = lda

        return lda

    def write_topics(self, num_words=5, outfile=sys.stdout):
        """
        Writes the topics to outfile.

        Parameters
        ----------
        outfile : filepath or buffer
            Designates file to write to.
        num_words : int
            number of words to write with each topic
        """
        with common.smart_open(outfile, 'w') as f:
            for t in xrange(self.num_topics):
                f.write('topic %s' % t + '\n')
                f.write(self.lda.print_topic(t, topn=num_words) + '\n')

    def write_doc_topics(self, save_path, sep='|'):
        """
        Creates a delimited file with doc_id and topics scores.
        """
        topics_df = self._get_topics_df()
        # Make sure the topic values sum to within atol of 1.0
        self._qa_topics(topics_df)
        topics_df.to_csv(save_path, sep=sep, header=True)

    def _qa_topics(self, topics_df):
        topic_sums = topics_df.sum(axis=1).values
        passed = np.fabs((topic_sums - 1)).max() < 0.1
        msg = '=' * 79 + '\n'
        msg += "Topics QA test passed:  %s\n" % passed
        print(msg)

    def _get_topics_df(self):
        topics_df = pd.concat(
            (pd.Series(dict(doc)) for doc in self.lda[self.corpus]), axis=1
            ).fillna(0).T
        topics_df.index = self.corpus.doc_id
        topics_df.index.name = 'doc_id'
        topics_df = topics_df.rename(
            columns={i: 'topic_' + str(i) for i in topics_df.columns})

        return topics_df

    def _print(self, msg):
        if self.verbose:
            sys.stdout.write(msg + '\n')

    def get_words_docfreq(self, plot_path=None):
        """
        Returns a df with token id, doc freq as columns and words as index.
        """
        id2token = dict(self.dictionary.items())
        words_df = pd.DataFrame(
                {id2token[tokenid]: [tokenid, docfreq]
                 for tokenid, docfreq in self.dictionary.dfs.iteritems()},
                index=['tokenid', 'docfreq']).T
        words_df = words_df.sort_index(by='docfreq', ascending=False)
        if plot_path:
            plt.figure()
            words_df.docfreq.apply(np.log10).hist(bins=200)
            plt.xlabel('log10(docfreq)')
            plt.ylabel('Count')
            plt.savefig(plot_path)

        return words_df
