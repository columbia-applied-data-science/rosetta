"""
Wrappers to help with Vowpal Wabbit (VW).
"""
import sys

import pandas as pd
import numpy as np

from . import text_processors
from ..common import smart_open
from ..common_math import series_to_frame


def parse_varinfo(varinfo_file):
    """
    Uses the output of the vw-varinfo utility to get a DataFrame with variable
    info.

    Parameters
    ----------
    varinfo_file : Path or buffer
        The output of vw-varinfo
    """
    with smart_open(varinfo_file) as open_file:
        # For some reason, pandas is confused...so just split the lines
        # Create a dict {item1: [...], item2: [...],...} for each item in the
        # header
        header = open_file.next().split()
        rows = {col_name: [] for col_name in header}
        for line in open_file:
            for i, item in enumerate(line.split()):
                rows[header[i]].append(item)

    # Create a data frame
    varinfo = pd.DataFrame(rows)
    # Format columns correctly
    varinfo.FeatureName = varinfo.FeatureName.str.replace('^', '')
    varinfo.HashVal = varinfo.HashVal.astype(int)
    varinfo.MaxVal = varinfo.MaxVal.astype(float)
    varinfo.MinVal = varinfo.MinVal.astype(float)
    varinfo.RelScore = (
        varinfo.RelScore.str.replace('%', '').astype(float) / 100)
    varinfo.Weight = varinfo.Weight.astype(float)

    # Rename columns to decent Python names
    varinfo = varinfo.rename(
        columns={'FeatureName': 'feature_name', 'HashVal': 'hash_val',
            'MaxVal': 'max_val', 'MinVal': 'min_val', 'RelScore': 'rel_score',
            'Weight': 'weight'}).set_index('hash_val')

    return varinfo


def parse_lda_topics(topics_file, num_topics, normalize=True):
    """
    Returns a DataFrame representation of the topics output of an lda VW run.

    Parameters
    ----------
    topics_file : filepath or buffer
        The --readable_model output of a VW lda run
    num_topics : Integer
        The number of topics in every valid row
    normalize : Boolean
        Normalize the rows so that they represent probabilities of topic
        given hash_val

    Notes
    -----
    The trick is dealing with lack of a marker for the information printed
    on top, and the inconsistant delimiter choice.
    """
    fmt = 'topic_%0' + str(num_topics // 10 + 1) + 'd'
    topics = {fmt % i: [] for i in range(num_topics)}
    topics['hash_val'] = []
    # The topics file contains a bunch of informational printout stuff at
    # the top.  Figure out what line this ends on
    with smart_open(topics_file, 'r') as open_file:
        # Once we detect that we're in the valid rows, there better not be
        # any exceptions!
        in_valid_rows = False
        for line in open_file:
            try:
                # If this row raises an exception, then it isn't a valid row
                # Sometimes trailing space...that's the reason for split()
                # rather than csv.reader or a direct pandas read.
                split_line = line.split()
                hash_val = int(split_line[0])
                topic_weights = [float(item) for item in split_line[1:]]
                assert len(topic_weights) == num_topics
                for i, weight in enumerate(topic_weights):
                    topics[fmt % i].append(weight)
                topics['hash_val'].append(hash_val)
                in_valid_rows = True
            except (ValueError, IndexError, AssertionError):
                if in_valid_rows:
                    raise

    topics = pd.DataFrame(topics).set_index('hash_val')
    if normalize:
        topics = topics.div(topics.sum(axis=1), axis=0)

    return topics


def find_start_line_lda_predictions(predictions_file, num_topics):
    """
    Return the line number (zero indexed) of the start of the last set of
    predictions in predictions_file.

    Parameters
    ----------
    predictions_file : filepath or buffer
        The -p output of a VW lda run
    num_topics : Integer
        The number of topics you should see

    Notes
    -----
    The predictions_file contains repeated predictions...one for every pass.
    We parse out and include only the last predictions by looking for repeats
    of the first lines doc_id field.  We thus, at this time, require the VW
    formatted file to have, in the last column, a unique doc_id associated
    with the doc.
    """
    with smart_open(predictions_file) as open_file:
        for line_num, line in enumerate(open_file):
            split_line = line.split()
            # Currently only deal with topics + a doc_id
            assert len(split_line) == num_topics + 1, "Is num_topics correct?"
            doc_id = split_line[-1]
            if line_num == 0:
                first_doc_id = doc_id
            if doc_id == first_doc_id:
                start_line = line_num

    return start_line


def parse_lda_predictions(
    predictions_file, num_topics, start_line, normalize=True):
    """
    Return a DataFrame representation of a VW prediction file.

    Parameters
    ----------
    predictions_file : filepath or buffer
        The -p output of a VW lda run
    num_topics : Integer
        The number of topics you should see
    start_line : Integer
        Start reading the predictions file here.
        The predictions file contains repeated predictions, one for every pass.
        You generally do not want every prediction.
    normalize : Boolean
        Normalize the rows so that they represent probabilities of topic
        given doc_id.
    """
    doc_id_stored = []
    lines = []
    # Use this rather than pandas.read_csv due to inconsistent use of sep
    with smart_open(predictions_file) as open_file:
        # We may have already opened and read this file in order to
        # find the start_line
        open_file.seek(0)
        for line_num, line in enumerate(open_file):
            if line_num < start_line:
                continue
            split_line = line.split()
            topic_weights = split_line[: -1]
            assert len(topic_weights) == num_topics, "Is num_topics correct?"
            lines.append(topic_weights)
            doc_id_stored.append(split_line[-1])

    fmt = 'topic_%0' + str(num_topics // 10 + 1) + 'd'
    topic_names = [fmt % i for i in range(num_topics)]
    predictions = pd.DataFrame(
        lines, index=doc_id_stored, columns=topic_names).astype(float)
    predictions.index.name = 'doc_id'

    if normalize:
        predictions = predictions.div(predictions.sum(axis=1), axis=0)

    return predictions


class LDAResults(object):
    """
    Facilitates working with results of VW lda runs.

    See http://hunch.net/~vw/  as a starting place for VW information.

    See https://github.com/JohnLangford/vowpal_wabbit/wiki/lda.pdf
    for a brief tutorial of lda in VW.
    """
    def __init__(
        self, topics_file, predictions_file, num_topics, sfile_filter):
        """
        Parameters
        ----------
        topics_file : filepath or buffer
            The --readable_model output of a VW lda run
        predictions_file : filepath or buffer
            The -p output of a VW lda run
        num_topics : Integer
            The number of topics in every valid row
        sfile_filter : filepath, buffer, or loaded text_processors.SFileFilter
        """
        self.num_topics = num_topics

        if not isinstance(sfile_filter, text_processors.SFileFilter):
            sfile_filter = text_processors.SFileFilter.load(sfile_filter)

        self.sfile_frame = sfile_filter.to_frame()

        # Load the topics file
        topics = parse_lda_topics(topics_file, num_topics, normalize=False)
        topics = topics.reindex(index=sfile_filter.id2token.keys())
        topics = topics.rename(index=sfile_filter.id2token)

        # Load the predictions
        start_line = find_start_line_lda_predictions(
            predictions_file, num_topics)
        predictions = parse_lda_predictions(
            predictions_file, num_topics, start_line, normalize=False)

        self.num_docs = len(predictions)
        self.num_tokens = len(topics)
        self.topics = topics.columns.tolist()
        self.tokens = topics.index.tolist()
        self.docs = predictions.index.tolist()

        # Check that the topics/docs/token names are unique with no overlap
        self._check_names(topics, predictions)

        # Set probabilities
        self._set_probabilities(topics, predictions)

    def __repr__(self):
        st = "LDAResults for %d topics, %d docs, %d topics, %d tokens" % (
            self.num_topics, self.num_docs, self.num_topics, self.num_tokens)

        return st

    def _check_names(self, topics, predictions):
        tokens = topics.index
        docs = predictions.index
        topics = topics.columns

        for names in [tokens, docs, topics]:
            assert len(set(names)) == len(names), "Names must be unique"

    def _set_probabilities(self, topics, predictions):
        topic_sums = topics.sum()
        self.pr_topic = topic_sums / topic_sums.sum()

        word_sums = topics.sum(axis=1)
        self.pr_token = word_sums / word_sums.sum()
        self.pr_topic_token = topics / topics.sum().sum()

        doc_sums = predictions.sum(axis=1)
        self.pr_doc = doc_sums / doc_sums.sum()
        self.pr_topic_doc = predictions / predictions.sum().sum()

        # New stuff
        self.pr_token_topic = topics / topics.sum().sum()
        self.pr_token_topic.index.name = 'token'
        self.pr_doc_topic = predictions / predictions.sum().sum()

    def prob_token_topic(
        self, token=None, topic=None, c_token=None, c_topic=None):
        """
        Return joint densities of (token, topic),
        restricted to subsets, conditioned on variables.

        Parameters
        ----------
        token : list-like or string
            Restrict returned probabilities to these tokens
        topic : list-like or string
            Restrict returned probabilities to these topics
        c_token : list-like or string
            Condition on token in c_token
        c_topic : list-like or string
            Condition on topic in c_topic

        Examples
        --------
        prob_token_topic(c_topic=['topic_0'])
          = P(token, topic | topic in ['topic_0'])
          for all possible (token, topic) pairs

        prob_token_topic(token=['war', 'peace'], c_topic=['topic_0'])
          = P(token, topic | topic in ['topic_0'])
          for all (token, topic) pairs with token in ['war', 'peace]

        prob_token_topic(token=['war', 'peace'], topic=['topic_0'])
          = P(token, topic)
          for all (token, topic) pairs
          with token in ['war', 'peace] and topic in ['topic_0']
        """
        df = self._prob_func(
            self.pr_token_topic, token, topic, c_token, c_topic)
        df.index.name = 'token'

        return df

    def prob_doc_topic(self, doc=None, topic=None, c_doc=None, c_topic=None):
        """
        Return joint probabilities of (doc, topic),
        restricted to subsets, conditioned on variables.

        Parameters
        ----------
        doc : list-like or string
            Restrict returned probabilities to these doc_ids
        topic : list-like or string
            Restrict returned probabilities to these topics
        c_doc : list-like or string
            Condition on doc_id in c_doc
        c_topic : list-like or string
            Condition on topic in c_topic

        Examples
        --------
        prob_doc_topic(c_topic=['topic_0'])
          = P(doc, topic | topic in ['topic_0'])
          for all possible (doc, topic) pairs

        prob_doc_topic(doc=['doc0', 'doc1'], c_topic=['topic_0'])
          = P(doc, topic | topic in ['topic_0'])
          for all (doc, topic) pairs with doc in ['doc0', 'doc1']

        prob_doc_topic(doc=['doc0', 'doc1'], topic=['topic_0'])
          = P(doc, topic)
          for all (doc, topic) pairs
          with doc in ['doc0', 'doc1'] and topic in ['topic_0']
        """
        df = self._prob_func(self.pr_doc_topic, doc, topic, c_doc, c_topic)
        df.index.name = 'doc'

        return df

    def cosine_similarity(self, frame1, frame2):
        """
        Computes doc-doc similarity between rows of two frames containing
        document topic weights.

        Parameters
        ----------
        frame1, frame2 : DataFrame or Series
            Rows are different records, columns are topic weights.
            self.pr_topic_g_doc is an example of a (large) frame of this type.

        Returns
        -------
        sims : DataFrame
            sims.ix[i, j] is similarity between frame1[i] and frame2[j]
        """
        # Convert to frames
        frame1 = series_to_frame(frame1)
        frame2 = series_to_frame(frame2)
        # Normalize
        norm = (frame1 * frame1).sum(axis=0).apply(np.sqrt)
        frame1 = frame1.div(norm, axis=1)

        norm = (frame2 * frame2).sum(axis=0).apply(np.sqrt)
        frame2 = frame2.div(norm, axis=1)

        return frame1.T.dot(frame2)

    def _prob_func(self, df, rows, cols, c_rows, c_cols):
        """
        General pmf for functions of two variables.
        For use with prob_token_topic, prob_doc_topic
        """
        df = df.copy()

        if isinstance(rows, basestring):
            rows = [rows]
        if isinstance(cols, basestring):
            cols = [cols]
        if isinstance(c_rows, basestring):
            c_rows = [c_rows]
        if isinstance(c_cols, basestring):
            c_cols = [c_cols]

        # Restrict using the conditionals
        if c_cols is not None:
            df = df.ix[:, c_cols]
        if c_rows is not None:
            df = df.ix[c_rows, :]
        df = df / df.sum().sum()

        # Cut out according to variables
        if cols is not None:
            df = df.ix[:, cols]
        if rows is not None:
            df = df.ix[rows, :]

        return df

    def predict(self, tokenized_text):
        """
        Returns a probability distribution over topics given that a (tokenized)
        document is equal to tokenized_text.

        This is NOT equivalent to prob_token_topic(c_token=tokenized_text),
        since that is an OR statement about the tokens, and this is an AND.

        Parameters
        ----------
        tokenized_text : List of strings
            Represents the tokens that are in some document text.

        Returns
        -------
        prob_topics : Series
            self.pr_topic_g_doc is an example of a (large) frame of this type.

        Notes
        -----
        P(topic | tok1, tok2) \propto P(topic) P(tok1, tok2 | topic)
                              = P(topic) P(tok1 | topic) P(tok2 | topic)
        """
        # P(topic | tok1, tok2) \propto P(topic) P(tok1, tok2 | topic)
        # = P(topic) P(tok1 | topic) P(tok2 | topic)

        # Multiply out P(tok1 | topic) P(tok2 | topic) 
        na_val = 1. / self.num_topics
        fun = lambda tok: (
            self.prob_token_topic(token=tok, topic=self.topics).fillna(na_val)
            .values.ravel())
        probs = reduce(
            lambda x, y: x * y, (fun(tok) for tok in tokenized_text))

        # Multiply by P(topic)
        probs = self.pr_topic * probs

        return probs / probs.sum()

    def print_topics(
        self, num_words=5, outfile=sys.stdout, show_doc_fraction=True):
        """
        Print the top results for self.pr_token_g_topic for all topics

        Parameters
        ----------
        num_words : Integer
            Print the num_words words (ordered by P[w|topic]) in each topic.
        outfile : filepath or buffer
            Write results to this file.
        show_doc_fraction : Boolean
            If True, print doc_fraction along with the topic weight
        """
        header = " Printing top %d tokens in every topic" % num_words
        outstr = "=" * 10 + header + "=" * 10

        for topic_name in self.pr_topic.index:
            outstr += (
                '\n' + "-" * 30 + '\nTopic name: %s.  P[%s] = %.4f' % 
                (topic_name, topic_name, self.pr_topic[topic_name]))
            sorted_topic = self.pr_token_g_topic[topic_name].order(
                ascending=False).head(num_words)

            if show_doc_fraction:
                sorted_topic = self.sfile_frame.join(sorted_topic, how='right')
                sorted_topic = sorted_topic[[topic_name, 'doc_freq']]

            outstr += "\n" + sorted_topic.to_string() + "\n"

        with smart_open(outfile, 'w') as f:
            f.write(outstr)

    @property
    def pr_token_g_topic(self):
        return self.pr_topic_token.div(self.pr_topic, axis=1)

    @property
    def pr_topic_g_token(self):
        return self.pr_topic_token.div(self.pr_token, axis=0).T

    @property
    def pr_doc_g_topic(self):
        # Note:  self.pr_topic is computed using a different file than
        # self.pr_topic_doc....the resultant implied pr_topic series differ
        # unless many passes are used.
        return self.pr_topic_doc.div(self.pr_topic, axis=1)

    @property
    def pr_topic_g_doc(self):
        return self.pr_topic_doc.div(self.pr_doc, axis=0).T
