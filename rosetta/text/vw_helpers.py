"""
Wrappers to help with Vowpal Wabbit (VW).
"""
import sys

from collections import Counter

import pandas as pd
import numpy as np
from scipy.special import psi  # gamma function utils

from . import text_processors
from ..common import smart_open, TokenError
from ..common_math import series_to_frame


###############################################################################
# Globals
###############################################################################

EPS = 1e-100


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
                 'MaxVal': 'max_val', 'MinVal': 'min_val',
                 'RelScore': 'rel_score', 'Weight': 'weight'}
    ).set_index('hash_val')

    return varinfo


def parse_lda_topics(topics_file, num_topics, max_token_hash=None,
                     normalize=True, get_iter=False):
    """
    Returns a DataFrame representation of the topics output of an lda VW run.

    Parameters
    ----------
    topics_file : filepath or buffer
        The --readable_model output of a VW lda run
    num_topics : Integer
        The number of topics in every valid row
    max_token_hash : Integer
        Reading of token probabilities from the topics_file will ignore all
        token with hash above this value. Useful, when you know the max hash
        value of your tokens.
    normalize : Boolean
        Normalize the rows of the data frame so that they represent
        probabilities of topic given hash_val.
    get_iter : Boolean
        if True will return a iterator yielding dict of hash and token vals

    Notes
    -----
    The trick is dealing with lack of a marker for the information printed
    on top, and the inconsistant delimiter choice.
    """
    topics_iter = _parse_lda_topics_iter(topics_file=topics_file,
                                         num_topics=num_topics,
                                         max_token_hash=max_token_hash,
                                         normalize=normalize)
    if get_iter:
        return topics_iter
    else:
        topics = [t for t in topics_iter]
        topics = pd.DataFrame.from_records(topics, index='hash_val')
        if normalize:
            topics = topics.div(topics.sum(axis=1), axis=0)
        return topics


def _parse_lda_topics_iter(topics_file, num_topics, max_token_hash,
                           normalize):
    fmt = 'topic_%0' + str(len(str(num_topics))) + 'd'
    fmt_array = [fmt % i for i in xrange(num_topics)]
    # The topics file contains a bunch of informational printout stuff at
    # the top.  Figure out what line this ends on
    with smart_open(topics_file, 'r') as open_file:
        # Once we detect that we're in the valid rows, there better not be
        # any exceptions!
        in_valid_rows = False
        for i, line in enumerate(open_file):
            try:
                # If this row raises an exception, then it isn't a valid row
                # Sometimes trailing space...that's the reason for split()
                # rather than csv.reader or a direct pandas read.
                split_line = line.split()
                hash_val = int(split_line[0])
                if max_token_hash is not None and hash_val > max_token_hash:
                    break
                topic_weights = np.array(split_line[1:]).astype(float)
                topic_len = len(topic_weights)
                assert topic_len == num_topics
                if normalize:
                    topic_weights = topic_weights/topic_weights.sum()
                topic_dict = dict(zip(fmt_array, topic_weights))
                topic_dict.update({'hash_val': hash_val})
                in_valid_rows = True
                yield topic_dict
            except (ValueError, IndexError, AssertionError):
                if in_valid_rows:
                    raise


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


def parse_lda_predictions(predictions_file, num_topics, start_line,
                          normalize=True, get_iter=False):
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
    get_iter : Boolean
        if True will return a iterator yielding dict of doc_id and topic probs
    """

    predictions_iter = _parse_lda_predictions_iter(predictions_file, num_topics,
                                                   start_line, normalize)
    if get_iter:
        return predictions_iter
    else:
        predictions = [p for p in predictions_iter]
        predictions = pd.DataFrame.from_records(predictions, index='doc_id')
        if normalize:
            predictions = predictions.div(predictions.sum(axis=1), axis=0)
        return predictions


def _parse_lda_predictions_iter(predictions_file, num_topics, start_line,
                                normalize):
    fmt = 'topic_%0' + str(len(str(num_topics))) + 'd'
    fmt_array = [fmt % i for i in xrange(num_topics)]
    # Use this rather than pandas.read_csv due to inconsistent use of sep
    with smart_open(predictions_file) as open_file:
        # We may have already opened and read this file in order to
        # find the start_line
        open_file.seek(0)
        for line_num, line in enumerate(open_file):
            if line_num < start_line:
                continue
            split_line = line.split()
            topic_weights = np.array(split_line[:-1]).astype(float)
            topic_len = len(topic_weights)
            assert topic_len == num_topics
            if normalize:
                topic_weights = topic_weights/topic_weights.sum()
            topic_dict = dict(zip(fmt_array, topic_weights))
            topic_dict.update({'doc_id': split_line[-1]})
            yield topic_dict


class LDAResults(object):
    """
    Facilitates working with results of VW lda runs.  Only useful when you're
    following the workflow outlined here:

    https://github.com/columbia-applied-data-science/rosetta/blob/master/examples/vw_helpers.md
    """
    def __init__(self, topics_file, predictions_file, sfile_filter,
                 num_topics=None, alpha=None, verbose=False):
        """
        Parameters
        ----------
        topics_file : filepath or buffer
            The --readable_model output of a VW lda run
        predictions_file : filepath or buffer
            The -p output of a VW lda run
        num_topics : Integer or None
            The number of topics in every valid row; if None will infer num
            topics from predictions_file
        sfile_filter : filepath, buffer, or loaded text_processors.SFileFilter
            Contains the token2id and id2token mappings
        alpha : Float
            Value of topics Dirichlet hyperparameter used (by VW).
            Needed if you want to do self.predict().
        verbose : Boolean
        """
        if num_topics is None:
            with open(predictions_file) as f:
                num_topics = len(f.readline().split())-1
        self.num_topics = num_topics
        self.alpha = alpha
        self.verbose = verbose

        if not isinstance(sfile_filter, text_processors.SFileFilter):
            sfile_filter = text_processors.SFileFilter.load(sfile_filter)

        self.sfile_frame = sfile_filter.to_frame()

        # Load the topics file
        topics = parse_lda_topics(topics_file, num_topics,
                                  max(sfile_filter.id2token.keys()),
                                  normalize=False)
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

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, tokenlist):
        self._tokens = tokenlist
        self._tokenset = set(tokenlist)

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
        """
        Set probabilities that we store as attributes.
        Others can be derived and appear as "properties"
        (in the decorator sense).
        """
        self._lambda_word_sums = topics.sum()
        self.pr_topic = self._lambda_word_sums / self._lambda_word_sums.sum()

        # tokens & topic
        word_sums = topics.sum(axis=1)
        self.pr_token = word_sums / word_sums.sum()
        self.pr_token_topic = topics / self._lambda_word_sums.sum()
        self.pr_token_topic.index.name = 'token'

        # docs & topics
        doc_sums = predictions.sum(axis=1)
        self.pr_doc = doc_sums / doc_sums.sum()
        self.pr_doc_topic = predictions / predictions.sum().sum()

    def prob_token_topic(self, token=None, topic=None, c_token=None,
                         c_topic=None):
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

    def predict(self, tokenized_text, maxiter=50, atol=1e-3,
                raise_on_unknown=False):
        """
        Returns a probability distribution over topics given that one
        (tokenized) document is equal to tokenized_text.

        This is NOT equivalent to prob_token_topic(c_token=tokenized_text),
        since that is an OR statement about the tokens, and this is an AND.

        Parameters
        ----------
        tokenized_text : List of strings
            Represents the tokens that are in some document text.
        maxiter : Integer
            Maximum iterations used in updating parameters.
        atol : Float
            Absolute tolerance for change in parameters before converged.
        raise_on_unknown : Boolean
            If True, raise TokenError when all tokens are unknown to
            this model.

        Returns
        -------
        prob_topics : Series
            self.pr_topic_g_doc is an example of a (large) frame of this type.

        Notes
        -----
        Treats this as a new document and figures out topic weights for it
        using the existing token-topic weights.  Does NOT update previous
        results/weights.
        """
        # Follows Hoffman et al "Online learning for latent Dirichlet..."
        # Code is adapted from gensim.LDAModel.__getitem__
        assert self.alpha is not None, (
            "Must set self.alpha to use predict.  "
            "Do this during initialization")

        counts = Counter(tokenized_text)
        counts = pd.Series(
            {k: counts[k] for k in counts if k in self._tokenset}
            ).astype(float)

        if len(counts) == 0 and raise_on_unknown:
            raise TokenError(
                "No tokens in tokenized_text have been seen before by this "
                "LDAResults")

        # Do an "E step"
        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = pd.Series(
            np.random.gamma(100., 1. / 100., self.num_topics),
            index=self.topics)
        Elogtheta = pd.Series(
            self._dirichlet_expectation(gamma), index=self.topics)
        expElogtheta = np.exp(Elogtheta)
        expElogbeta = self._expElogbeta.loc[counts.keys()]

        # The optimal phi_{dwk} (as a function of k) is proportional to
        # expElogtheta_k * expElogbeta_w.
        # phinorm is the normalizer.
        phinorm = expElogbeta.dot(expElogtheta) + EPS

        loop_count = 0
        mean_change = atol + 1
        while (loop_count < maxiter) and (mean_change > atol):
            lastgamma = gamma

            # We represent phi implicitly here.
            # Substituting the value of the optimal phi back into
            # the update for gamma gives this update. Cf. Lee&Seung 2001.
            gamma = (
                self.alpha
                + expElogtheta
                * (counts / phinorm).dot(expElogbeta))
            Elogtheta = self._dirichlet_expectation(gamma)
            expElogtheta = np.exp(Elogtheta)
            phinorm = expElogbeta.dot(expElogtheta) + EPS
            # If gamma hasn't changed much, we're done.
            mean_change = (np.fabs(gamma - lastgamma)).mean()

            loop_count += 1

        self._print(
            "Prediction done:  Converged = %s.  loop_count = %d, mean_change"
            "= %f" % (mean_change <= atol, loop_count, mean_change))

        return gamma / gamma.sum()

    def _print(self, msg, outfile=sys.stderr):
        if self.verbose:
            outfile.write(msg)

    @property
    def _expElogbeta(self):
        """
        Return exp{E[log(beta)]} for beta ~ Dir(lambda), and lambda the
        topic-word weights.
        """
        # Get lambda, the dirichlet parameter originally returned by VW.
        lam = self._lambda_word_sums * self.pr_token_topic

        return np.exp(self._dirichlet_expectation(lam + EPS))

    def _dirichlet_expectation(self, alpha):
        """
        For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`, equal to a
        digamma function.
        """
        return psi(alpha) - psi(alpha.sum())

    def print_topics(self, num_words=5, outfile=sys.stdout,
                     show_doc_fraction=True):
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
        return self.pr_token_topic.div(self.pr_topic, axis=1)

    @property
    def pr_topic_g_token(self):
        return self.pr_token_topic.div(self.pr_token, axis=0).T

    @property
    def pr_doc_g_topic(self):
        # Note:  self.pr_topic is computed using a different file than
        # self.pr_doc_topic....the resultant implied pr_topic series differ
        # unless many passes are used.
        return self.pr_doc_topic.div(self.pr_topic, axis=1)

    @property
    def pr_topic_g_doc(self):
        return self.pr_doc_topic.div(self.pr_doc, axis=0).T
