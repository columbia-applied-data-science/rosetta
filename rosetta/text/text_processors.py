"""
Tokenizer
---------
Classes with a .text_to_token_list method (and a bit more).  Used by other
modules as a means to convert stings to lists of strings.

If you have a function that converts strings to lists of strings, you can
make a tokenizer from it by using MakeTokenizer(my_tokenizing_func).

SparseFormatter
---------------
Classes for converting text to sparse representations (e.g. VW or SVMLight).

SFileFilter
-----------
Classes for filtering words/rows from a sparse formatted file.
"""
from collections import Counter, defaultdict
import hashlib
import random
import re
import math

import nltk
import numpy as np
import pandas as pd

from ..common import smart_open, DocIDError
from ..common_abc import SaveLoad
from . import nlp

from . import streaming_filters

class BaseTokenizer(SaveLoad):
    """
    Base class, don't use directly.
    """
    def text_to_counter(self, text):
        """
        Return a counter associated to tokens in text.
        Filter/transform words according to the scheme this Tokenizer uses.

        Parameters
        ----------
        text : String

        Returns
        -------
        tokens : Counter
            keys = the tokens
            values = counts of the tokens in text
        """
        return Counter(self.text_to_token_list(text))


class MakeTokenizer(BaseTokenizer):
    """
    Makes a subclass of BaseTokenizer out of a function.
    """
    def __init__(self, tokenizer_func):
        """
        Parameters
        ----------
        tokenizer_func : Function
            Takes in strings, spits out lists of strings.
        """
        self.text_to_token_list = tokenizer_func


class TokenizerBasic(BaseTokenizer):
    """
    A simple tokenizer.  Extracts word counts from text.

    Keeps only non-stopwords, converts to lowercase,
    keeps words of length >=2.
    """
    def text_to_token_list(self, text):
        """
        Return a list of tokens.
        Filter/transform words according to the scheme this Tokenizer uses.

        Parameters
        ----------
        text : String

        Returns
        -------
        tokens : List
            Tokenized text, e.g. ['hello', 'my', 'name', 'is', 'ian']
        """
        tokens = nlp.word_tokenize(text, L=2, numeric=False)

        return [word.lower() for word in tokens if not nlp.is_stopword(word)]


class TokenizerPOSFilter(BaseTokenizer):
    """
    Tokenizes, does POS tagging, then keeps words that match particular POS.
    """
    def __init__(
        self, pos_types=[], sent_tokenizer=nltk.sent_tokenize,
        word_tokenizer=TokenizerBasic(), word_tokenizer_func=None,
        pos_tagger=nltk.pos_tag):
        """
        Parameters
        ----------
        pos_types : List of Strings
            Parts of Speech to keep
        sent_tokenizer : Sentence tokenizer function.
            Default: nltk.sent_tokenize
            Splits text into a list of sentences (each sentence is a string)
        word_tokenizer : Subclass of BaseTokenizer.
            Default: TokenizerBasic
            For tokenizing the words.
        word_tokenizer_func : Function
            Converts strings to list of strings.  If given, use this in place
            of word_tokenizer.
        pos_tagger : POS tagging function
            Default: nltk.pos_tag
            Given a list of words, returns a list of tuples (word, POS)
        """
        self.pos_types = set(pos_types)
        self.sent_tokenizer = sent_tokenizer
        self.pos_tagger = pos_tagger
        if not word_tokenizer:
            self.word_tokenizer = MakeTokenizer(word_tokenizer_func)
        else:
            self.word_tokenizer = word_tokenizer

    def text_to_token_list(self, text):
        """
        Tokenize a list of text that (possibly) includes multiple sentences.
        """
        # sentences = [['I am Ian.'], ['Who are you?']]
        sentences = self.sent_tokenizer(text)
        # tokenized_sentences = [['I', 'am', 'Ian.'], ['Who', 'are', 'you?']]
        func = self.word_tokenizer.text_to_token_list
        tokenized_sentences = [func(sent) for sent in sentences]
        # tagged_sentences = [[('I', 'PRP'), ('am', 'VBP'), ...]]
        tagged_sentences = [
            self.pos_tagger(sent) for sent in tokenized_sentences]

        # Returning a list of words that meet the filter criteria
        token_list = sum(
            [self._sent_filter(sent) for sent in tagged_sentences], [])

        return token_list

    def _sent_filter(self, tokenized_sent):
        return [
            word for (word, pos) in tokenized_sent if pos in self.pos_types]


class SparseFormatter(object):
    """
    Base class for sparse formatting, e.g. VW or svmlight.
    Not meant to be directly used.
    """
    def _parse_feature_str(self, feature_str):
        """
        Parses a sparse feature string and returns
        feature_values = {feature1: value1, feature2: value2,...}
        """
        # We currently don't support namespaces, so feature_str must start
        # with a space then feature1[:value1] feature2[:value2] ...
        assert feature_str[0] == ' '
        feature_str = feature_str[1:]

        # The regex splits 'hi:1 bye:' into [('hi', '1'), ('bye', '')]
        fv_list = re.findall(r'(\S+):(\S*)', feature_str)

        feature_values = {
            f: self._string_to_number(v, empty_sub=1) for (f, v) in fv_list}

        return feature_values

    def sstr_to_dict(self, sstr):
        """
        Returns a dict representation of sparse record string.

        Parameters
        ----------
        sstr : String
            String representation of one record.

        Returns
        -------
        record_dict : Dict
            possible keys = 'target', 'importance', 'doc_id', 'feature_values'

        Notes
        -----
        rstrips newline characters from sstr before parsing.
        """
        sstr = sstr.rstrip('\n').rstrip('\r')

        idx = sstr.index(self.preamble_char)
        preamble, feature_str = sstr[:idx], sstr[idx + 1:]

        record_dict = self._parse_preamble(preamble)

        record_dict['feature_values'] = self._parse_feature_str(feature_str)

        return record_dict

    def sstr_to_info(self, sstr):
        """
        Returns the full info dictionary corresponding to a sparse record
        string.  This holds "everything."

        Parameters
        ----------
        sstr : String
            String representation of one record.

        Returns
        -------
        info : Dict
            possible keys = 'tokens', 'target', 'importance', 'doc_id',
                'feature_values', etc...
        """
        info = self.sstr_to_dict(sstr)
        info['tokens'] = self._dict_to_tokens(info)

        return info

    def _dict_to_tokens(self, record_dict):
        token_list = []
        if 'feature_values' in record_dict:
            for feature, value in record_dict['feature_values'].iteritems():
                # If the value is a non-integer score (e.g. tfidf), then
                # it cannot correspond to a number of tokens
                int_value = int(value)
                assert int_value == value
                token_list += [feature] * int_value

        return token_list

    def sstr_to_token_list(self, sstr):
        """
        Convertes a sparse record string to a list of tokens (with repeats)
        corresponding to sstr.

        E.g. if sstr represented the dict {'hi': 2, 'bye': 1}, then
        token_list = ['hi', 'hi', 'bye']  (up to permutation).

        Parameters
        ----------
        sstr : String
            Formatted according to self.format_name
            Note that the values in sstr must be integers.

        Returns
        -------
        token_list : List of Strings
        """
        record_dict = self.sstr_to_dict(sstr)
        return self._dict_to_tokens(record_dict)

    def sfile_to_token_iter(self, filepath_or_buffer, limit=None):
        """
        Return an iterator over filepath_or_buffer that returns, line-by-line,
        a token_list.

        Parameters
        ----------
        filepath_or_buffer : string or file handle / StringIO.
            File should be formatted according to self.format.

        Returns
        -------
        token_iter : Iterator
            E.g. token_iter.next() gets the next line as a list of tokens.
        """
        with smart_open(filepath_or_buffer) as open_file:
            for index, line in enumerate(open_file):
                if index == limit:
                    raise StopIteration
                yield self.sstr_to_token_list(line)

    def _string_to_number(self, string, empty_sub=None):
        """
        Convert a string to either an int or a float, with optional
        substitution for empty strings.
        """
        try:
            return int(string)
        except ValueError:
            pass  # fallback to float
        try:
            return float(string)
        except ValueError:
            # See if it is empty and there is an empty_sub value
            if (string == '') and (empty_sub is not None):
                return empty_sub
            else:
                raise


class VWFormatter(SparseFormatter):
    """
    Converts in and out of VW format (namespaces currently not supported).
    Many valid VW inputs are possible, we ONLY support

    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...

    Every single whitespace, pipe, colon, and newline is significant.

    See:
    https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    http://hunch.net/~vw/validate.html
    """
    def __init__(self):
        self.format_name = 'vw'
        self.preamble_char = '|'

    def get_sstr(
        self, feature_values=None, target=None, importance=None, doc_id=None):
        """
        Return a string reprsenting one record in sparse VW format:

        Parameters
        ----------
        feature_values : Dict-like
            {feature1: value1,...}
        target : Real number
            The value we are trying to predict.
        importance : Real number
            The importance weight to associate to this example.
        doc_id : Number or string
            A name for this example.

        Returns
        -------
        formatted : String
            Formatted in VW format
        """
        if doc_id:
            if re.search(r"[|\s:']", doc_id):
                msg = (
                "Malformed VW string %s.  Strings cannot have |, :, ', "
                "or whitespace")
                raise DocIDError(msg)
            # If doc_id, then we must have importance.
            # The doc_id sits right against the pipe.
            assert importance is not None
            formatted = " %s|" % doc_id
            # If no doc_id, insert a space to the left of the pipe.
        else:
            formatted = " |"

        if importance:
            # Insert a space to the left of importance.
            formatted = " " + str(importance) + formatted

        if target:
            # target gets stuck on the end
            formatted = str(target) + formatted

        # The feature part must start with a space unless there is a namespace.
        formatted += ' '
        for word, count in feature_values.iteritems():
            formatted += "%s:%s " % (word, count)

        # Remove the trailing space...not required but it's screwy to have a
        # space-delimited file with a trailing space but nothing after it!
        if len(feature_values) > 0:
            formatted = formatted.rstrip()

        return formatted

    def _parse_preamble(self, preamble):
        """
        Parse the VW preamble: [target] [Importance [Tag]]
        and return a dict with keys 'doc_id', 'target', 'importance' iff
        the corresponding values were found in the preamble.
        """
        # If preamble was butted directly against a pipe, then the right-most
        # part is a doc_id....extract it and continue.
        if preamble[-1] != ' ':
            doc_id_left = preamble.rfind(' ')
            doc_id = preamble[doc_id_left + 1:]
            preamble = preamble[: doc_id_left]
        else:
            doc_id = None

        # Step from left to right through preamble.
        # We are in the target until we encounter the first space...if there
        # is no target, then the first character will be a space.
        in_target = True
        target = ''
        importance = ''
        for char in preamble:
            if char == ' ':
                in_target = False
            elif in_target:
                target += char
            else:
                importance += char

        parsed = {}
        items = (
            ('doc_id', doc_id), ('target', target), ('importance', importance))
        for key, value in items:
            if value:
                if key in ['target', 'importance']:
                    parsed[key] = self._string_to_number(value)
                else:
                    parsed[key] = value

        return parsed


class SVMLightFormatter(SparseFormatter):
    """
    For formatting in/out of SVM-Light format (info not currently supported)
    http://svmlight.joachims.org/

    <line> .=. <target> <feature>:<value> <feature>:<value> ...
    <target> .=. +1 | -1 | 0 | <float>
    <feature> .=. <integer> | "qid"
    <value> .=. <float>
    <info> .=. <string>
    """
    def __init__(self):
        """
        """
        self.format_name = 'svmlight'
        self.preamble_char = ' '

    def get_sstr(
        self, feature_values=None, target=1, importance=None, doc_id=None):
        """
        Return a string reprsenting one record in SVM-Light sparse format
        <line> .=. <target> <feature>:<value> <feature>:<value>

        Parameters
        ----------
        feature_values : Dict-like
            {hash1: value1,...}
        target : Real number
            The value we are trying to predict.

        Returns
        -------
        formatted : String
            Formatted in SVM-Light
        """
        # For now, just use 0 for <target>
        formatted = str(target) + ' '

        for word, count in feature_values.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted

    def _parse_preamble(self, preamble):
        return {'target': float(preamble)}


class SFileFilter(SaveLoad):
    """
    Filters results stored in sfiles (sparsely formattted bag-of-words files).
    """
    def __init__(self, formatter, bit_precision=18, sfile=None, verbose=True):
        """
        Parameters
        ----------
        formatter : Subclass of SparseFormatter
        bit_precision : Integer
            Hashes are taken modulo 2**bit_precision.  Currently must be < 32.
        sfile : filepath or buffer
            Load this sfile during init
        verbose : Boolean
        """
        assert isinstance(bit_precision, int)

        self.formatter = formatter
        self.bit_precision = bit_precision
        self.verbose = verbose

        self.precision = 2**bit_precision
        self.sfile_loaded = False
        self.bit_precision_required = bit_precision

        if sfile is not None:
            self.load_sfile(sfile)

    def _get_hash_fun(self):
        """
        The fastest is the built in function hash.  Quick experimentation
        shows that this function maps similar words to similar values (not
        cryptographic) and therefore increases collisions...no big deal.

        hashlib.sha224 is up to 224 bit.
        """
        if self.bit_precision <= 64:
            hash_fun = lambda w: hash(w) % self.precision
        elif self.bit_precision <= 224:
            hash_fun = lambda w: (
                int(hashlib.sha224(w).hexdigest(), 16) % self.precision)
        else:
            raise ValueError("Precision above 224 bit not supported")

        return hash_fun

    def load_sfile(self, sfile):
        """
        Load an sfile, building self.token2id

        Parameters
        ----------
        sfile : String or open file
            The sparse formatted file we will load.

        Returns
        -------
        self
        """
        # TODO Allow loading of more than one sfile
        assert not self.sfile_loaded

        # Build token2id
        token2id, token_score, doc_freq, num_docs, idf = (
            self._load_sfile_fwd(sfile))

        self.token2id = token2id
        self.token_score = token_score
        self.doc_freq = doc_freq
        self.num_docs = num_docs
        self.idf = idf

        self.sfile_loaded = True
        self.collisions_resolved = False

    def _load_sfile_fwd(self, sfile):
        """
        Builds the "forward" objects involved in loading an sfile.
        """
        token2id = {}
        token_score = defaultdict(float)
        doc_freq = defaultdict(int)
        num_docs = 0
        idf = defaultdict(float)

        hash_fun = self._get_hash_fun()

        with smart_open(sfile) as open_file:
            # Each line represents one document
            for line in open_file:
                num_docs += 1
                record_dict = self.formatter.sstr_to_dict(line)
                for token, value in record_dict['feature_values'].iteritems():
                    hash_value = hash_fun(token)
                    token2id[token] = hash_value
                    token_score[token] += value
                    doc_freq[token] += 1
                    idf[token] += 1

        for token in idf.iterkeys():
            idf[token] = math.log(num_docs / idf[token])

        return token2id, token_score, doc_freq, num_docs, idf

    def set_id2token(self, seed=None):
        """
        Sets self.id2token, resolving collisions as needed (which alters
        self.token2id)
        """
        self._resolve_collisions(seed=seed)

        self.id2token = {v: k for k, v in self.token2id.iteritems()}

    def _resolve_collisions(self, seed=None):
        """
        Alters self.token2id by finding new id values used using a
        "random probe" method.

        Meant to be called by self.set_id2token.  If you call this by itself,
        then self.token2id is altered, but self.id2token is not!!!!
        """
        id_counts = Counter(self.token2id.values())
        vocab_size = self.vocab_size

        # Make sure we don't have too many collisions
        num_collisions = vocab_size - len(id_counts)
        self._print(
            "collisions = %d, vocab_size = %d" % (num_collisions, vocab_size))
        if num_collisions > vocab_size / 2.:
            msg = (
                "Too many collisions to be efficient: "
                "num_collisions = %d.  vocab_size = %d.  Try using the "
                "function collision_probability to estimate needed precision"
                % (num_collisions, vocab_size))
            raise CollisionError(msg)

        # Seed for testing
        random.seed(seed)

        # Resolve the collisions in this loop
        collisions = (
            tok for tok in self.token2id if id_counts[self.token2id[tok]] > 1)

        for token in collisions:
            old_id = self.token2id[token]
            new_id = old_id
            # If id_counts[old_id] > 1, then the collision still must be
            # resolved.  In that case, change new_id and update id_counts
            if id_counts[old_id] > 1:
                # id_counts is the only dict (at this time) holding every
                # id you have ever seen
                while new_id in id_counts:
                    new_id = random.randint(0, self.precision - 1)
                    new_id = new_id % self.precision
                id_counts[old_id] -= 1
                id_counts[new_id] = 1
            # Update dictionaries
            self.token2id[token] = new_id

        self._print("All collisions resolved")
        self.collisions_resolved = True

    def compactify(self):
        """
        Removes "gaps" in the id values in self.token2id.  Every single id
        value will (probably) be altered.
        """
        # You can't compactify if self.bit_precision is too low
        min_precision = int(np.ceil(np.log2(self.vocab_size)))

        if self.bit_precision < min_precision:
            raise CollisionError(
                "Cannot compactify unless you increase self.bit_precision "
                "to >= %d or remove some tokens" % min_precision)

        new_token2id = {}
        for i, tok in enumerate(self.token2id):
            new_token2id[tok] = i
        self.token2id = new_token2id

        if hasattr(self, 'id2token'):
            self.set_id2token()

        self.set_bit_precision_required()
        self._print(
            "Compactification done.  self.bit_precision_required = %d"
            % self.bit_precision_required)

    def set_bit_precision_required(self):
        """
        Sets self.bit_precision_required to the minimum bit precision b such
        that all token id values are less than 2^b.

        The idea is that only compactification can change this, so we only
        (automatically) call this after compactification.
        """
        max_id = np.max(self.token2id.values())

        self.bit_precision_required = int(np.ceil(np.log2(max_id)))

    def filter_sfile(
        self, infile, outfile, doc_id_list=None, enforce_all_doc_id=True,
        min_tf_idf=0, filters=None):
        """
        Alter an sfile by converting tokens to id values, and removing tokens
        not in self.token2id.  Optionally filters on doc_id, tf_idf and
        user-defined filters.

        Parameters
        ----------
        infile : file path or buffer
        outfile : file path or buffer
        doc_id_list : Iterable over strings
            Keep only rows with doc_id in this list
        enforce_all_doc_id : Boolean
            If True (and doc_id is not None), raise exception unless all doc_id
            in doc_id_list are seen.
        min_tf_idf : int or float
            Keep only tokens whose term frequency-inverse document frequency
            is greater than this threshold. Given a token t and a document d
            in a corpus of documents D, tf_idf is given by the following
            formula:
                tf_idf(t, d, D) = tf(t, d) x idf(t, D),
            where
                (1) tf(t, d) is the number of times the term t shows up in the
                    document d,
                (2) idf(t, D) = log (N / M), where N is the total number of
                    documents in D and M is the number of documents in D which
                    contain the token t. The logarithm is base e.
        filters : iterable over functions
            Each function must take a record_dict as a parameter and return a
            boolean. The record_dict may (and usually should) be altered in
            place. If the return value is False, the record_dict (corresponding
            to a document) is filtered out of the sfile. Both the doc_id_list
            and min_tf_idf parameters are implemented in this style internally.
            If the doc_id_list or min_tf_idf flags are set, those filters will
            run before the those found in filters. See
                rosetta/text/streaming_filters.py
            in the rosetta repository for the implementation details of the
            record_dict and built-in filters as well as explanations of how to
            define more filters.
        """
        assert self.sfile_loaded, "Must load an sfile before you can filter"
        if not hasattr(self, 'id2token'):
            self._print(
                "WARNING:  Filtering an sfile before setting self.id2token.  "
                "The resultant outfile will have collisions and you will not "
                "be able to convert ids back to tokens.\nIt is recommended to "
                "call: self.compactify() then either self.set_id2token() or "
                " self.save() before filtering")

        if filters is None:
            filters = []

        # The doc_id_filter should be run before everything else to avoid
        # unnecessary computations. The min_tf_idf filter is run next. If for
        # some reason this is not the desired the ordering, the user needs to
        # leave the the doc_id_list and min_tf_idf flags must be unset and pass
        # user-defined filters to the filters flag explicitly.
        prefilters = []
        if doc_id_list is not None:
            doc_id_set = set(doc_id_list)
            prefilters.append(streaming_filters.get_doc_id_filter(doc_id_set))
        else:
            doc_id_set = set()

        if min_tf_idf != 0:
            prefilters.append(
                streaming_filters.get_tf_idf_filter(self, min_tf_idf))

        # The token_to_id_filter should be run last so that only the necessary
        # conversions are made.
        postfilters = [streaming_filters.get_token_to_id_filter(self)]

        filters = prefilters + filters + postfilters

        doc_id_seen = set()

        with smart_open(infile) as f, smart_open(outfile, 'w') as g:
            # Each line represents one document
            for line in f:
                record_dict = self.formatter.sstr_to_dict(line)

                doc_id = record_dict['doc_id']
                doc_id_seen.add(doc_id)

                if all(func(record_dict) for func in filters):
                    new_sstr = self.formatter.get_sstr(**record_dict)
                    g.write(new_sstr + '\n')

        if enforce_all_doc_id:
            # Make sure we saw all the doc_id we're supposed to
            assert doc_id_set.issubset(doc_id_seen), (
                "Did not see every doc_id in the passed doc_id_list")

    def filter_extremes(
        self, doc_freq_min=0, doc_freq_max=np.inf, doc_fraction_min=0,
        doc_fraction_max=1, token_score_min=0, token_score_max=np.inf,
        token_score_quantile_min=0, token_score_quantile_max=1):
        """
        Remove extreme tokens from self (calling self.filter_tokens).

        Parameters
        ----------
        doc_freq_min : Integer
            Remove tokens that in less than this number of documents
        doc_freq_max : Integer
        doc_fraction_min : Float in [0, 1]
            Remove tokens that are in less than this fraction of documents
        doc_fraction_max : Float in [0, 1]
        token_score_quantile_min : Float in [0, 1]
            Minimum quantile that the token score (usually total token count)
            can be in.
        token_score_quantile_max : Float in [0, 1]
            Maximum quantile that the token score can be in

        Returns
        -------
        self
        """
        frame = self.to_frame()
        to_remove_mask = (
            (frame.doc_freq < doc_freq_min)
            | (frame.doc_freq > doc_freq_max)
            | (frame.doc_freq < (doc_fraction_min * self.num_docs))
            | (frame.doc_freq > (doc_fraction_max * self.num_docs))
            | (frame.token_score < token_score_min)
            | (frame.token_score > token_score_max)
            | (frame.token_score
                < frame.token_score.quantile(token_score_quantile_min))
            | (frame.token_score
                > frame.token_score.quantile(token_score_quantile_max))
            )

        self._print(
            "Removed %d/%d tokens" % (to_remove_mask.sum(), len(frame)))
        self.filter_tokens(frame[to_remove_mask].index)

    def filter_tokens(self, tokens):
        """
        Remove tokens from appropriate attributes.

        Parameters
        ----------
        tokens : String or iterable over strings
            E.g. a single token or list of tokens

        Returns
        -------
        self
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        for tok in tokens:
            id_value = self.token2id[tok]
            self.token2id.pop(tok)
            self.token_score.pop(tok)
            self.doc_freq.pop(tok)
            if hasattr(self, 'id2token'):
                self.id2token.pop(id_value)

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def to_frame(self):
        """
        Return a dataframe representation of self.
        """
        token2id = self.token2id
        token_score = self.token_score
        doc_freq = self.doc_freq

        frame = pd.DataFrame(
            {'token_score': [token_score[tok] for tok in token2id],
             'doc_freq': [doc_freq[tok] for tok in token2id]},
            index=[tok for tok in token2id])
        frame['doc_fraction'] = frame.doc_freq / float(self.num_docs)
        frame.index.name = 'token'

        return frame

    @property
    def vocab_size(self):
        return len(self.token2id)

    def save(self, savepath, protocol=-1, set_id2token=True):
        """
        Pickle self to outfile.

        Parameters
        ----------
        savefile : filepath or buffer
        protocol : 0, 1, 2, -1
            0 < 1 < 2 in terms of performance.  -1 means use highest available.
        set_id2token : Boolean
            If True, set self.id2token before saving.
            Used to associate tokens with the output of a VW file.
        """
        if set_id2token:
            self.set_id2token()

        SaveLoad.save(self, savepath, protocol=protocol)


def collision_probability(vocab_size, bit_precision):
    """
    Approximate probability of at least one collision
    (assuming perfect hashing).  See the Wikipedia article on
    "The birthday problem" for details.

    Parameters
    ----------
    vocab_size : Integer
        Number of unique words in vocabulary
    bit_precision : Integer
        Number of bits in space we are hashing to
    """
    exponent = - vocab_size * (vocab_size - 1) / 2.**bit_precision

    return 1 - np.exp(exponent)


class CollisionError(Exception):
    pass
