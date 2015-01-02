"""
Streaming Filters
-----------------

These are filters meant to be used by the SFileFilter class in its
filter_sfile() method. It is a "streaming" filter in the sense that it works
only line by line during its filtering and hence any "global" data needs to be
pre-computed by the SFileFilter class itself. The "lines" are represented
internally in SFileFilter by record_dict dictionaries. The functions in this
file are not streaming filters themselvers, but instead return streaming
filters (the reason being because start-up work is necessary for each
function).

The prototype of a streaming filter is as follows:

    Parameters
    ----------
    record_dict : dict

    Returns
    -------
    keep_doc : bool

The job of the streaming filter is two-fold: (1) to make any desired _changes_
to the record_dict before it is formatted and written out; and (2) to decide
whether the line should be dropped altogether. The record_dict should be
changed in-place. An example record_dict is as follows:

    record_dict = {
        'feature_values': {'word1': 1, 'word2': 2},
        'target': 0.3,
        'importance': 1,
        'doc_id': 'doc1'}

Not all keys need be present, but it is important that _no_ other keys be
added. In almost all cases the thing to do is to somehow change
record_dict['feature_values'] (for example, dropping some tokens under certain
conditions). The return value keep_doc is what determines if the line should be
dropped altogether: you should return false if you wish to drop a line.

If f1 an f2 are streaming filters, then an example of their use is as follows:
    
    s_file_filter.filter_sfile(infile, outfile, filters=[f1, f2])

The order of the filters is important! The filters are applied in the order
they are listed. It is up to the user to make sure that his/her ordering makes
sense.
"""

def get_doc_id_filter(doc_id_set):
    """
    Parameters
    ----------
    doc_id_set : set

    Returns
    -------
    doc_id_filter : function

    Note
    ----
    It is recommended that this filter be used before any others in order to
    minimize unnecessary computations.
    """
    def doc_id_filter(record_dict):
        doc_id = record_dict['doc_id']
        keep_doc = doc_id in doc_id_set
        return keep_doc
    return doc_id_filter


def get_tf_idf_filter(sfile_filter, min_tf_idf):
    """
    Parameters
    ----------
    sfile_filter : instance of SFileFilter
    min_tf_idf : numeric

    Returns
    -------
    tf_idf_filter : function
    """
    idf = sfile_filter.idf
    def tf_idf_filter(record_dict):
        feature_values = record_dict['feature_values']
        tokens = feature_values.keys()
        for token in tokens:
            if idf[token] * feature_values[token] < min_tf_idf:
                del feature_values[token]
        keep_doc = True
        return keep_doc
    return tf_idf_filter


def get_min_token_filter(min_tokens):
    """
    Parameters
    ----------
    min_tokens : numeric

    Returns
    -------
    min_token_filter : function

    Note
    ----
    If the goal is to only keep documents whose final token count is greater
    than min_tokens, this filter should be used last.
    """
    def min_token_filter(record_dict):
        token_count = sum(record_dict['feature_values'].values())
        keep_doc = token_count >= min_tokens
        return keep_doc
    return min_token_filter


def get_token_to_id_filter(sfile_filter):
    """
    Parameters
    ----------
    sfile_filter : instance of SFileFilter

    Returns
    -------
    token_to_id_filter : function
    """
    token2id = sfile_filter.token2id
    def token_to_id_filter(record_dict):
        record_dict['feature_values'] = {
            token2id[token]: value
            for token, value
            in record_dict['feature_values'].iteritems()
            if token in token2id}
        keep_doc = True
        return keep_doc
    return token_to_id_filter
