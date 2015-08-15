import re

try:
    from itertools import izip, chain
except ImportError:
    from itertools import chain
    izip = zip


###############################################################################
# Globals
###############################################################################
stopwords_eng = set(
    'a,able,about,across,after,all,almost,also,am,among,an,'
    'and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,'
    'do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,'
    'hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,'
    'likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,'
    'only,or,other,our,own,rather,said,say,says,she,should,since,so,some,'
    'than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,'
    'us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,'
    'with,would,yet,you,your'.split(','))


def word_tokenize(text, L=1, numeric=True):
    """
    Word tokenizer to replace the nltk.word_tokenize()

    Paramters
    ---------
    text : string
    L : int, min length of word to return
    numeric : bool, True if you want to include numerics
    """
    text = re.sub(
        r'(?:\s|\[|\]|\(|\)|\{|\}|;|,|\.(\s|$)|:|\n|\r|\?|\!|\"|\-)', r'  ',
        text)
    if numeric:
        word_list = re.findall(
            r'(?:\s|^)([A-Za-z\.\'&]{%s,}|[0-9]{%s,}|\
            (?:(?<=.|\s)[A-Z]\.)+)(?:\s|$)' % (L, L), text)
    else:
        word_list = re.findall(
            r'(?:\s|^)([A-Za-z\.\'&]{%s,}|(?:(?<=.|\s)[A-Z]\.)+)(?:\s|$)' % L,
            text)

    return word_list


def is_stopword(string):
    return string.lower() in stopwords_eng


def is_letter(s):
    try:
        return len(s) == 1 and s.isalpha()
    except:
        return False


def bigram_tokenize_iter(
    text, word_tok=word_tokenize, skip_regex=r'\.|,|:|;|\?|!',
    **word_tok_kwargs):
    """
    Bigram tokenizer generator function.

    Paramters
    ---------
    text : string
    word_tok : function
        a word tokenizer function that takes in a string and returns a list
        of strings (tokens)
    skip_regex : string, or raw string, regular expression
        if a word pair is seperated by a match of the regular expression the
        pair will be ingored; for example, r'\.|,|:|;|\?|!' makes sure no word
        pairs separated by basic punctation are included; to inlcude all word
        pairs let skip_regex=''
    word_tok_kwargs : kwargs dict
        kwargs compatible with the work_tok api
    Returns
    -------
    bigram_iter : iterator
    """
    text_frags = re.split(skip_regex, text)
    word_lists = [word_tok(frag, **word_tok_kwargs) for frag in text_frags]
    bigram_iter = izip(word_lists[0], word_lists[0][1:])
    for words in word_lists[1:]:
        bigram_iter = chain(bigram_iter, izip(words, words[1:]))
    return bigram_iter


def bigram_tokenize(text, word_tok=word_tokenize,
        skip_regex=r'\.|,|:|;|\?|!', **word_tok_kwargs):
    """
    Same as bigram_tokenize_iter, except returns a list.
    """
    bigram_iter = bigram_tokenize_iter(text, word_tok, skip_regex,
            **word_tok_kwargs)
    return [bg for bg in bigram_iter]

bigram_tokenize.__doc__ += bigram_tokenize_iter.__doc__
