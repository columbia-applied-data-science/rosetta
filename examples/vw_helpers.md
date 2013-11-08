Working with Vowpal Wabbit (VW)
===============================

To work with the `dspy` utilities you need to:

* Clone the [dspy repo][dspyrepo]  and read `README.md`.

Create the sparse file (sfile)
------------------------------

Assume you have a `base_path` (directory), called `my_base_path`, under which you have all the documents you want to analyze.

### Method 1: From a `TextFileStreamer`

A `TextFileStreamer` creates streams of `info` (e.g. tokens, `doc_id`, and more) from a source of text files.  We can use the `.to_vw()` method to convert this stream into a VW formatted file.

The `TextFileStreamer` needs a method to convert the text files to a list of strings (the *tokens*).  To do this we will use a `Tokenizer`.  We have provided a very simple one for you, the `TokenizerBasic`.  To create your own, you simply need to subclass `BaseTokenizer` with a class that has a method, `.text_to_token_list()` that takes in a string (representing a single document) and spits out a list of strings (the *tokens*).  If you already have such a function, then you can create a tokenizer by doing:

    my_tokenizer = MakeTokenizer(my_tokenizing_func)

Once you have a tokenizer, just initialize a streamer and write the VW file.

```python
from dspy import TextFileStreamer, TokenizerBasic

my_tokenizer = TokenizerBasic()
stream = TextFileStreamer(text_base_path='bodyfiles', tokenizer=my_tokenizer)
stream.to_vw('doc_tokens.vw', n_jobs=-1)
```

### Method 2: `files_to_vw.py`
`files_to_vw.py` is a fast and simple command line utility for converting files to VW format.  Installing `dspy` will put these utilities in your path.

* Try converting the first 5 files in `my_base_path`.  The following should print 5 lines of of results, in [vw format][vwinput]

```bash
find my_base_path -type f | head -n 5 | python files_to_vw.py
```

Convert the entire directory quickly.

```bash
files_to_vw.py --base_path my_base_path --n_jobs -2 -o doc_tokens.vw
```

* The `-o` option is the path to your output file.
* For lots of small files, set `--chunksize` to something larger than the default (1000).  This is the number one parameter for performance optimization.
* To see an explanation for all options, type `files_to_vw.py -h`.


#### To use a custom tokenizer with Method 1
The utility `files_to_vw` uses a `Tokenizer` to convert the text files to lists of strings (the *tokens*).  The default tokenizer is `TokenizerBasic`, which removes stopwords, converts to lowercase, and that's it.  You of course will want to create custom tokenizers.  To create your own, you simply need to subclass `BaseTokenizer` with a class that has a method, `.text_to_token_list()` that takes in a string (representing a single document) and spits out a list of strings (the *tokens*).  If you already have such a function, then you can create a tokenizer by doing:

    my_tokenizer = MakeTokenizer(my_tokenizing_func)

In any case, the steps are:

* Create a `Tokenizer` and pickle it using `my_tokenizer.save(my_filename.pkl)`.  Note that any subclass of `BaseTokenizer` automatically inherits a `.save` method.
* Pass this path as the `--tokenizer_pickle` option to `files_to_vw.py`
* If you think this tokenizer is useful for everyone, then submit an issue requesting this be added to the standard tokenizers, then it can be called with the `--tokenizer_type` argument.



Quick test of VW on this `sfile`
--------------------------------

    rm -f *cache
    vw --lda 5 --cache_file doc_tokens.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16 --lda_D 10000 --lda_rho 0.1 --lda_alpha 1 doc_tokens.vw

* The call `vw --lda 5` means run LDA and use 5 topics.
* The final argument, `doc_tokens.vw` is our input file.  Alternatively this could be piped in.
* The `--cache_file` option means "during the first pass, convert the input to a binary 'cached' format and use that for subsequent results.  The `rm -f *cache` is important since if you don't erase the cache file, `VW` will re-use the old one, even if you specify a new input file!  Alternatively, you can pass the `-k` flag to "kill the cache file."
* `--passes 10` means do 10 passes
* `-p prediction.dat` stores the predictions (topic weights for each doc) in the file `prediction.dat`
* `--readable_model topics.dat` stores the word-topic weights in the file `topics.dat`
* The `--bit_precision 16` option means: "Use 16 bits of precision" when [hashing][hashing] tokens.  This will cause many collisions but won't effect the results much at all.
* `--lda_D 10000` means, "we will see 10000 unique documents."  If this is too low, the prior has too much of an effect.  If this is too high, the prior does nothing.
* `--lda_rho` is the prior parameter controlling word probabilities (== 1 if you expect a flat distribution of words).  We set it to 0.1 since most words appear not very often.
* `--lda_alpha` is the prior parameter controlling topic probabilities (== 1 if you expect a flat distribution of topics).
* See [this slideshow][vwlda] about LDA in VW, and [this slideshow][vwtricks] for some VW technical tricks.

This produces two files:

* `prediction.dat`.  Each row is one document.  The last column is the `doc_id`, the first columns are the (un-normalized) topic weights.  Dividing each row by the row sum, each row would be `P[topic | document]`.  Note that a new row is printed for *every* pass.  So if you run with `--passes 10`, there total number of rows will be 10 times the number of documents.
* `topics.dat`.  Each row is a token.  The first column is the hash value for that token.  The columns are, after normalization, `P[token | topic]`.  Note that the hash values run from 0 to `2^bit_precision`.  So even if the token corresponding to hash value 42 never appears in your documents, it will appear in this output file (probably with a complete garbage value).


Working with an `SFileFilter` and `LDAResults`
----------------------------------------------

There are some issues with using the raw `prediction.dat` and `topics.dat` files.  For one, the token hash values are not very interpretable--you want to work with actual English words.  Moreover, unless you allow for a very large hash space, you will have collisions.  Second, you will want some quick means to drop tokens from the vocabulary, or drop documents from the corpus without having to regenerate the VW file.  And finally, they need to be loaded into some suitable data structure for analysis.

### Step 1:  Make an `SFileFilter`

```python
from declass import SFileFilter, VWFormatter
sff = SFileFilter(VWFormatter())
sff.load_sfile('doc_tokens.vw')

df = sff.to_frame()
df.head()
df.describe()

sff.filter_extremes(doc_freq_min=5, doc_fraction_max=0.8)
sff.compactify()
sff.save('sff_file.pkl')
```

* `.to_frame()` returns a DataFrame representation that is useful for deciding which tokens to filter.
* `.filter_extremes` removes low/high frequency tokens from our filter's internal dictionaries.  It's just like those tokens were never present in the original text.
* `.compactify` removes "gaps" in the sequence of numbers (ids) in `self.token2id`.  Once you have done this, the `ids` used by your filter will be the numbers 0 to V-1 where V is the total vocabulary size (= `sff.vocab_size`) rather than 0 to 2^b - 1.  This is often a much lower number (= `sff.bit_precision_required`).
* `.save` first sets the inverse mapping, `self.id2token`, then saves to disk.  To set the inverse mapping, we first resolve collisions by changing the id values for tokens that collide.  Note that if we didn't filter extreme tokens before resolving collisions, then we would have many tokens in our vocab, and there is a good chance the collisions would not be able to be resolved!

### Step 2a:  Run VW on filtered output
First save a "filtered" version of `doc_tokens.vw`.

```python
sff.filter_sfile('doc_tokens.vw', 'doc_tokens_filtered.vw')
```
Our filtered output, `doc_tokens_filtered.vw` has replaced tokens with the id values that the `sff` chose.  This forces VW to use the values we chose (VW's hasher maps integers to integers, modulo `2^bit_precision`).  We can also filter based on `doc_id` as follows

```python
meta = pd.read_csv('path_to_metadata.csv').set_index('doc_id')
doc_id_to_keep = meta[meta['administration'] == 'Nixon'].index
sff.filter_sfile(
    'doc_tokens.vw', 'doc_tokens_filtered.vw', doc_id_list=doc_id_to_keep)
```

Now run VW.

```
rm -f *cache
vw --lda 5 --cache_file ddrs.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16 doc_tokens_filtered.vw
```
It is very important that the bit precision for VW, set with `--bit_precision 16` is greater than or equal to `sff.bit_precision_required`.  If you don't then the hash values used by VW will not match up with the tokens stored in `sff.id2token`.


### Step 2b:  Filter "on the fly" using a saved `sff`
The workflow in step 2a requires making the intermediate file `doc_tokens_filtered.vw`.  Keeping track of all these filtered outputs is an issue.  If you already need to keep track of a saved sff, you might as well use that as your [one and only one][spot] reference.

```
rm -f *cache
filter_sfile.py -s sff_file.pkl  doc_tokens.vw  \
    | vw --lda 5 --cache_file ddrs.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16
```
The python function `filter_sfile.py` takes in `ddrs.vw` and streams a filtered sfile to stdout.  The `|` connects VW to this stream.  Notice we no longer specify an input file to VW (previously we passed it a `doc_tokens_filtered.vw` positional argument).

### Step 3:  Read the results with `LDAResults`

You can view the topics and predictions with this:

```python
from dspy.text.vw_helpers import LDAResults
num_topics = 5
lda = LDAResults('topics.dat', 'prediction.dat', num_topics, 'sff_file.pkl')
lda.print_topics()
```

`.print_topics()` prints the "top" topic weights (= `P[token | topic]`) along with the document frequencies.  If the top topics have low document frequency, then something went wrong!

`lda` stores many joint and marginal probability distributions.  These are stored as pandas Series and DataFrame attributes with the prefix `pr_`.  For example, `lda.pr_token_topic` is the joint distribution of tokens and topics.  `lda.pr_token_g_topic` is the conditional distribution of tokens given topics.  `lda_pr_token` is the marginal density of tokens.

Since these structures are Pandas Series/DataFrames, you can access them with the usual methods.

```python
# The joint density P(token, topic)
lda.prob_token_topic()

# The conditional density restricted to one token
# P(token=kennedy | topic=topic_0)
lda.prob_token_topic(token='kennedy', c_topic='topic_0')

# P(token=kennedy | topic in [topic_0, topic_3])
lda.prob_token_topic(token='kennedy', c_topic=['topic_0', 'topic_3'])

# P(token | topic='topic_0')
lda.prob_token_topic(c_topic='topic_0')

# DataFrame with column k = P(token | topic=topic_k)
lda.pr_token_g_topic

# Similar structures available for doc/topic.
```

In addition, the `doc_freq` and `token_score` (and anything else that is in `sff.to_frame()` is accessible in `lda.sfile_frame`.


Contribute!
-----------
* Submit feature requests or bug reports by opening an [issue][issue]
* There are many ways to analyze topics...some should be built into `LDAResults`.  If you have ideas, let us know.
* Optimize slow spots and submit a pull request.  In particular, `SparseFormatter._parse_feature_str` could be re-written in Cython or Numba.




[vwinput]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
[dspyrepo]: https://github.com/columbia-applied-data-science/dspy
[vwlda]: https://github.com/JohnLangford/vowpal_wabbit/wiki/lda.pdf
[vwtricks]: www.slideshare.net/jakehofman/technical-tricks-of-vowpal-wabbit‎
[hashing]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Feature-Hashing-and-Extraction
[spot]: http://en.wikipedia.org/wiki/Single_Point_of_Truth
[issue]: https://github.com/columbia-applied-data-science/dspy/issues
