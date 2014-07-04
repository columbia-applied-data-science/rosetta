Rosetta
====

Tools for data science with a focus on text processing.

* Focuses on "medium data", i.e. data too big to fit into memory but too small to necessitate the use of a cluster.
* Integrates with existing scientific Python stack as well as select outside tools.

Examples
--------

* See the `examples/` directory.  
* The [docs](http://pythonhosted.org/rosetta/#examples) contain plots of example output.


Packages
--------

### `cmdutils` 
* Unix-like command line utilities.  Filters (read from stdin/write to stdout) for files.
* Focus on stream processing and csv files.

### `parallel` 
* Wrappers for Python multiprocessing that add ease of use
* Memory-friendly multiprocessing

### `text`
* Stream text from disk to formats used in common ML processes
* Write processed text to sparse formats
* Helpers for ML tools (e.g. Vowpal Wabbit, Gensim, etc...)
* Other general utilities

### `workflow`
* High-level wrappers that have helped with our workflow and provide additional examples of code use

### `modeling`
* General ML modeling utilities

Install
-------
Check out the master branch from the [rosettarepo][rosettarepo].  Then, (so long as you have `pip`).
    
    cd rosetta
    make
    make test
    
If you update the source, you can do

    make reinstall
    make test

The above `make` targets use `pip`, so you can of course do `pip uninstall` at any time.

Getting the source (above) is the preferred method since the code changes often, but if you don't use Git you can download a tagged release (tarball) [here](https://github.com/columbia-applied-data-science/rosetta/releases).  Then

    pip install rosetta-X.X.X.tar.gz

Development
-----------

### Code

You can get the latest sources with

    git clone git://github.com/columbia-applied-data-science/rosetta

### Contributing

Feel free to contribute a bug report or a request by opening an [issue](https://github.com/columbia-applied-data-science/rosetta/issues)

The preferred method to contribute is to fork and send a pull request.  Before doing this, read [CONTRIBUTING.md](CONTRIBUTING.md)

Dependencies
------------

* Major dependencies on *Pandas* and *numpy*.
* Minor dependencies on *Gensim* and *statsmodels*.
* Some examples need *scikit-learn*.
* Minor dependencies on *docx*
* Minor dependencies on the unix utilities *pdftotext* and *catdoc*

Testing
-------
From the base repo directory, `rosetta/`, you can run all tests with

    make test

Documentation
-------------

Documentation for releases is hosted at [pypi](http://pythonhosted.org/rosetta).  This does NOT auto-update.


History
-------
*Rosetta* refers to the [Rosetta Stone](http://en.wikipedia.org/wiki/Rosetta_Stone), the ancient Egyptian tablet discovered just over 200 years ago. The tablet contained fragmented text in three different languages and the uncovering of its meaning is considered an essential key to our understanding of Ancient Egyptian civilization. We would like this project to provide individuals the necessary tools to process and unearth insight in the ever-growing volumes of textual data of today.

[rosettarepo]: https://github.com/columbia-applied-data-science/rosetta
