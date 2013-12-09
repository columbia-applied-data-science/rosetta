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

### `cmd` 
* Unix-like command line utilities.  Filters (read from stdin/write to stdout) for files

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

Getting the source (above) is the preferred method since the code changes often, but if you don't use Git you can download a tagged release (tarball) [here](https://github.com/columbia-applied-data-science/rosetta/releases).  Then

    pip install rosetta-X.X.X.tar.gz

Development
-----------

### Code

You can check the latest sources with

    git clone git://github.com/columbia-applied-data-science/rosetta

### Contributing

Feel free to contribute a bug report or a request by opening an [issue](https://github.com/columbia-applied-data-science/rosetta/issues)

The preferred method to contribute is to fork and send a pull request.  Before doing this, read `CONTRIBUTING.md`

Dependencies
------------

* Major dependencies on Pandas and numpy.
* Minor dependencies on Gensim.
* Some examples need scikit-learn.

Testing
-------
From the base repo directory, `rosetta/`, you can run all tests with

    make test

Documentation
-------------

Documentation is hosted at [here](http://pythonhosted.org/rosetta).  This does NOT auto-update.  To make new docs:

    cd docs/
    make html

Note: you need to upload this documentation manually on pypi.   You can create the proper zipfile with `make zip-docs`.

Releases
--------
* Github: Rosetta releases are hosted [here](https://github.com/columbia-applied-data-science/rosetta/releases) and you can create new releases via "draft new release."
* PiPy: Rosetta releases are hosted [here](https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=rosetta). As a registered owner you can create a release by:

1. Run all tests with `make test`
2. Make new documentation (see the *Documenation* section).
3. Update the release version in setup.py.  We will use [semantic versioning](http://semver.org/).
4. Do `make release` to upload the installers to *PyPi*.
5. Manually upload the new doc zip-file to *PyPi*.

History
-------
*Rosetta* refers to the [Rosetta Stone](http://en.wikipedia.org/wiki/Rosetta_Stone), the ancient Egyptian tablet discovered just over 200 years ago. The tablet contained fragmented text in three different languages and the uncovering of its meaning is considered an essential key to our understanding of Ancient Egyptian civilization. We would like this project to provide individuals the necessary tools to process and unearth insight in the ever-growing volumes of textual data of today.

[rosettarepo]: https://github.com/columbia-applied-data-science/rosetta
