Rosetta
====

Tools for data science with a focus on text processing.

* Focuses on "medium data", i.e. data too big to fit into memory but too small to necessitate the use of a cluster.
* Integrates with existing scientific Python stack as well as select outside tools.

Packages
--------

See the `examples/` directory for more details.

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
Check out the dev branch or a tagged release from the [rosettarepo][rosettarepo].  Then (so long as you have `pip`).

    make
    make test

Development
-----------

### Code

You can check the latest sources with

    git clone git://github.com/columbia-applied-data-science/rosetta

### Contributing

Feel free to contribute a bug report or a request by opening an [issue](https://github.com/columbia-applied-data-science/rosetta/issues)

Before contributing code, read `CONTRIBUTING.md`

Dependencies
------------

Testing
-------
From the base repo directory, `rosetta/`, you can run all tests with

    make test

History
-------
The *DS* in Rosetta clearly relates to *Data Science*.  However, it came first from *Data Structure* and the *Dead Sea*.  The tools concentrate on streaming text, and the dead sea scrolls are the most famous version of text in a stream (a lake actually...but just pretend and it's really cool).

[rosettarepo]: https://github.com/columbia-applied-data-science/rosetta
