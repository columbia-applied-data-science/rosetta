DSpy
====

Tools, wrappers, etc... for data science with a concentration on text processing

* Utilities to move data from one Data Structure to another
* Strong focus on stream processing of text
* Utilities to use packages outside the normal Python ecosystem
* Command line utilities
* Focus on "medium data", i.e. data too big to fit into memory but too small to necessitate the use of a cluster.
* The *DS* in DSpy clearly relates to *Data Science*.  However, it came first from *Data Structure* and the *Dead Sea*.  The tools concentrate on streaming text, and the dead sea scrolls are the most famous version of text in a stream (a lake actually...but just pretend and it's really cool).


Packages
--------

See the `examples/` directory for more details.

* `cmd` Command line utilities
* `modeling` Utilities to help common modeling tasks
* `parallel` Wrappers for Python multiprocessing that add much needed usability and allow for stream processing
* `text` Text processing
* `workflow` High-level wrappers that have helped with our workflow and provide additional examples of code use

Install
-------
Check out the dev branch or a tagged release from the [dspyrepo][dspyrepo].  Then (so long as you have `pip`).

    make
    make test

Development
-----------

### Code

You can check the latest sources with

    git clone git://github.com/columbia-applied-data-science/dspy

### Contributing

Feel free to contribute a bug report or a request by opening an [issue](https://github.com/columbia-applied-data-science/dspy/issues)

Before contributing code, read `CONTRIBUTING.md`

Dependencies
------------

Testing
-------
From the base repo directory, `dspy/`, you can run all tests with

    make test

[dspyrepo]: https://github.com/columbia-applied-data-science/dspy
