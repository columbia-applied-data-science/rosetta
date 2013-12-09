
Contributing code
=================

Please be sure to read this carefully to make
the code review process go as smoothly as possible and maximize the
likelihood of your contribution being merged.**

How to contribute
-----------------

The preferred way to contribute to rosetta is to fork the 
[project repository](https://github.com/columbia-applied-data-science/rosetta/) on
GitHub:

1. Fork the [project repository](https://github.com/columbia-applied-data-science/rosetta/):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/rosetta.git

3. Create a branch to hold your changes:

          $ git checkout -b my-feature

   and start making changes. Never work in the ``master`` branch!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-feature

Finally, go to the web page of the your fork of the rosetta repo,
and click 'Pull request' to send your changes to the maintainers for
review. request. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All public methods should have informative docstrings complying with
   the [numpy standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard).

-  When adding additional functionality, provide at least one
   example script in the `examples/` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in rosetta.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

You can also check for common programming errors with

    make code-analysis

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
