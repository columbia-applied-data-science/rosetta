
Contributing code
=================

Please be sure to read this carefully to make
the code review process go as smoothly as possible and maximize the
likelihood of your contribution being merged.**

Data
----

* Don't commit data to the repo
* Read README_data.md for an example of how to manage data in a medium size project


How to contribute
-----------------

The preferred way to contribute to dspy is to fork the 
[project repository](https://github.com/columbia-applied-data-science/dspy/) on
GitHub:

1. Fork the [project repository](https://github.com/columbia-applied-data-science/dspy/):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/dspy.git

3. Create a branch to hold your changes:

          $ git checkout -b my-feature

   and start making changes. Never work in the ``master`` branch!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-feature

Finally, go to the web page of the your fork of the dspy repo,
and click 'Pull request' to send your changes to the maintainers for
review. request. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All public methods should have informative docstrings complying with
   the [numpy standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard).

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in scikit-learn.

-  At least one paragraph of narrative documentation with links to
````   references in the literature (with PDF links when possible) and
   the example.

The documentation should also include expected time and space
complexity of the algorithm and scalability, e.g. "this algorithm
can scale to a large number of samples > 100000, but does not
scale in dimensionality: n_features is expected to be lower than
100".

You can also check for common programming errors with the following
tools:

-  Code with good unittest coverage (at least 80%), check with:

          $ pip install nose coverage
          $ nosetests --with-coverage path/to/tests_for_package

-  No pyflakes warnings, check with:

           $ pip install pyflakes
           $ pyflakes path/to/module.py

-  No PEP8 warnings, check with:

           $ pip install pep8
           $ pep8 path/to/module.py

-  AutoPEP8 can help you fix some of the easy redundant errors:

           $ pip install autopep8
           $ autopep8 path/to/pep8.py

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).
