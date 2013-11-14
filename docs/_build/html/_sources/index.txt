.. rosetta documentation master file, created by
   sphinx-quickstart on Thu Nov 14 11:27:45 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rosetta's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

All Modules and Classes
=======================


cmd
---

Unix-like command line utilities.  Filters (read from stdin/write to stdout) for files
Installation should put these in your path.  To see help, do

.. code-block:: none

   module_name.py -h

cut
~~~
.. automodule:: rosetta.cmd.cut

subsample
~~~~~~~~~
.. automodule:: rosetta.cmd.subsample

split
~~~~~
.. automodule:: rosetta.cmd.split

row_filter
~~~~~~~~~~
.. automodule:: rosetta.cmd.row_filter

files_to_vw
~~~~~~~~~~~
.. automodule:: rosetta.cmd.files_to_vw

join_csv
~~~~~~~~~~~
.. automodule:: rosetta.cmd.join_csv

concat_csv
~~~~~~~~~~~
.. automodule:: rosetta.cmd.concat_csv

parallel
--------

* Wrappers for Python multiprocessing that add ease of use
* Memory-friendly multiprocessing

parallel_easy
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rosetta.parallel.parallel_easy
   :members:

pandas_easy
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rosetta.parallel.pandas_easy
   :members:

text
----

Text-processing specific

* Stream text from disk to formats used in common ML processes
* Write processed text to sparse formats
* Helpers for ML tools (e.g. Vowpal Wabbit, Gensim, etc...)
* Other general utilities

filefilter
~~~~~~~~~~
.. automodule:: rosetta.text.filefilter
   :members:

streamers
~~~~~~~~~
.. automodule:: rosetta.text.streamers
   :members:

text_processors
~~~~~~~~~~~~~~~
.. automodule:: rosetta.text.text_processors
   :members:

nlp
~~~~
.. automodule:: rosetta.text.nlp
   :members:

vw_helpers
~~~~~~~~~~
.. automodule:: rosetta.text.vw_helpers
   :members:

gensim_helpers
~~~~~~~~~~~~~~
.. automodule:: rosetta.text.gensim_helpers
   :members:

modeling
--------

* General ML modeling utilities

eda
~~~~~~~~~~
.. automodule:: rosetta.modeling.eda
   :members:

prediction_plotter
~~~~~~~~~~~~~~~~~~
.. automodule:: rosetta.modeling.prediction_plotter
   :members:

var_create
~~~~~~~~~~
.. automodule:: rosetta.modeling.var_create
   :members:

fitting
~~~~~~~~~~
.. automodule:: rosetta.modeling.fitting
   :members:

categorical_fitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rosetta.modeling.categorical_fitter
   :members:

shared modules
--------------

Shared by other modules.

common
~~~~~~
.. automodule:: rosetta.common

common_math
~~~~~~~~~~~
.. automodule:: rosetta.common_math
