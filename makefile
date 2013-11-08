# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
UNITTEST ?= unittest
CTAGS ?= ctags

TESTDIR=dspy/tests

all: install test


# First install with pip
install: clean
	$(PYTHON) setup.py sdist
	pip install dist/*

# Reinstall with pip
reinstall: clean
	pip uninstall dspy
	$(PYTHON) setup.py sdist
	pip install dist/*

#install:
#	$(PYTHON) setup.py install

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean --all
	rm -rf dist

tests: test  # Common misname...

test:
	$(PYTHON) -m $(UNITTEST) discover -s $(TESTDIR) -v

test-text:
	$(PYTHON) -m $(UNITTEST) discover -s $(TESTDIR) -p '*text*' -v

test-parallel:
	$(PYTHON) -m $(UNITTEST) discover -s $(TESTDIR) -p '*parallel*' -v

test-common:
	$(PYTHON) -m $(UNITTEST) discover -s $(TESTDIR) -p '*common*' -v

test-cmd:
	$(PYTHON) -m $(UNITTEST) discover -s $(TESTDIR) -p '*cmd*' -v

trailing-spaces:
	find dspy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

code-analysis:
	flake8 dspy | grep -v __init__ | grep -v external
	pylint -E -i y dspy/ -d E1103,E0611,E1101

