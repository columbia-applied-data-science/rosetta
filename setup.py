from distutils.core import setup

DISTNAME = 'rosetta'
LICENSE = 'BSD'
MAINTAINER = "The Rosetta Development Team"
EMAIL = "ianlangmore@gmail.com"
URL = "https://github.com/columbia-applied-data-science/rosetta"
DESCRIPTION = (
    "Tools, wrappers, etc... for data science with a concentration on text "
    "processing")

SCRIPTS = [
    'rosetta/cmdutils/' + name for name in
    ['concat_csv.py', 'cut.py', 'join_csv.py',
    'row_filter.py', 'split.py', 'subsample.py',
    'files_to_vw.py', 'filter_sfile.py', 'groupby_reduce.py']]

PACKAGES =  ['rosetta'] + [
    'rosetta.' + name for name in
    ['cmdutils', 'modeling', 'parallel', 'text', 'workflow']]

setup(
    name=DISTNAME,
    version='0.3',
#    py_modules=PY_MODULES,
    packages=PACKAGES,
    scripts=SCRIPTS,
    license=LICENSE,
    url=URL,
    maintainer_email=EMAIL,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    long_description=open('README.md').read()
)
