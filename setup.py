from distutils.core import setup

DISTNAME = 'dspy'
LICENSE = 'BSD'
MAINTAINER = "The DsPy Development Team"
EMAIL = "ianlangmore@gmail.com"
URL = "https://github.com/columbia-applied-data-science/dspy"
DESCRIPTION = (
    "Tools, wrappers, etc... for data science with a concentration on text "
    "processing")

SCRIPTS = [
    'dspy/cmd/' + name for name in 
    ['concat_csv.py', 'cut.py', 'join_csv.py',
    'row_filter.py', 'split.py', 'subsample.py',
    'files_to_vw.py', 'filter_sfile.py']]

PACKAGES =  ['dspy'] + [
    'dspy.' + name for name in 
    ['cmd', 'modeling', 'parallel', 'text', 'workflow']]


#PY_MODULES =  ['dspy'] + [
#    'dspy/' + name for name in 
#    ['cmd', 'modeling', 'parallel', 'text', 'workflow']]


setup(
    name=DISTNAME,
    version='0.1.0dev',
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
