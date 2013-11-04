from distutils.core import setup

DISTNAME = 'DSpy'
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
    'row_filter.py', 'split.py', 'subsample.py']]

PACKAGES = [
    'dspy/' + name for name in 
    ['cmd', 'modeling', 'parallel', 'tests', 'text', 'workflow']]

setup(
    name=DISTNAME,
    version='0.1.0dev',
    packages=PACKAGES,
    scripts=SCRIPTS,
    license=LICENSE,
    url=URL,
    maintainer_email=EMAIL,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    long_description=open('README.md').read()
)
