from distutils.core import setup

DISTNAME = 'DSpy'
LICENSE = 'BSD'
MAINTAINER = "The DsPy Development Team"
EMAIL = "ianlangmore@gmail.com"
URL = "https://github.com/columbia-applied-data-science/dspy"
DESCRIPTION = ""

setup(
    name=DISTNAME,
    version='0.1.0dev',
    packages=[
        'dspy',
        'dspy.cmd'],
    scripts=['dspy/cmd/cut.py'],
    license=LICENSE,
    url=URL,
    maintainer_email=EMAIL,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    long_description=open('README.md').read()
)
