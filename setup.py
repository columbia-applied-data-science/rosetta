from distutils.core import setup

DISTNAME = 'DsPy'
LICENSE = 'BSD'
MAINTAINER = "The DsPy Development Team"
EMAIL = "dspy_team@gmail.com"
URL = "https://github.com/declassengine/declass"
DESCRIPTION = ""

setup(
    name=DISTNAME,
    version='0.1.0dev',
    packages=['dspy',],
    license=LICENSE,
    url=URL,
    maintainer_email=EMAIL,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    long_description=open('README.md').read()
)
