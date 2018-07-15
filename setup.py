import sys
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
prjdir = os.path.dirname(__file__)

def read(filename):
    return open(os.path.join(prjdir, filename)).read()

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('version.py').read())
setup(
    name='eif',
    version=__version__,
    author='Matias Carrasco Kind , Sahand Hariri',
    author_email='mcarras2@illinois.edu', 'sahandha@gmail.com',
    scripts=[],
    py_modules=['eif','version'],
    packages=[],
    license='License.txt',
    description='Extended Isolation Forest for anomaly detection',
    long_description=read('README.md'),
    url='https://github.com/sahandha/eif',
    install_requires=["numpy"],
)
