import sys
import os
import numpy
from Cython.Distutils import build_ext
try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
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
    author_email='mcarras2@illinois.edu , sahandha@gmail.com',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("eif",
                 sources=["_eif.pyx", "eif.cxx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-Wcpp'],
                 language="c++")],
    scripts=[],
    py_modules=['eif_old', 'version'],
    packages=[],
    license='License.txt',
    description='Extended Isolation Forest for anomaly detection',
    long_description=read('README.md'),
    url='https://github.com/sahandha/eif',
    install_requires=["numpy"],
)
