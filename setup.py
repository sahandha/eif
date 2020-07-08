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


if sys.platform.startswith("win"):
    compilation_flags = ['/std:c++latest']
else:
    compilation_flags = ['-std=c++11', '-Wcpp']

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('version.py').read())
setup(
    name='eif',
    version=__version__,
    author='Matias Carrasco Kind , Sahand Hariri, Seng Keat Yeoh',
    author_email='mcarras2@illinois.edu',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("eif",
                 sources=["_eif.pyx", "eif.cxx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=compilation_flags,
                 language="c++")],
    scripts=[],
    py_modules=['eif_old', 'version'],
    packages=[],
    license='License.txt',
    include_package_data=True,
    description='Extended Isolation Forest for anomaly detection',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/sahandha/eif',
    install_requires=["numpy", "cython"],
)
