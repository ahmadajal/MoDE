from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

sourcefiles = ['fastgd/bootstrap.pyx', 'fastgd/fastgd_cython.pyx', 'fastgd/fastgd_base.pyx','fastgd/fastgd_faster.pyx']

extensions = cythonize(Extension(
            name="fastgd.bootstrap",
            sources = sourcefiles,
            include_dirs=[numpy.get_include()]
    ))


kwargs = {
      'name':'fastgd',
      'packages':find_packages(),
      'ext_modules':  extensions,
}


setup(**kwargs)