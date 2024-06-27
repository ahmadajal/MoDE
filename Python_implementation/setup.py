"""Setup."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# sourcefiles = [
#     "src/MoDE_embeddings/fastgd/bootstrap.c",
#     "src/MoDE_embeddings/fastgd/fastgd_base.c",
#     "src/MoDE_embeddings/fastgd/fastgd_faster.c",
#     "src/MoDE_embeddings/fastgd/fastgd_cython.c"
# ]
exts = [
    Extension(
        name="MoDE_embeddings.fastgd.bootstrap",
        sources=["src/MoDE_embeddings/fastgd/bootstrap.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="MoDE_embeddings.fastgd.fastgd_base",
        sources=["src/MoDE_embeddings/fastgd/fastgd_base.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="MoDE_embeddings.fastgd.fastgd_faster",
        sources=["src/MoDE_embeddings/fastgd/fastgd_faster.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="MoDE_embeddings.fastgd.fastgd_cython",
        sources=["src/MoDE_embeddings/fastgd/fastgd_cython.c"],
        include_dirs=[numpy.get_include()],
    ),
]

c_exts = cythonize(exts)

setup(
    name="MoDE_embeddings",
    version="0.1.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"MoDE_embeddings.fastgd": ["*.pyx", "*.pxd", "*.c"]},
    ext_modules=c_exts,
    include_package_data=True,
)
