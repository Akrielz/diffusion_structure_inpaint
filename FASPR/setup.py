from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

faspr = Pybind11Extension(
    'faspr',
    [str(fn) for fn in Path('src').glob('*.cpp')],
    include_dirs=['include'],
    extra_compile_args=['-O3']
)

setup(
    name='faspr',
    version=0.1,
    ext_modules=[faspr],
    cmdclass={'build_ext': build_ext},
)

