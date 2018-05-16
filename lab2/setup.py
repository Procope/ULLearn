from setuptools import setup, find_packages

try:
    import utils
    import skipgram
    import bsg
    import embedalign
except ImportError:
    raise ImportError("Error")

ext_modules = []

setup(
    name='ULL-lab2',
    author='Mario Giulianelli, Florian Mohnert',
    description='Learning word representations',
    packages=find_packages(),
    include_dirs=[],
    ext_modules=ext_modules,
)
