from distutils.core import setup
from Cython.Build import cythonize

modules = ["mod_hive_mem.pyx", "hive_mem.pyx"]
setup(
    ext_modules = cythonize(modules)
)
