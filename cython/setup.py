from distutils.core import setup
from Cython.Build import cythonize

modules = ["mod_hive_mem.pyx", "hive_mem.pyx", "mod_hive_mem_funcs.pyx"]
setup(
    ext_modules = cythonize(modules)
)
