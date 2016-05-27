from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy

setup(
		    ext_modules=[
			            Extension("cwsabie_inner", ["cwsabie_inner.c"],
					                      include_dirs=[numpy.get_include()]),],
		    )
setup(
		    ext_modules = cythonize("cwsabie_inner.pyx")
		    )
