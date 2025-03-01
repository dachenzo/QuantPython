from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fastfactor",
        ["fastfactor.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    )
]
#
setup(
    name="fastfactor",
    ext_modules=ext_modules
)