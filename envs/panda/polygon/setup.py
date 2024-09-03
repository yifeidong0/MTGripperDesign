from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext, Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "polygon",
        ["src/polygon.cpp"],
    ),
]

setup(
    name="polygon",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
