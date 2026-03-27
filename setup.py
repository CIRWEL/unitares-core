import os
from setuptools import setup, find_packages

use_cython = os.environ.get("USE_CYTHON", "1") == "1"

if use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        "governance_core/*.py",
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,
        },
        exclude=["governance_core/__init__.py"],
    )
    package_data = {}
else:
    # Packaging mode: include pre-built .so files as package data
    ext_modules = []
    package_data = {"governance_core": ["*.so"]}

setup(
    ext_modules=ext_modules,
    packages=find_packages(include=["governance_core*"]),
    package_data=package_data,
)
