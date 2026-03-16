import os
from setuptools import setup, find_packages

if os.environ.get("USE_CYTHON", "1") == "1":
    from Cython.Build import cythonize
    ext_modules = cythonize(
        "governance_core/*.py",
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,
        },
        exclude=["governance_core/__init__.py"],
    )
else:
    ext_modules = []

setup(
    ext_modules=ext_modules,
    packages=find_packages(include=["governance_core*"]),
)
