#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


long_description_raw: str = "%s\n%s" % (
    re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
    re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
)
split_string: str = "Demo one-liners"

setup(
    name="isthmuslib",
    version="0.0.107",
    license="MIT",
    description="Tooling for rapid data exploration, timeseries analysis, log extraction & visualization, etc",
    long_description=long_description_raw.split(split_string)[0],
    author="Isthmus (Mitchell P. Krawiec-Thayer)",
    author_email="isthmuslib@mitchellpkt.com",
    url="https://github.com/mitchellpkt/python-isthmuslib",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
    ],
    project_urls={
        "Changelog": "https://github.com/mitchellpkt/python-isthmuslib/blob/master/CHANGELOG.rst",
        "Issue Tracker": "https://github.com/mitchellpkt/python-isthmuslib/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "loguru",
        "pydantic",
        "statsmodels",
        "pytz",
        "pyarrow",
        "tqdm",
        # Temporarily(?) deprecated pending python 3.11 support:
        # "scikit-learn",
        # "ipywidgets",
        # "stumpy",
    ],
    extras_require={},
    entry_points={
        "console_scripts": [
            "isthmuslib = isthmuslib.cli:main",
        ]
    },
)
