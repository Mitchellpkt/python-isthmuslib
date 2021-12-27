========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |requires|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |requires| image:: https://requires.io/github/mitchellpkt/python-isthmus/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/mitchellpkt/python-isthmus/requirements/?branch=master

.. |version| image:: https://img.shields.io/pypi/v/isthmuslib.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/isthmuslib

.. |wheel| image:: https://img.shields.io/pypi/wheel/isthmuslib.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/isthmuslib

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/isthmuslib.svg
    :alt: Supported versions
    :target: https://pypi.org/project/isthmuslib

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/isthmuslib.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/isthmuslib

.. |commits-since| image:: https://img.shields.io/github/commits-since/mitchellpkt/python-isthmus/v0.0.4.svg
    :alt: Commits since latest release
    :target: https://github.com/mitchellpkt/python-isthmus/compare/v0.0.4...master



.. end-badges

Convenience utils for plotting, styling, and manipulating high-dimensional vectors.

* Analyses and plotting methods are one line to call, and produce consistently-formatted publication-ready plots.
* Enables rapid exploratory data analysis (EDA) and prototyping, perfect for taking a quick peek at data or making a quick figure to stash in the lab book (with labels and titles automatically included). See `examples here <https://github.com/Mitchellpkt/python-isthmuslib/blob/main/demo.ipynb>`_.
* Designed for easy drop-in use for other projects, whether using internally to the code or for clean notebooks. Import isthmuslib to avoid writing many lines of plotting code when it would distract or detract from the main focus of your project.
* The visual and text configuration objects (`Style` and `Rosetta`, respectively) can be directly attached to a given data set, so you can "set it and forget it" at instantiation. All subsequent outputs will automatically have matching colors, sizes, labels, etc.
* The `VectorSequence` object is designed for handling, plotting, and manipulating timeseries-like high-dimensional vectors. Its functionality includes: dimensionality reduction via singular vealue decomposition, seasonal (e.g. weekly, monthly, ...) timeseries decomposition, infosurface generation, and more.
* Uses industry standard libraries (pyplot, numpy, seaborn, pandas, etc) under the hood, and exposes their underlying functionality through the wrappers.

Free software for personal or academic use: **GNU Lesser General Public License v3 (LGPLv3).** Contact licensing@mitchellpkt.com for commercial applications.

Installation
============

::

    pip install isthmuslib

Documentation
=============


To use the project:

.. code-block:: python

    import isthmuslib


Demo
=============
See a light demo here: https://github.com/Mitchellpkt/python-isthmuslib/blob/main/demo.ipynb
