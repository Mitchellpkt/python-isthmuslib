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

Misc utilities, mostly related to plotting

* Free software: GNU Lesser General Public License v3 (LGPLv3)

Installation
============

::

    pip install isthmuslib

You can also install the in-development version with::

    pip install https://github.com/mitchellpkt/python-isthmus/archive/master.zip


Documentation
=============


To use the project:

.. code-block:: python

    import isthmuslib
    isthmuslib.longest()


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
