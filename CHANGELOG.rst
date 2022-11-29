Changelog
=========

0.0.4 (2021-02-14)
------------------

* First release on PyPI.

0.0.5 (2021-02-15)
------------------

* Extra data set
* Dictionary scatter

0.0.6 (2021-02-15)
------------------

* Minor tweaks (extra passthroughs)

0.0.7 (2021-02-15)
------------------

* Minor tweaks
* Fixes a few bad links in README
* Updates the changelog

0.0.8 (2021-02-17)
------------------

* Adds data frame overviews


0.0.9 (2021-02-17)
------------------

* Allows specification of plot types in 2d

0.0.10 (2021-02-19)
-------------------

* Cumulative histograms

0.0.11 (2021-02-19)
-------------------

* Histogram normalization

0.0.12 (2021-02-25)
-------------------

* Heatmaps

0.0.13 (2021-02-26)
-------------------

* Feature engineering code generator

0.0.14 (2021-02-26)
-------------------

* Adds print template

0.0.15 (2021-03-14)
-------------------

* Adds requirements

0.0.16 (2021-03-14)
-------------------

* Fixes histogram color passthrough

0.0.17 (2021-03-14)
-------------------

* Adds internal face color for histogram (probably extend to others later)

0.0.18 (2021-03-28)
-------------------

* Adds gaussian demo data
* Histagram alpha parameter

0.0.19 (2021-04-09)
-------------------

* Adds rolling mean and median to scatter function

0.0.20 (2021-04-09)
-------------------

* Fixes mean / median bugs

0.0.21 (2021-04-09)
-------------------

* Fixes mean / median bugs

0.0.22 (2021-04-09)
-------------------

* Adds line of best fit

0.0.23 (2021-04-11)
-------------------

* Adds watermark

0.0.24 (2021-04-11)
-------------------

* Adds color

0.0.25 (2021-06-01)
-------------------

* Adds timeseries tooling and misc utils

0.0.26 (2021-06-01)
-------------------

* Repackage

0.0.27 (2021-11-26)
-------------------

* Include pydantic and loguru in dependencies

0.0.28 (2021-11-26)
-------------------

* bugfix

0.0.29 (2021-11-28)
-------------------

* tweak naming and add high dimension generalisation

0.0.30 (2021-11-29)
-------------------

* ability to invert y-axis on heatmaps

0.0.31 (2021-11-29)
-------------------

* deprecate univariate sequences

0.0.32 (2021-11-29)
-------------------

* method hint

0.0.33 (2021-11-29)
-------------------

* drop stray logger

0.0.34 (2021-11-29)
-------------------

* fix: sorting

0.0.35 (2021-12-04)
-------------------

* fix: vector ordering

0.0.36 (2021-12-08)
-------------------

* seasonal decomposition of vector sequence

0.0.37 (2021-12-12)
-------------------

* Singular vector decomposition infosurface

0.0.38 (2021-12-13)
-------------------

* fix: frequency decomposition label and xlims

0.0.39 (2021-12-27)
-------------------

* refactor: everything

0.0.40 (2021-12-28)
-------------------

* extra configurability, colors, etc

0.0.41 (2021-12-28)
-------------------

* fix: boolean grid parameter

0.0.42 (2021-12-31)
-------------------

* tweak: infer labels when scattering dataframes

0.0.43 (2022-01-01)
-------------------

* tweak: no legend on viz_x_y(dataframe, x string, y string)

0.0.44 (2022-01-18)
-------------------

* chore: fill ratio and vectorized machine time

0.0.45 (2022-01-20)
-------------------

* chore: add input interpreter in front of histogram functionality

0.0.46 (2022-01-21)
-------------------

* chore: attach correlation matrix wrapper

0.0.47 (2022-01-21)
-------------------

* tweak: kwargs > style > Style()

0.0.48 (2022-01-23)
-------------------

* chore: surfaces, svd container, and feature selection / aggregation

0.0.49 (2022-01-30)
-------------------

* chore: rework back down to python 3.8 compatibility

0.0.50 (2022-01-30)
-------------------

* chore: vectors from text

0.0.51 (2022-01-31)
-------------------

* chore: set >=3.8 in setup.py

0.0.52 (2022-01-31)
-------------------

* fix: module imports

0.0.53 (2022-01-31)
-------------------

* fix: attach extraction code to vectors module

0.0.54 (2022-01-06)
-------------------

* chore: automatic parsing

0.0.55 (2022-01-06)
-------------------

* chore: add progress bar

0.0.56 (2022-01-06)
-------------------

* chore: progress bar where I meant to put it in 0.0.55

0.0.57 (2022-03-10)
-------------------

* chore: optionally parallel sliding windows

0.0.58 (2022-03-10)
-------------------

* chore: slice out dataframes from vector sequence

0.0.59 (2022-03-16)
-------------------

* chore: parsing optimizations

0.0.60 (2022-03-17)
-------------------

* tweak: plot arg handling

0.0.61 (2022-03-17)
-------------------

* tweak: better type handling


0.0.62 (2022-03-31)
-------------------

* tweak: more efficient parsing with tokens

0.0.63 (2022-04-01)
-------------------

* tweak: kwargs passthrough

0.0.64 (2022-04-01)
-------------------

* fix: vector multiset viz y label

0.0.65 (2022-04-01)
-------------------

* chore: vector sequence basis manipulation

0.0.66 (2022-04-05)
-------------------

* fix: misc tweaks for kwargs and type robustness

0.0.67 (2022-04-05)
-------------------

* refactor: extraction functions

0.0.68 (2022-04-05)
-------------------

* tweak: rename extraction functions

0.0.69 (2022-04-13)
-------------------

* refactor: faster data reshaping (parallelize pandas method)

0.0.70 (2022-04-13)
-------------------

* tweak: Timeseries alias for VectorSequence

0.0.71 (2022-04-17)
-------------------

* tweak: attach hist2d to VectorMultiset

0.0.72 (2022-04-18)
-------------------

* chore: cast vector data to numeric, prevent nested multiprocessing

0.0.73 (2022-04-18)
-------------------

* chore: combinatorics

0.0.74 (2022-04-22)
-------------------

* tweak: better input handling and escapes

0.0.77 (2022-06-09)
-------------------

* tweak: wrap profiling methods

0.0.78 (2022-06-11)
-------------------

* tweak: slice method kwargs and no inplace return

0.0.81 (2022-06-15)
-------------------

* tweak: wrap FLUSS

0.0.82 (2022-07-11)
-------------------

* chore: memory and annotation

0.0.83 (2022-08-04)
-------------------

* chore: match facecolor for figure and axes

0.0.84 (2022-08-07)
-------------------

* chore: human-readable x-axis for scatter plots with time
* chore: use pandas builtin for datetime to timestamp conversion (much faster)

0.0.85 (2022-08-08)
-------------------

* chore: better config parameter handling

0.0.86 (2022-08-10)
-------------------

* chore: VectorSequence split

0.0.87 (2022-08-22)
-------------------

* refactor: batch evaluation. NB: this may change behavior!

0.0.88 (2022-08-23)
-------------------

* tweak: additional batch evaluation logging

0.0.89 (2022-09-15)
-------------------

* tweak: DataFrame method passthrough

0.0.90 (2022-09-15)
-------------------

* tweak: dictionary styling

0.0.91 (2022-11-29)
-------------------

* tweak: flag for Mac OS

0.0.92 (2022-11-29)
-------------------

* tweak: deprecate some matrixprofile methods due to end of support for the underlying library
