from typing import List, Tuple, Dict, Any, Union
from matplotlib import pyplot as plt
import pandas as pd
from .config import Style
from .plotting import visualize_x_y, visualize_1d_distribution
import numpy as np


def data_diffs(array: Any) -> List[Union[float, int]]:
    """ Tiny helper function that returns the differences

    :param array: the array (in any form)
    :return: spacings between the elements
    """
    return [x[1] - x[0] for x in zip(array, array[1:])]


def is_uniform(array: Any) -> bool:
    """ Tiny helper function, checks if an array appears to be a uniformly spaced list of numbers

    :param array: the array (in any form)
    :return: True if the elements are uniformly spaced
    """
    return len(set(data_diffs(array))) == 1


def any_missing_data(array: Any) -> bool:
    """ Tiny helper function, checks whether a data array appears to have any missing data, wraps a pandas util

    :param array: the array (in any form)
    :return: True if any data points appear to be missing
    """
    if isinstance(array, pd.DataFrame):
        return array.isnull().values.any()
    else:
        return pd.DataFrame({"_": array}).isnull().values.any()


def basis_quality_checks(array: Any) -> Tuple[bool, str]:
    """ Checks data quality and returns both a True/False flag and a string with an explanation

    :param array: the array (in any form)
    :return: True if OK, and a string with explanation either way
    """
    if ~any_missing_data(array):
        summary_string: str = '[OK] COMPLETENESS: no apparent missing data\n'
    else:
        summary_string: str = '[WARNING] COMPLETENESS: appears to be missing data'
        return False, f"{summary_string}\n (note: cannot check uniformity with missing data)"

    if is_uniform(array):
        summary_string += f'[OK] UNIFORMITY: appears to be uniform with spacing: {data_diffs(array)[0]}\n'
        return True, summary_string
    else:
        summary_string += f'[WARNING] UNIFORMITY: multiple spacings: {data_diffs(array)}\n'
        return False, summary_string


def passes_basis_quality_checks(array: Any) -> bool:
    """ Super thin wrapper around basis_quality_checks that drops the explanation string

    :param array: the array (in any form)
    :return: True if OK
    """
    result, _ = basis_quality_checks(array)
    return result


def fill_ratio(array: Any) -> float:
    """ Helper function that identifies the ratio of missing values in a uniformly-spaced sequence of values.
        Attempts to infer spacing from median of diffs. If used on non-uniform data the result is not meaningful.

    Use example: [10, 20, 40, 50] has 0.8 fill ratio because it is missing the 30 to complete the set

    :param array: Any 1D vector, like a list of integers or floats, numpy array, etc
    :return: fractional amount of the array
    """
    ordered: Any = sorted(array)
    theoretical_count = 1 + (max(array) - min(array)) / np.nanmedian([x[1] - x[0] for x in zip(ordered, ordered[1:])])
    return len(array) / theoretical_count


def basis_quality_plots(array: List[float], style: Style = None, which_plots: List[str] = None) -> List[plt.Figure]:
    """ Generates some standard plots for checking completeness and uniformity of an array

    :param array: the array (in any form)
    :param style: optional Style configuration object
    :param which_plots: which plots to make, choices include:
        - 'missing_data_scatter'
        - 'interval_histogram'
        - 'interval_scatter'
    :return:
    """
    # Style and init
    if not style:
        style: Style = Style()
    mappings: Dict[bool, Tuple[str, Any]] = {True: ("[OK]", style.good_color), False: ('[WARNING]', style.bad_color)}
    h: List[plt.Figure] = []

    # Scatter plot of missing data
    if (not which_plots) or ('missing_data_scatter' in which_plots):
        if not any_missing_data(array):
            title: str = f"{mappings[True][0]} no missing data to plot"
        else:
            title: str = f"{mappings[False][0]} missing data"
        title += f"\n(want all 0's, on the {style.good_color} bottom line)"
        vals_y = [int((x is None) or np.isnan(x)) for x in array]
        f: plt.Figure = visualize_x_y(range(len(vals_y)), vals_y, xlabel='index', ylabel='is_missing (1=True)',
                                      types='scatter',
                                      style=style.override({'color': 'k', 'markersize': 2 * style.markersize}),
                                      title=title)
        plt.axhline(y=0, linewidth=2, linestyle=':', c=style.good_color)
        plt.axhline(y=1, linewidth=2, linestyle=':', c=style.bad_color)
        plt.ylim([-0.5, 1.5])
        xlims: Tuple[float, float] = plt.xlim()
        plt.text(xlims[0] + 0.02 * (xlims[1] - xlims[0]), 0.05, "0 -> good", color=style.good_color,
                 size=style.label_fontsize)
        plt.text(xlims[0] + 0.02 * (xlims[1] - xlims[0]), 1.05, "1 = NaN or None -> bad", color=style.bad_color,
                 size=style.label_fontsize)
        h.append(f)

    # Histogram of data spacing
    if (not which_plots) or ('intervals_histogram' in which_plots):
        if not any_missing_data(array):
            vals: List[Any] = sorted(data_diffs(array))
            title: str = f'{mappings[is_uniform(array)][0]} data intervals distribution \n(want only 1 value)'
            color: Any = mappings[is_uniform(array)][1]
        else:
            vals: List[Any] = [[]]
            title: str = f'{mappings[False][0]} Cannot make intervals hist with missing data! Plot is blank.'
            color: Any = mappings[False][1]
        h.append(visualize_1d_distribution(vals, xlabel='difference between elements', ylabel='counts', title=title,
                                           style=style.override({'color': color})))

    # Scatter plot of data spacing
    if (not which_plots) or ('intervals_scatter' in which_plots):
        if not any_missing_data(array):
            sorted_array: Any = sorted(array)
            vals_x: List[Any] = sorted_array[1:]
            vals_y: List[Any] = data_diffs(sorted_array)
            title: str = f'{mappings[is_uniform(array)][0]} data intervals over time\n(want 1 horizontal line)'
            color: Any = mappings[is_uniform(array)][1]
        else:
            vals_x = vals_y = [[]]
            title: str = f'{mappings[False][0]} Cannot make diffs plot with missing data! Plot is blank.'
            color: Any = mappings[False][1]
        h.append(visualize_x_y(vals_x, vals_y, types='line', xlabel='basis', ylabel='difference between elements',
                               title=title, style=style.override({'color': color})))

    return h
