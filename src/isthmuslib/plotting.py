import datetime
from typing import List, Any, Union, Tuple, Callable, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from .config import Style
from .utils import (
    looks_like_list_of_lists,
    margin_calc,
    to_list_if_other_array,
    make_dict,
)


##################
# Helper functions
##################


def plot_best_fit_line(
    x_data: Any,
    y_data: Any,
    degree: int = None,
    color: str = "k",
    style: Style = None,
    **kwargs,
) -> None:
    """Adds the line of best fit to a plot

    :param x_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param y_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param degree: degree oft the polynomial to fit
    :param color: color for the line
    :param style: optional style object (which contains a line width parameter)
    :param kwargs: additional keyword arguments for matplotlib.pyplot.plt
    :return: None
    """
    if kwargs.get("linewidth"):
        line_width = kwargs["linewidth"]
    else:
        if (not style) or (not style.linewidth):
            line_width: float = Style().linewidth
        else:
            line_width: float = style.linewidth
    if not degree:
        degree: int = 1
    z = np.polyfit(np.array(x_data), np.array(y_data), degree)
    plt.plot(
        x_data,
        [z[0] * x + z[1] for x in x_data],
        color=color,
        linewidth=line_width,
        **kwargs,
    )


def adjust_axes(
    log_axes: Union[str, List[str]] = "",
    style: Style = None,
    xlim: Any = None,
    ylim: Any = None,
    x_axis_human_tick_labels: bool = False,
    x_axis_formatter: str = None,
) -> None:
    """Helper function that adjusts the axes of a plot as specified

    :param log_axes: can look like: 'x' or 'y' or 'xy' // or look like: ['x'] or ['y'] or ['x', 'y']
    :param style: the Style object to apply
    :param xlim: bound for the x-axis
    :param ylim: bound for the y-axis
    :param x_axis_human_tick_labels: True if the x-axis labels should be displayed as human-readable
    :param x_axis_formatter: format string for the x-axis if human-readable
    :return: None
    """
    if not style:
        style = Style()
    if style.grid:
        plt.grid()
    if style.tight_axes:
        plt.autoscale(enable=True, axis="x", tight=True)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if "x" in log_axes:
        plt.xscale("log")
    if "y" in log_axes:
        plt.yscale("log")
    if x_axis_human_tick_labels:
        if x_axis_formatter is None:
            x_axis_formatter: str = "%Y-%m-%d"
        xfmt = matplotlib.dates.DateFormatter(x_axis_formatter)
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.xaxis.set_major_formatter(xfmt)


def apply_plot_labels(
    xlabel: str = "", ylabel: str = "", title: str = "", style: Style = None
) -> None:
    """Helper function to apply labels (xlabel, ylabel, title) according to the style guide (including translation)

    :param xlabel: text or rosetta key for the x-axis label
    :param ylabel: text or rosetta key for the y-axis label
    :param title: text or rosetta key for the title
    :param style: config Style object
    """
    if not style:
        style: Style = Style()
    plt.xlabel(
        style.rosetta.translate(xlabel, missing_response="return_input"),
        fontsize=style.label_fontsize,
    )
    plt.ylabel(
        style.rosetta.translate(ylabel, missing_response="return_input"),
        fontsize=style.label_fontsize,
    )
    plt.title(
        style.rosetta.translate(title, missing_response="return_input"),
        fontsize=style.title_fontsize,
    )


def apply_watermark(
    watermark_text: str, style: Style = None, use_default: bool = True, **kwargs
) -> None:
    """Helper function to apply watermark text to a plot based on config Style object parameters

    :param watermark_text: Text to display
    :param style: configuration object (Style)
    :type use_default: if True, then the config watermark_text will override the input argument watermark_text
    :param kwargs: additional keyword arguments for matplotlib.pyplot.text
    :return: None
    """

    if not watermark_text:
        if use_default:
            watermark_text: str = style.watermark_text
        else:
            return None

    if not style:
        style: Style = Style()

    # Accept either (x,y) placement OR a single scalar margin that will be used in both dimensions
    if isinstance(style.watermark_placement, (float, int)):
        position: Tuple[float, float] = (
            style.watermark_placement,
            style.watermark_placement,
        )
    elif len(style.watermark_placement) == 2:
        position: Tuple[float, float] = style.watermark_placement
    else:
        raise ValueError(f"Unable to interpret watermark {style.watermark_placement=}")

    ax = plt.gca()
    x_coordinate: float = margin_calc(position[0], plt.xlim(), ax.xaxis.get_scale())
    y_coordinate: float = margin_calc(position[1], plt.ylim(), ax.yaxis.get_scale())
    plt.text(
        x_coordinate,
        y_coordinate,
        watermark_text,
        fontsize=style.watermark_fontsize,
        c=style.watermark_color,
        **kwargs,
    )


##################
# Core visualize
# functionality
##################


def visualize_1d_distribution(
    data: Any,
    xlabel: str = "",
    ylabel: str = "counts",
    title: str = "",
    log_axes: str = "",
    style: Style = None,
    watermark: str = "",
    multi: bool = None,
    legend_strings: Union[Tuple[str], List[str]] = None,
    xlim: Any = None,
    ylim: Any = None,
    **kwargs,
) -> plt.Figure:
    """Core function for visualizing 1-dimensional distribution(s)

    :param data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param xlabel: label text for the x-axis
    :param ylabel: label text for the y-axis
    :param title: title text
    :param log_axes: which axes should be on a log scale, e.g. 'x' or ['x','y'] or 'xy'
    :param style: configuration object (optional)
    :param watermark: watermark text
    :param multi: flag to specify whether multiple traces are desired, if the automatic inference is incorrect
    :param legend_strings: override legend strings
    :param xlim: optional bound for the x-axis
    :param ylim: optional bound for the y-axis
    :param kwargs: additional keyword arguments for matplotlib.pyplot.hist()
    :return: figure handle for the plot
    """
    # Set style. Overrides: kwargs > style input > Style() defaults
    config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})
    kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k not in config.dict()}

    # If not specified, try to ascertain whether one or multiple data sets are being provided
    if multi or ((multi is None) and looks_like_list_of_lists(data)):
        config.color = None
        kwargs.setdefault("alpha", config.multi_hist_alpha)
    else:
        data: List[Any] = [data]

    # Plot the data and (and best fits if applicable)
    figure_handle: plt.Figure = plt.figure(
        facecolor=config.facecolor, figsize=config.figsize
    )
    plt.axes().set_prop_cycle(config.cycler)
    for data_set in data:
        # plotting on log x-axis requires special pre-treatment (log-distributed bin edges)
        if "x" in log_axes:
            if bins := kwargs.get("bins"):
                if not isinstance(bins, (int, float)):
                    hist_bins = bins
                else:
                    hist_bins = np.logspace(
                        np.log10(min(data_set)), np.log10(max(data_set)), bins
                    )
            else:
                hist_bins = np.logspace(
                    np.log10(min(data_set)),
                    np.log10(max(data_set)),
                    config.histogram_bins,
                )
        else:
            hist_bins = kwargs.get("bins")
        plt.hist(
            data_set,
            color=config.color,
            bins=hist_bins,
            **{k: v for k, v in kwargs.items() if k != "bins"},
        )

    # Make sure that the axes facecolor matches the figure facecolor
    plt.gca().set(facecolor=config.facecolor)

    # Adjust view & style where applicable
    xlabel_buffer: str = config.translate(xlabel, missing_response="return_input")
    ylabel_translated: str = config.translate(ylabel, missing_response="return_input")
    if kwargs.get("cumulative"):
        ylabel_buffer: str = f"cumulative {ylabel_translated}"
    else:
        ylabel_buffer: str = ylabel_translated
    if kwargs.get("density"):
        ylabel_buffer += " (density)"
    apply_plot_labels(
        xlabel=xlabel_buffer, ylabel=ylabel_buffer, title=title, style=config
    )
    if legend_strings:
        plt.legend(legend_strings, fontsize=config.legend_fontsize)

    adjust_axes(log_axes=log_axes, style=config, xlim=xlim, ylim=ylim)
    apply_watermark(watermark, style=config)
    return figure_handle


def visualize_x_y(
    x_data: Any,
    y_data: Any,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    cumulative: str = "",
    log_axes: Union[str, List[str]] = "",
    types: Union[str, List[str]] = "scatter",
    style: Style = None,
    watermark: str = "",
    multi: bool = None,
    legend_strings: Union[Tuple[str], List[str]] = None,
    xlim: Any = None,
    ylim: Any = None,
    plot_best_fit: Union[bool, int] = False,
    rolling_mean_width: int = None,
    rolling_median_width: int = None,
    show_colorbar: bool = False,
    log_norm_colors: bool = False,
    colorbar_label: str = None,
    x_axis_human_tick_labels: bool = False,
    x_axis_formatter: str = "%Y-%m-%d",
    **kwargs,
) -> plt.Figure:
    """Core function for visualizing 2-dimensional data sets

    :param x_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param y_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param xlabel: label text for the x-axis
    :param ylabel: label text for the y-axis
    :param title: title text
    :param cumulative: which axes to make cumulative, e.g. 'x' or ['x','y'] or 'xy'
    :param log_axes: which axes should be on a log scale, e.g. 'x' or ['x','y'] or 'xy'
    :param types: which types of plot to make, currently supports: 'scatter' and 'plot'
    :param style: configuration object (optional)
    :param watermark: watermark text
    :param multi: flag to specify whether multiple traces are desired, if the automatic inference is incorrect
    :param legend_strings: override legend strings
    :param plot_best_fit: whether to plot the best fit lines (specify degree, or pass True for degree 1)
    :param xlim: optional bound for the x-axis
    :param ylim: optional bound for the y-axis
    :param kwargs: additional keyword arguments for matplotlib.pyplot.scatter()
    :param rolling_median_width: window width for rolling average taken by df.y.rolling(rolling_mean_width).mean()
    :param rolling_mean_width: window width for rolling average taken by df.y.rolling(rolling_median_width).median()
    :param colorbar_label: optional label for the colorbar
    :param log_norm_colors: set to True to normalize the colorbar scale
    :param show_colorbar: set to True to show colorbar
    :param x_axis_human_tick_labels: set to True to convert numeric values along the x-axis to human-readable timestamps
    :param x_axis_formatter: format string for the x-axis if human-readable
    :return: figure handle for the plot
    """
    # Set style. Overrides: kwargs > style input > Style() defaults
    config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})
    kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k not in config.dict()}

    x_data: List[Any] = to_list_if_other_array(x_data)
    y_data: List[Any] = to_list_if_other_array(y_data)

    # If not specified, try to ascertain whether one or multiple data sets are being provided
    if multi or ((multi is None) and looks_like_list_of_lists(y_data)):
        config.color = None
    else:
        x_data: List[Any] = [x_data]
        y_data: List[Any] = [y_data]

    # Plot the data and (and best fits if applicable)
    figure_handle: plt.Figure = plt.figure(
        facecolor=config.facecolor, figsize=config.figsize
    )
    plt.axes().set_prop_cycle(config.cycler)
    scatter_handles: List[Any] = []

    includes_line_plot: bool = False
    for i, data_set in enumerate(zip(x_data, y_data)):
        if "x" in cumulative:
            x_array: np.ndarray = np.cumsum(data_set[0])
        else:
            x_array: np.ndarray = np.asarray(data_set[0])
        if "y" in cumulative:
            y_array: np.ndarray = np.cumsum(data_set[1])
        else:
            y_array: np.ndarray = np.asarray(data_set[1])

        # Convert to datetime for plotting if intending to use human-readable format
        if x_axis_human_tick_labels and any(
            not isinstance(x, datetime.datetime) for x in x_array
        ):
            if any(np.isnan(x_array)):
                raise ValueError(
                    "Cannot convert NaN to datetime. Use numeric x-axis labels or filter NaNs upstream."
                )
            x_array: List[datetime.datetime] = [
                datetime.datetime.fromtimestamp(ts) for ts in x_array
            ]

        if "scatter" in types:
            if log_norm_colors:
                kwargs.setdefault("norm", matplotlib.colors.LogNorm())
            if kwargs.get("c") is None:
                kwargs.setdefault("color", config.color)
            scatter_handles.append(
                plt.scatter(x_array, y_array, config.markersize, **kwargs)
            )

        if any(x in types for x in ["plot", "line"]):
            includes_line_plot: bool = True
            p = plt.plot(
                x_array, y_array, color=config.color, linewidth=config.linewidth
            )

        # Make sure that the axes facecolor matches the figure facecolor
        plt.gca().set(facecolor=config.facecolor)

        if plot_best_fit:
            if includes_line_plot:
                color: Any = p[0].get_color()  # noqa: misses earlier assignment
            else:
                color: Any = None
            if isinstance(plot_best_fit, int):
                degree: int = plot_best_fit
            else:
                degree: int = 1
            plot_best_fit_line(
                x_array, y_array, degree=degree, color=color, style=config
            )

        if rolling_mean_width or rolling_median_width:
            df: pd.DataFrame = pd.DataFrame({"x": x_array, "y": y_array})
            df.sort_values(by="x", ascending=True, inplace=True, ignore_index=True)
            if rolling_mean_width:
                plt.plot(
                    df.x,
                    df.y.rolling(rolling_mean_width).mean(),
                    color=config.mean_linecolor,
                    linewidth=config.mean_linewidth,
                    linestyle=config.mean_linestyle,
                )
            if rolling_median_width:
                plt.plot(
                    df.x,
                    df.y.rolling(rolling_median_width).median(),
                    color=config.median_linecolor,
                    linewidth=config.median_linewidth,
                    linestyle=config.median_linestyle,
                )

    # Adjust view & style where applicable
    xlabel_buffer: str = config.translate(xlabel, missing_response="return_input")
    if "x" in cumulative:
        xlabel_buffer += " (cumulative)"
    ylabel_buffer: str = config.translate(ylabel, missing_response="return_input")
    if "y" in cumulative:
        ylabel_buffer += " (cumulative)"
    apply_plot_labels(
        xlabel=xlabel_buffer, ylabel=ylabel_buffer, title=title, style=config
    )
    if show_colorbar or colorbar_label:
        cbar: plt.colorbar.Colorbar = plt.colorbar()
        if colorbar_label:
            cbar.set_label(colorbar_label, rotation=90, fontsize=config.label_fontsize)
    if legend_strings:
        if "scatter" in types:
            plt.legend(scatter_handles, legend_strings, fontsize=config.legend_fontsize)
        elif includes_line_plot:  # scatter legend overrides plot legend for now
            plt.legend(legend_strings, fontsize=config.legend_fontsize)
    adjust_axes(
        log_axes=log_axes,
        style=config,
        xlim=xlim,
        ylim=ylim,
        x_axis_human_tick_labels=x_axis_human_tick_labels,
        x_axis_formatter=x_axis_formatter,
    )
    apply_watermark(watermark, style=config)
    return figure_handle


def visualize_1d_distribution_interpreter(*args, **kwargs) -> plt.Figure:
    """Wrapper for visualize_x_y that can take as inputs:
    + arrays
    + lists
    + data frame (+ feature names to plot)
    """
    config: Style = kwargs.get("style", Style())
    data: Any = None
    if (num_positional_arguments := len(args)) == 1:
        data = args[0]

    # Received two positional inputs (interpreted as x_data & y_data arrays, or a list of such arrays)
    elif num_positional_arguments == 2:
        if isinstance(args[0], pd.DataFrame):
            if isinstance(args[1], list):
                data = [args[0].loc[:, x].tolist() for x in args[1]]
                kwargs.setdefault(
                    "legend_strings", [config.translate(x) for x in args[1]]
                )
            elif isinstance(args[1], str):
                data = args[0].loc[:, args[1]].tolist()
                kwargs.setdefault("xlabel", config.translate(args[1]))

    # Pass through to visualize_1d_distribution
    if data is not None:
        return visualize_1d_distribution(data=data, **kwargs)
    else:
        raise ValueError(
            "Issue encountered in visualize_1d_distribution_interpreter(), could not interpret the inputs"
        )


def visualize_x_y_input_interpreter(*args, **kwargs) -> plt.Figure:
    """Wrapper for visualize_x_y that can take as inputs:
    + arrays
    + lists
    + dictionary
    + data frame (+ feature names to plot)
    + VectorMultiSet
    + VectorSequence
    """
    config: Style = kwargs.get("style", Style())
    x_data: list = []
    y_data: list = []
    legend_strings: List[str] = []
    # Received a single positional input (so each item value must contain both x & y data)
    if (num_positional_arguments := len(args)) == 1:
        if isinstance((solo_input := args[0]), dict):
            # Here we have a dictionary with key=name, value=[x_data, y_data]
            x_data = [z[0] for z in solo_input.values()]
            y_data = [z[1] for z in solo_input.values()]
            legend_strings += solo_input.keys()
        elif isinstance(solo_input, list) and looks_like_list_of_lists(solo_input):
            # infer [[x1, y1], [x2, y2], ..., [xN, yN]]
            x_data = [z[0] for z in solo_input]
            y_data = [z[1] for z in solo_input]
        else:
            raise ValueError(f"Unknown input type: {type(solo_input)}")

    # Received two positional inputs (interpreted as x_data & y_data arrays, or a list of such arrays)
    elif num_positional_arguments == 2:
        if isinstance(args[0], pd.DataFrame) and isinstance(args[1], list):
            # infer: fxn(dataframe, [[x1name, y1name], [x2name, y2name], ...]
            x_data = [args[0].loc[:, z[0]] for z in args[1]]
            y_data = [args[0].loc[:, z[1]] for z in args[1]]
        elif len(args[0]) == len(args[1]):
            # infer: [x1, x2, ..., xN] and [y1, y2, ..., yN]   OR   [x_vec1, x_vec2, ...] and [y_vec1, y_vec2, ...]
            x_data = args[0]
            y_data = args[1]
        elif (not looks_like_list_of_lists(args[0])) or (
            looks_like_list_of_lists(args[0]) and (len(args[0]) == 1)
        ):
            if len(args[1]) > 1:
                # infer: fxn([x_all], [y1, y2, ..., yN], ...); also accepts: fxn(x_all, [y1, y2, ..., yN], ...)
                x_data = [args[0]] * len(args[1])
                y_data = args[1]

    # 3 inputs: for now this only handles a single x & y feature.  # TODO: extend to arbitrary number of pairs or y-axes
    elif num_positional_arguments == 3:
        # Check that the 2nd and 3rd inputs are strings (feature names)
        if all(isinstance(element, str) for element in args[1:3]):
            if isinstance(args[0], pd.DataFrame):
                df: pd.DataFrame = args[0]
            else:
                # If first argument NOT a dataframe, try to convert. This allows plots for isthmuslib vector collections
                try:
                    df: pd.DataFrame = pd.DataFrame(**args[0].dict())
                except AttributeError as ae:
                    raise AttributeError(
                        f"Tried to cast input to dict & dataframe, but was incompatible. Raised: {ae}"
                    )
                    # Check that the requested attributes are in the resultant data frame
            if not all(x in args[0].keys() for x in args[1:3]):
                raise ValueError(
                    f"Keys {args[1]} & {args[2]} not both in dataframe with: {df.keys().tolist()}"
                )
        else:
            raise ValueError(
                f"Expected positional arguments for feature names, got {args[1]} and {args[2]}"
            )
        x_data = df[args[1]]
        y_data = df[args[2]]
        kwargs.setdefault("xlabel", config.translate(args[1]))
        kwargs.setdefault("ylabel", config.translate(args[2]))

    # Only use the auto-generated legend strings if user did not specify legend_strings
    kwargs.setdefault("legend_strings", legend_strings)

    # Pass through to visualize_x_y
    if (len(x_data) > 0) and (len(y_data) == len(x_data)):
        return visualize_x_y(x_data=x_data, y_data=y_data, **kwargs)
    else:
        raise ValueError(
            "Issue encountered in visualize_x_y_input_interpreter(), could not interpret the inputs"
        )


def visualize_hist2d(
    x_data: Any,
    y_data: Any,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    style: Style = None,
    watermark: str = "",
    plot_best_fit: Union[bool, int] = False,
    xlim: Any = None,
    ylim: Any = None,
    show_colorbar: bool = True,
    colorbar_label: str = "counts",
    zscale: str = "linear",
    **kwargs,
) -> plt.Figure:
    """Visualize a 2-dimensional histogram (basically a heatmap of counts)

    :param x_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param y_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param xlabel: label text for the x-axis
    :param ylabel: label text for the y-axis
    :param title: title text
    :param style: configuration object (optional)
    :param watermark: watermark text
    :param plot_best_fit: whether to plot the best fit lines (specify degree, or pass True for degree 1)
    :param xlim: optional bound for the x-axis
    :param ylim: optional bound for the y-axis
    :param zscale: whether the colormap should be 'linear' or 'log'
    :param colorbar_label: text label to place next to the colorbar
    :param show_colorbar: whether to include the colorbar on teh plot
    :param kwargs: additional keyword arguments for matplotlib.pyplot.hist2d()
    :return: figure handle for the plot
    """
    # Set style. Overrides: kwargs > style input > Style() defaults
    config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})
    kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if (k not in config.dict()) or (k == "cmap")
    }
    kwargs.setdefault("cmap", config.sequential_cmap)

    x_data: List[Any] = to_list_if_other_array(x_data)
    y_data: List[Any] = to_list_if_other_array(y_data)

    # Make the plot
    figure_handle: plt.Figure = plt.figure(
        facecolor=config.facecolor, figsize=config.figsize
    )
    if zscale.lower() == "log":
        kwargs.setdefault("norm", matplotlib.colors.LogNorm())
    elif zscale.lower() != "linear":
        raise ValueError(f"zscale should be 'linear' or 'log' but received: {zscale}")
    plt.hist2d(x_data, y_data, **kwargs)
    if plot_best_fit:
        if isinstance(plot_best_fit, int):
            degree: int = plot_best_fit
        else:
            degree: int = 1
        plot_best_fit_line(x_data, y_data, degree=degree, color="k", style=config)

    # Adjust view & style where applicable
    plt.xlim(xlim)
    plt.ylim(ylim)
    apply_plot_labels(xlabel=xlabel, ylabel=ylabel, title=title, style=config)
    apply_watermark(watermark, style=config)
    if show_colorbar:
        cbar: plt.colorbar.Colorbar = plt.colorbar()
        cbar.set_label(colorbar_label, rotation=90, fontsize=config.label_fontsize)
    return figure_handle


def visualize_surface(
    x_data,
    y_data,
    z_data,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    xlim: Any = None,
    ylim: Any = None,
    style: Style = None,
    watermark: str = None,
    y_axis_ascending: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plots a surface (note: seaborn heatmap has somewhat rigid requirements for data completeness)

    :param x_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param y_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param z_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param xlabel: label text for the x-axis
    :param ylabel: label text for the y-axis
    :param title: title text
    :param xlim: optional bound for the x-axis
    :param ylim: optional bound for the y-axis
    :param style: configuration object (optional)
    :param watermark: watermark text
    :param y_axis_ascending: sort the axis low to high values bottom to top
    :param kwargs: additional keyword arguments for seaborn heatmap()
    :return: figure handle for the plot
    """
    # Set style. Overrides: kwargs > style input > Style() defaults
    config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})

    kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if (k not in config.dict()) or (k == "cmap")
    }
    kwargs.setdefault("cmap", config.sequential_cmap)

    x_data: List[Any] = to_list_if_other_array(x_data)
    y_data: List[Any] = to_list_if_other_array(y_data)
    z_data: List[Any] = to_list_if_other_array(z_data)

    # Initial data wrangling
    df: pd.DataFrame = pd.DataFrame({"x": x_data, "y": y_data, "z": z_data})
    pivoted: pd.DataFrame = df.pivot("y", "x", "z")

    # Make the plot
    figure_handle: plt.Figure = plt.figure(
        facecolor=config.facecolor, figsize=config.figsize
    )
    ax: plt.Axes = sns.heatmap(pivoted, **kwargs)

    # Adjust view & style where applicable
    plt.xlim(xlim)
    plt.ylim(ylim)

    apply_plot_labels(xlabel=xlabel, ylabel=ylabel, title=title, style=config)
    apply_watermark(watermark, style=config)
    if y_axis_ascending:
        ax.invert_yaxis()
    return figure_handle


def visualize_embedded_surface(
    x_data,
    y_data,
    z_data,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    xlim: Any = None,
    ylim: Any = None,
    style: Style = None,
    show_colorbar: bool = True,
    log_norm_colors: bool = True,
    show: bool = False,
    **kwargs,
) -> plt.Figure:
    """Plots a 2D surface in 3D

    :param x_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param y_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param z_data: can be an list-like object or a list of list-like objects (for multiple traces)
    :param xlabel: label text for the x-axis
    :param ylabel: label text for the y-axis
    :param title: title text
    :param xlim: optional bound for the x-axis
    :param ylim: optional bound for the y-axis
    :param style: configuration object (optional)
    :param kwargs: additional keyword arguments for seaborn heatmap()
    :param log_norm_colors: set to True to normalize the colorbar scale
    :param show_colorbar: set to True to show colorbar
    :param show: set to true to trigger plt.show()
    :return: figure handle for the plot
    """
    # Set style. Overrides: kwargs > style input > Style() defaults
    config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})
    kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if (k not in config.dict()) or (k == "cmap")
    }
    kwargs.setdefault("cmap", config.sequential_cmap)

    x_data: List[Any] = to_list_if_other_array(x_data)
    y_data: List[Any] = to_list_if_other_array(y_data)
    z_data: List[Any] = to_list_if_other_array(z_data)

    # Make the plot
    figure_handle: plt.Figure = plt.figure(
        facecolor=config.facecolor, figsize=config.figsize
    )
    ax = Axes3D(figure_handle)
    if log_norm_colors:
        kwargs.setdefault("norm", matplotlib.colors.LogNorm())
    figure_surface = ax.plot_trisurf(x_data, y_data, z_data, **kwargs)

    # Adjust view & style where applicable
    plt.xlim(xlim)
    plt.ylim(ylim)

    if show_colorbar:
        figure_handle.colorbar(figure_surface, shrink=0.5, aspect=5)
    apply_plot_labels(xlabel=xlabel, ylabel=ylabel, title=title, style=config)
    if show:
        plt.show()
    return figure_handle


def surface_from_dataframe(
    df: pd.DataFrame,
    x_col_name: str = "x",
    y_col_name: str = "y",
    z_col_name: str = "z",
    **kwargs,
) -> plt.Figure:
    """Extremely thin wrapper around plot_surface() that extracts the data from a dataframe"""
    col_names: List[str] = [x_col_name, y_col_name, z_col_name]
    if any(element not in (keys := df.keys().tolist()) for element in col_names):
        raise ValueError(
            f"Could not find all of {col_names} in data frame keys: {keys}"
        )
    return visualize_surface(df[x_col_name], df[y_col_name], df[z_col_name], **kwargs)


###################
# Aliases to mirror
# matplotlib.pyplot
###################

hist: Callable[[Any], plt.Figure] = visualize_1d_distribution_interpreter
hist2d: Callable[[Any], plt.Figure] = visualize_hist2d
surface: Callable[[Any], plt.Figure] = visualize_surface


def scatter(*args, **kwargs) -> plt.Figure:
    """Super thin wrapper for visualize(), to mimic matplotlib.plt interface"""
    return visualize_x_y_input_interpreter(*args, **kwargs, types="scatter")


def plot(*args, **kwargs) -> plt.Figure:
    """Super thin wrapper for visualize(), to mimic matplotlib.plt interface"""
    return visualize_x_y_input_interpreter(*args, **kwargs, types="plot")
