import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .config import Style
from .utils import looks_like_list_of_lists
from typing import List, Any, Union, Tuple, Callable, Dict


##################
# Helper functions
##################

def plot_best_fit_line(x_data: Any, y_data: Any, degree: int = None, color: str = 'k', style: Style = None,
                       **kwargs) -> None:
    """ Adds the line of best fit to a plot

    :param x_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param y_data: can be an array-like object or a list of array-like objects (for multiple traces)
    :param degree: degree oft the polynomial to fit
    :param color: color for the line
    :param style: optional style object (which contains a line width parameter)
    :param kwargs: additional keyword arguments for matplotlib.pyplot.plt
    :return: None
    """
    if kwargs.get("linewidth"):
        line_width = kwargs['linewidth']
    else:
        if (not style) or (not style.linewidth):
            line_width: float = Style().linewidth
        else:
            line_width: float = style.linewidth
    if not degree:
        degree: int = 1
    z = np.polyfit(np.array(x_data), np.array(y_data), degree)
    plt.plot(x_data, [z[0] * x + z[1] for x in x_data], color=color, linewidth=line_width, **kwargs)


def adjust_axes(log_axes: Union[str, List[str]] = '', style: Style = None, xlim: Any = None, ylim: Any = None) -> None:
    """ Helper function that adjusts the axes of a plot as specified

    :param log_axes: can look like: 'x' or 'y' or 'xy' // or look like: ['x'] or ['y'] or ['x', 'y']
    :param style: the Style object to apply
    :param xlim: bound for the x-axis
    :param ylim: bound for the y-axis
    :return: None
    """
    if not style:
        style = Style()
    plt.grid(style.grid)
    if xlim:
        plt.xlim(xlim)
    else:
        if style.tight_axes:
            plt.autoscale(enable=True, axis='x', tight=True)
    if ylim:
        plt.ylim(ylim)
    else:
        if style.tight_axes:
            plt.autoscale(enable=True, axis='y', tight=True)
    if 'x' in log_axes:
        plt.xscale('log')
    if 'y' in log_axes:
        plt.yscale('log')


def applylabels(xlabel: str = '', ylabel: str = '', title: str = '', style: Style = None) -> None:
    """ Helper function to apply labels (xlabel, ylabel, title) according to the style guide (including translation)

    :param xlabel: text or rosetta key for the x-axis label
    :param ylabel: text or rosetta key for the y-axis label
    :param title: text or rosetta key for the title
    :param style: config Style object
    """
    if not style:
        style: Style = Style()
    plt.xlabel(style.rosetta.translate(xlabel, missing_response='return_input'), fontsize=style.label_fontsize)
    plt.ylabel(style.rosetta.translate(ylabel, missing_response='return_input'), fontsize=style.label_fontsize)
    plt.title(style.rosetta.translate(title, missing_response='return_input'), fontsize=style.title_fontsize)


def apply_watermark(watermark_text: str, style: Style = None, **kwargs) -> None:
    """ Helper function to apply watermark text to a plot based on config Style object parameters

    :param watermark_text: Text to display
    :param style: configuration object (Style)
    :param kwargs: additional keyword arguments for matplotlib.pyplot.text
    :return: None
    """
    if not watermark_text:
        return None

    if not style:
        style: Style = Style()

    # Accept either (x,y) placement OR a single scalar margin that will be used in both dimensions
    if isinstance(style.watermark_placement, (float, int)):
        position: Tuple[float, float] = (style.watermark_placement, style.watermark_placement)
    elif len(style.watermark_placement) == 2:
        position: Tuple[float, float] = style.watermark_placement
    else:
        raise ValueError(f"Unable to interpret watermark {style.watermark_placement=}")

    margin_calc: Callable[[Union[float, int]], float] = lambda margin, span: span[0] + margin * (span[1] - span[0])
    x_coordinate: float = margin_calc(position[0], plt.xlim())
    y_coordinate: float = margin_calc(position[1], plt.ylim())
    plt.text(x_coordinate, y_coordinate, watermark_text, fontsize=style.watermark_fontsize, c=style.watermark_color,
             **kwargs)


##################
# Core visualize
# functionality
##################

def visualize_1d_distribution(data: Any, xlabel: str = '', ylabel: str = 'counts', title: str = '', log_axes: str = '',
                              style: Style = None, watermark: str = '', multi: bool = None,
                              legend_strings: Union[Tuple[str], List[str]] = None, xlim: Any = None, ylim: Any = None,
                              **kwargs) -> plt.Figure:
    """ Core function for visualizing 1-dimensional distribution(s)

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
    # Use package style defaults for any fields not specified in style (or all fields if style object is not provided)
    config: Style = Style()
    if style:
        config: Style = Style(**{**Style().dict(), **style.dict()})

    # If not specified, try to ascertain whether one or multiple data sets are being provided
    if multi or ((multi is None) and looks_like_list_of_lists(data)):
        config.color = None
        kwargs.setdefault('alpha', config.multi_hist_alpha)
    else:
        data: List[Any] = [data]

    # Plot the data and (and best fits if applicable)
    figure_handle: plt.Figure = plt.figure(facecolor=config.facecolor, figsize=config.figsize)
    for data_set in data:
        # plotting on log x-axis requires special pre-treatment (log-distributed bin edges)
        if 'x' in log_axes:
            if bins := kwargs.get("bins"):
                if not isinstance(bins, (int, float)):
                    hist_bins = bins
                else:
                    hist_bins = np.logspace(np.log10(min(data_set)), np.log10(max(data_set)), bins)
            else:
                hist_bins = np.logspace(np.log10(min(data_set)), np.log10(max(data_set)), config.histogram_bins)
        else:
            hist_bins = kwargs.get("bins")
        plt.hist(data_set, color=config.color, bins=hist_bins, **{k: v for k, v in kwargs.items() if k != 'bins'})

    # Adjust view & style where applicable
    xlabel_buffer: str = config.translate(xlabel, missing_response='return_input')
    ylabel_translated: str = config.translate(ylabel, missing_response='return_input')
    if kwargs.get('cumulative'):
        ylabel_buffer: str = f'cumulative {ylabel_translated}'
    else:
        ylabel_buffer: str = ylabel_translated
    if kwargs.get('density'):
        ylabel_buffer += " (density)"
    applylabels(xlabel=xlabel_buffer, ylabel=ylabel_buffer, title=title, style=config)
    if legend_strings:
        plt.legend(legend_strings, fontsize=config.legend_fontsize)

    adjust_axes(log_axes=log_axes, style=config, xlim=xlim, ylim=ylim)
    apply_watermark(watermark, style=config)
    return figure_handle


def visualize_x_y(x_data: Any, y_data: Any, xlabel: str = '', ylabel: str = '', title: str = '', cumulative: str = '',
                  log_axes: Union[str, List[str]] = '', types: Union[str, List[str]] = 'scatter', style: Style = None,
                  watermark: str = '', multi: bool = None,
                  legend_strings: Union[Tuple[str], List[str]] = None, xlim: Any = None, ylim: Any = None,
                  plot_best_fit: Union[bool, int] = False, **kwargs) -> plt.Figure:
    """ Core function for visualizing 2-dimensional data sets

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
    :return: figure handle for the plot
    """
    # Use package style defaults for any fields not specified in style (or all fields if style object is not provided)
    config: Style = Style()
    if style:
        config: Style = Style(**{**Style().dict(), **style.dict()})

    # If not specified, try to ascertain whether one or multiple data sets are being provided
    if multi or ((multi is None) and looks_like_list_of_lists(y_data)):
        config.color = None
    else:
        x_data: List[Any] = [x_data]
        y_data: List[Any] = [y_data]

    # Plot the data and (and best fits if applicable)
    figure_handle: plt.Figure = plt.figure(facecolor=config.facecolor, figsize=config.figsize)
    scatter_handles: List[Any] = []
    for data_set in zip(x_data, y_data):
        if 'x' in cumulative:
            x_array: np.ndarray = np.cumsum(data_set[0])
        else:
            x_array: np.ndarray = np.asarray(data_set[0])
        if 'y' in cumulative:
            y_array: np.ndarray = np.cumsum(data_set[1])
        else:
            y_array: np.ndarray = np.asarray(data_set[1])

        if 'scatter' in types:
            scatter_handles.append(plt.scatter(x_array, y_array, config.markersize, config.color, **kwargs))

        if includes_line_plot := any(x in types for x in ['plot', 'line']):
            p = plt.plot(x_array, y_array, color=config.color, linewidth=config.linewidth)

        if plot_best_fit:
            if includes_line_plot:
                color: Any = p[0].get_color()  # noqa: misses earlier assignment
            else:
                color: Any = None
            if isinstance(plot_best_fit, int):
                degree: int = plot_best_fit
            else:
                degree: int = 1
            plot_best_fit_line(x_array, y_array, degree=degree, color=color, style=config)

    # Adjust view & style where applicable
    xlabel_buffer: str = config.translate(xlabel, missing_response='return_input')
    if 'x' in cumulative:
        xlabel_buffer += ' (cumulative)'
    ylabel_buffer: str = config.translate(ylabel, missing_response='return_input')
    if 'y' in cumulative:
        ylabel_buffer += ' (cumulative)'
    applylabels(xlabel=xlabel_buffer, ylabel=ylabel_buffer, title=title, style=config)
    if legend_strings and ('scatter' in types):
        plt.legend(scatter_handles, legend_strings, fontsize=config.legend_fontsize)
    adjust_axes(log_axes=log_axes, style=config, xlim=xlim, ylim=ylim)
    apply_watermark(watermark, style=config)
    return figure_handle


def visualize_x_y_input_interpreter(*args, **kwargs) -> plt.Figure:
    """ Wrapper for visualize_x_y that can take as inputs:
        + arrays
        + lists
        + dictionary
        + data frame (+ feature names to plot)
        + VectorMultiSet
        + VectorSequence
        """

    x_data: list = []
    y_data: list = []
    legend_strings: List[str] = []
    # Received a single positional input (so each item value must contain both x & y data)
    if (num_positional_arguments := len(args)) == 1:
        if isinstance((solo_input := args[0]), dict):
            x_data = [z[0] for z in solo_input.values()]
            y_data = [z[1] for z in solo_input.values()]
            legend_strings += solo_input.keys()
        else:
            raise ValueError(f"Unknown input type: {type(solo_input)}")

    # Received two positional inputs (interpreted as x_data & y_data arrays, or a list of such arrays)
    elif num_positional_arguments == 2:
        x_data = args[0]
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
                    raise AttributeError(f"Tried to cast input to dict & dataframe, but was incompatible. Raised: {ae}")
                    # Check that the requested attributes are in the resultant data frame
            if not all(x in args[0].keys() for x in args[1:3]):
                raise ValueError(f"Keys {args[1]} & {args[2]} not both in dataframe with: {df.keys().tolist()}")
        else:
            raise ValueError(f"Expected positional arguments for feature sames, got {args[1]} and {args[2]}")
        x_data = df[args[1]]
        y_data = df[args[2]]
        legend_strings = [args[1], args[2]]

    # Only use the auto-generated legend strings if user did not specify legend_strings
    kwargs.setdefault("legend_strings", legend_strings)

    # Pass through to visualize_x_y
    if (len(x_data) > 0) and (len(y_data) == len(x_data)):
        return visualize_x_y(x_data=x_data, y_data=y_data, **kwargs)
    else:
        raise ValueError("Something went wrong in interpret_viz_inputs(), could not interpret the inputs")


def visualize_hist2d(x_data: Any, y_data: Any, xlabel: str = '', ylabel: str = '', title: str = '', style: Style = None,
                     watermark: str = '', plot_best_fit: Union[bool, int] = False,
                     xlim: Any = None, ylim: Any = None, show_colorbar: bool = True,
                     colorbar_label: str = 'counts', **kwargs) -> plt.Figure:
    """ Visualize a 2-dimensional histogram (basically a heatmap of counts)

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
    :param kwargs: additional keyword arguments for matplotlib.pyplot.hist2d()
    :return: figure handle for the plot
    """
    # Use package style defaults for any fields not specified in style (or all fields if style object is not provided)
    config: Style = Style()
    if style:
        config: Style = Style(**{**Style().dict(), **style.dict()})

    # Make the plot
    figure_handle: plt.Figure = plt.figure(facecolor=config.facecolor, figsize=config.figsize)
    plt.hist2d(x_data, y_data, **kwargs)
    if plot_best_fit:
        if isinstance(plot_best_fit, int):
            degree: int = plot_best_fit
        else:
            degree: int = 1
        plot_best_fit_line(x_data, y_data, degree=degree, color='k', style=config)

    # Adjust view & style where applicable
    plt.xlim(xlim)
    plt.ylim(ylim)
    applylabels(xlabel=xlabel, ylabel=ylabel, title=title, style=config)
    apply_watermark(watermark, style=config)
    if show_colorbar:
        cbar: plt.colorbar.Colorbar = plt.colorbar()
        cbar.set_label(colorbar_label, rotation=90, fontsize=config.label_fontsize)
    return figure_handle


def visualize_surface(x_data, y_data, z_data, xlabel: str = '', ylabel: str = '', title: str = '', xlim: Any = None,
                      ylim: Any = None, style: Style = None, watermark: str = None, y_axis_ascending: bool = True,
                      **kwargs) -> plt.Figure:
    """ Plots a surface (note: seaborn heatmap has somewhat rigid requirements for data completeness)

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
    :param kwargs: additional keyword arguments for seaborn heatmap()
    :return: figure handle for the plot
    """
    # Use package style defaults for any fields not specified in style (or all fields if style object is not provided)
    config: Style = Style()
    if style:
        config: Style = Style(**{**Style().dict(), **style.dict()})

    # Initial data wrangling
    df: pd.DataFrame = pd.DataFrame({'x': x_data, 'y': y_data, 'z': z_data})
    pivoted: pd.DataFrame = df.pivot("y", "x", "z")

    # Make the plot
    figure_handle: plt.Figure = plt.figure(facecolor=config.facecolor, figsize=config.figsize)
    ax: plt.Axes = sns.heatmap(pivoted, **kwargs)

    # Adjust view & style where applicable
    plt.xlim(xlim)
    plt.ylim(ylim)
    applylabels(xlabel=xlabel, ylabel=ylabel, title=title, style=config)
    apply_watermark(watermark, style=config)
    if y_axis_ascending:
        ax.invert_yaxis()
    return figure_handle


def surface_from_dataframe(df: pd.DataFrame, x_col_name: str = 'x', y_col_name: str = 'y', z_col_name: str = 'z',
                           **kwargs) -> plt.Figure:
    """ Extremely thin wrapper around plot_surface() that extracts the data from a dataframe """
    col_names: List[str] = [x_col_name, y_col_name, z_col_name]
    if any(element not in (keys := df.keys().tolist()) for element in col_names):
        raise ValueError(f"Could not find all of {col_names} in data frame keys: {keys}")
    return visualize_surface(df[x_col_name], df[y_col_name], df[z_col_name], **kwargs)


###################
# Aliases to mirror
# matplotlib.pyplot
###################

hist: Callable[[Any], plt.Figure] = visualize_1d_distribution
hist2d: Callable[[Any], plt.Figure] = visualize_hist2d
surface: Callable[[Any], plt.Figure] = visualize_surface


def scatter(*args, **kwargs) -> plt.Figure:
    """ Super thin wrapper for visualize(), to mimic matplotlib.plt interface """
    return visualize_x_y_input_interpreter(*args, **kwargs, types='scatter')


def plot(*args, **kwargs) -> plt.Figure:
    """ Super thin wrapper for visualize(), to mimic matplotlib.plt interface """
    return visualize_x_y_input_interpreter(*args, **kwargs, types='plot')
