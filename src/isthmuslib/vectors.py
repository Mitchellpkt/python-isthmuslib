import time as time
from typing import List, Any, Tuple, Callable, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from .config import Style
from .utils import PickleUtils, Rosetta, as_list
import matplotlib as mpl
import seaborn as sns


class VectorMultiset(PickleUtils, Style, Rosetta):
    """ A set of vectors (which may or may not be ordered)"""
    data: pd.DataFrame = None
    name_root: str = None

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        """ The length of the vector set is the length of the data frame """
        return len(self.data)

    def values(self, attribute: str, cumulative: bool = False, *args) -> List[Any]:
        """ Retrieves a particular data feature by attribute name. Additional args unpack deeper

        :param attribute: name of feature to retrieve
        :param cumulative: whether to apply a cumulative sum (default = False)
        :return: Extracted data
        """
        values: pd.Series = self.data[attribute]
        if args:
            for unpack_next in args:
                values: List[Any] = [x.__getattribute__(unpack_next) for x in values]
        if cumulative:
            return np.cumsum(values).tolist()
        return values.tolist()

    ################
    # I/O
    ################

    def to_df(self) -> pd.DataFrame:
        """ Trivial helper function

        :return: hands back the self.data data frame
        """
        return self.data

    def from_dataframe(self, data_frame: pd.DataFrame, inplace: bool = True, **kwargs):
        """ Makes an VectorMultiset from a pandas data frame

        :param data_frame: data frame to import
        :param inplace: import inplace (default) or return the result
        :return: data frame if not inplace
        """
        if inplace:
            self.data = data_frame
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        else:
            return self.__class__(data=data_frame, **kwargs)

    def to_csv(self, file_path: str = None) -> None:
        """ Saves the data as a CSV file (note: this drops the name_root)

        :param file_path: path to write the file
        """
        if file_path is None:
            f'default_filename_{time.time():.0f}.csv'
        self.data.to_csv(file_path)

    def read_csv(self, file_path: str, inplace: bool = True, **kwargs):
        """Reads data from a CSV file

        :param file_path: file to read
        :param inplace: import inplace (default) or return the result
        :return:
        """
        data: pd.DataFrame = pd.read_csv(file_path)
        if inplace:
            self.data = data
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        else:
            return self.__class__(data=data, **kwargs)

    ################
    # Visualizations
    ################

    def visualize_x_y(self, x_name: str, y_name: str, cumulative: str = '', figsize: Any = None, title: str = None,
                      log_axes: str = '', types: Union[str, List[str]] = 'line', **kwargs) -> plt.Figure:
        """ Creates a 2D (scatter and/or line) plot of x and y data

        :param y_name: field or subfield to plot on x-axis
        :param x_name: field or subfield to plot on y-axis
        :param cumulative: 'x' or 'y' or 'xy' to plot cumulative data on that axis
        :param figsize: figure size
        :param title: figure title (raw strings are translated)
        :param log_axes: 'x' or 'y' or 'xy' to plot that axis/axes on a log scale
        :param types: any subset of ['scatter, 'plot']
        :return: figure handle
        """

        if figsize is None:
            figsize: Tuple[float, float] = self.figsize

        x_data: List[Any] = self.values(x_name, cumulative='x' in cumulative)
        y_data: List[Any] = self.values(y_name, cumulative='y' in cumulative)
        x_label, y_label = (self.translate(item) for item in [x_name, y_name])
        if 'x' in cumulative:
            x_data: List[Any] = np.cumsum(x_data)
            x_label += ' (cumulative)'
        if 'y' in cumulative:
            y_data: List[Any] = np.cumsum(y_data)
            y_label += ' (cumulative)'

        figure_handle: plt.Figure = plt.figure(facecolor=self.facecolor, figsize=figsize)
        if 'line' in (styles_list := as_list(types)):
            plt.scatter(x_data, y_data, self.markersize, self.color, **kwargs)
        if 'plot' in styles_list:
            plt.plot(x_data, y_data, color=self.color, linewidth=self.linewidth)

        plt.xlabel(x_label, fontsize=self.label_fontsize)
        plt.ylabel(y_label, fontsize=self.label_fontsize)
        if title:
            plt.title(title, fontsize=self.title_fontsize)
        elif self.name_root:
            plt.title(self.translate(self.name_root), fontsize=self.title_fontsize)
        plt.grid(self.grid)

        if self.tight_axes:
            plt.xlim([np.nanmin(x_data), np.nanmax(x_data)])
            plt.ylim([np.nanmin(y_data), np.nanmax(y_data)])

        if 'x' in log_axes:
            plt.xscale('log')
        if 'y' in log_axes:
            plt.yscale('log')

        return figure_handle

    def scatter(self, *args, **kwargs) -> plt.Figure:
        """ Creates a 2D scatter plot of x and y data (wraps visualize_x_y) """
        if isinstance(self, VectorSequence) and (len(args) == 1):
            args: tuple = (self.basis_col_name, args[0])  # sequences we can infer the basis for the x-axis
        return self.visualize_x_y(types='line', *args, **kwargs)

    def plot(self, *args, **kwargs) -> plt.Figure:
        """ Creates a 2D line plot of x and y data (wraps visualize_x_y) """
        if isinstance(self, VectorSequence) and (len(args) == 1):
            args: tuple = (self.basis_col_name, args[0])  # sequences we can infer the basis for the x-axis
        if isinstance(self, SlidingWindowResults) and (len(args) == 1):
            raise ValueError(f"Plot requires x & y. Hint: You might want SlidingWindowResults.plot_results()")
        return self.visualize_x_y(types='plot', *args, **kwargs)

    def hist(self, col_name: str, bins: Any = None, log_axes: str = '', title: str = None, **kwargs) -> plt.Figure:
        """ Plot a histogram of data. Useful kwargs: [cumulative, density]

        :param col_name: name of field to plot
        :param bins: number of bins or bins edges
        :param log_axes: 'y' to plot log-scale y-axis
        :param title: figure title (raw strings are translated)
        :return: figure handle
        """
        figure_handle: plt.Figure = plt.figure(facecolor=self.facecolor, figsize=self.figsize)
        plt.hist((data := self.values(col_name)), color=self.color, bins=bins, **kwargs)
        plt.xlabel(self.translate(col_name), fontsize=self.label_fontsize)

        # if cumulative:
        if kwargs.get('cumulative'):
            ylabel: str = 'cumulative counts'
        else:
            ylabel: str = 'counts'
        if kwargs.get('density'):
            ylabel += " (density)"
        plt.ylabel(ylabel, fontsize=self.label_fontsize)

        if title:
            plt.title(title, fontsize=self.title_fontsize)
        elif self.name_root:
            plt.title(self.translate(self.name_root), fontsize=self.title_fontsize)

        if self.tight_axes:
            plt.xlim([np.nanmin(data), np.nanmax(data)])
        if 'y' in log_axes:
            plt.yscale('log')
        if 'x' in log_axes:
            raise NotImplementedError("log x histogram requires different bin handling")

        return figure_handle

    def heatmap(self, x_name: str, y_name: str, z_name: str, y_axis_ascending: bool = True,
                figsize: Any = None, title: str = None, **kwargs) -> plt.Figure:
        """ Creates a heatmap from any 3 features of the vector set.

        :param x_name: name of column to use for x-axis values
        :param y_name: name of column to use for y-axis values
        :param z_name: name of column to use for z-axis (color) values
        :param y_axis_ascending: flip the y-axis to read low to high ascending (overrides weird sns default)
        :param figsize: figsize
        :param title: custom title
        :param kwargs: sns.heatmap() kwargs (hint: annot=True adds labels)
        :return: figure_handle

        to-do: aggregation_function: Callable[[Any], float] for handling multiple z values at one (x,y) coordinate
        """

        if isinstance(self, VectorSequence) and (x_name is None):
            x_name: str = self.basis_col_name
        if not kwargs.get(figsize) is None:
            figsize: Any = self.figsize
        df: pd.DataFrame = self.data.loc[:, [x_name, y_name, z_name]].pivot(y_name, x_name, z_name)
        figure_handle: plt.Figure = plt.figure(figsize=figsize, facecolor=self.facecolor)
        ax: plt.Axes = sns.heatmap(df, **kwargs)
        if y_axis_ascending:
            ax.invert_yaxis()
        plt.xlabel(self.translate(x_name), size=self.label_fontsize)
        plt.ylabel(self.translate(y_name), size=self.label_fontsize)
        if title is None:
            title: str = self.translate(z_name)
            if self.name_root is not None:
                title += f" ({self.translate(self.name_root)})"
        plt.title(title, size=self.title_fontsize)

        return figure_handle


class SlidingWindowResults(VectorMultiset):
    """ Results from Sequence.sliding_window(), which is a VectorMultiset with extra context baked in """
    window_width_col_name: str = 'window_width'
    window_start_col_name: str = 'window_start'

    def group_by_window_width(self) -> Dict[float, pd.DataFrame]:
        """ Helper function that returns individual dataframes for each window width """
        return {x: self.data[self.data['window_width'] == x] for x in set(self.data.window_width)}

    def plot_results(self, col_name: str, cumulative: bool = False, figsize: Any = None, title: str = None,
                     legend_override: List[str] = None, **kwargs) -> plt.Figure:
        """ Plot any sequence (based on the data frame column name)

        :param col_name: dataframe column to plot
        :param cumulative: plot cumulative values (default: False)
        :param figsize: specify a custom figure size if desired
        :param title: specify a custom title if desired
        :param legend_override: specify custom legend keys if desired
        :return: figure handle
        """
        if figsize is None:
            figsize = self.figsize

        if title is None:
            if self.name_root:
                title: str = f"{self.translate_time(self.name_root)}: {self.translate(col_name)}"
            else:
                title: str = self.translate(col_name)

        traces: List[mpl.collections.PathCollection] = []
        labels: List[str] = []
        figure_handle = plt.figure(figsize=figsize, facecolor=self.facecolor)

        # For each `window_width` add a trace to the plot
        group_by_window_width: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key in sorted(group_by_window_width.keys()):  # looping over group_by_window_width.items() gives wrong order
            value: pd.DataFrame = group_by_window_width[key]
            y_vec: List[float] = value[col_name]
            if cumulative:
                y_vec: List[float] = np.cumsum(y_vec)
            h: mpl.collections.PathCollection = plt.scatter(value[self.window_start_col_name], y_vec,
                                                            self.markersize, **kwargs)
            traces.append(h)
            labels.append(f"Window width: {key}")
            plt.plot(value[self.window_start_col_name], y_vec, linewidth=self.linewidth)

        plt.xlabel('Window start', fontsize=self.label_fontsize)
        plt.ylabel(self.translate(col_name), fontsize=self.label_fontsize)
        plt.title(title, fontsize=self.title_fontsize)
        if legend_override:
            plt.legend(handles=traces, labels=legend_override, fontsize=self.legend_fontsize)
        else:
            plt.legend(handles=traces, labels=labels, fontsize=self.legend_fontsize)

        return figure_handle

    def plot_pdfs(self, col_name: str, legend_override: List[str] = None, **kwargs) -> plt.Figure:
        """ Plot the probability density function(s) of the sliding window results. Useful kwargs: [cumulative, density]

        :param col_name: which field should be plotted
        :param legend_override: custom legend labels if desired (must have same length as number of unique widths)
        :param kwargs: plt.hist() keyword arguments.
]        :return: figure handle
        """
        if not kwargs.get('alpha'):
            kwargs['alpha'] = 0.5  # Transparency is helpful for reading overlapping histograms
        figure_handle: plt.Figure = plt.figure(figsize=self.figsize, facecolor=self.facecolor)
        grouped: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key, value in grouped.items():
            plt.hist(value[col_name], **kwargs)
        plt.legend(grouped.keys())
        if legend_override:
            plt.legend(legend_override, fontsize=self.legend_fontsize)
        else:
            plt.legend([f"{x} weeks " for x in grouped], fontsize=self.legend_fontsize)
        plt.xlabel(self.translate(col_name), fontsize=self.label_fontsize)
        if kwargs.get("cumulative"):
            ylabel: str = 'cumulative counts'
        else:
            ylabel: str = 'counts'
        if kwargs.get("density"):
            ylabel += " (density)"
        plt.ylabel(ylabel, fontsize=self.label_fontsize)
        return figure_handle

    def heatmap_feature(self, col_name: str, **kwargs) -> plt.Figure:
        return self.heatmap(self.window_start_col_name, self.window_width_col_name, col_name, **kwargs)


class VectorSequence(VectorMultiset):
    """ Set of vectors in a sequence ordered according to some basis (e.g. height, wavelength, time) """

    data: pd.DataFrame = None
    basis_col_name: str = 'basis'
    name_root: str = None

    class Config:
        arbitrary_types_allowed = True

    def index_to_basis(self) -> None:
        """  Adds (or overwrites) 'basis' column with a default index [0, 1, 2, ...] """
        self.data[self.basis_col_name] = list(range(len(self.data)))

    def sort(self, inplace: bool = True, reset_index: bool = True):
        """ Sorts the data points by the basis

        :param inplace: import inplace (default) or return the result
        :param reset_index: indicate whether or not to reset the data frame index (default = True)
        :return: sorted data frame, if inplace is False
        """
        sorted_data: pd.DataFrame = self.data.sort_values(by=self.basis_col_name, inplace=False)
        if reset_index:
            sorted_data.reset_index(inplace=True)
        if inplace:
            self.data = sorted_data
        else:
            return VectorSequence(basis_col_name=self.basis_col_name, name_root=self.name_root, data=sorted_data)

    def slice(self, start_at: Any = None, stop_at: Any = None, inplace: bool = False, reset_index: bool = True):
        """ Slices the VectorSequence according to the basis

        :param start_at: Start point (can be None)
        :param stop_at: Stop point (can be None)
        :param inplace: import inplace or return the result (default)
        :param reset_index: indicate whether or not to reset the data frame index (default = True)
        :return: VectorSequence if not inplace
        """
        if start_at is None:
            start_at: Any = min(self.data[self.basis_col_name])
        if stop_at is None:
            stop_at: Any = max(self.data[self.basis_col_name])
        in_range: pd.DataFrame = self.data[self.data[self.basis_col_name].between(start_at, stop_at)]
        if reset_index:
            in_range.reset_index(inplace=True)
        if inplace:
            self.data = in_range
        else:
            return VectorSequence(basis_col_name=self.basis_col_name, name_root=self.name_root, data=in_range)

    def sliding_window(self, function: Callable[[Any, List[Any], Dict[str, Any]], Dict[str, Any]],
                       window_widths: List[float], verbose: bool = False, step_size: float = None,
                       overlapping: bool = False, *args, **kwargs) -> SlidingWindowResults:
        """ Apply function in a sliding window over the sequence

        :param function: callable to be applied
        :param window_widths: list of window widths to use
        :param step_size: how far apart to space windows
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        :param verbose: verbose output?
        :param overlapping: whether the windows should overlap
        :return: SlidingWindowResults (see the dataframe attribute for results)
        """

        # Validation
        if self.basis_col_name is None:
            raise AttributeError('cannot apply sliding window analysis without knowing which field is the basis')
        if self.basis_col_name not in self.data.keys():
            raise AttributeError(f'basis_col_name {self.basis_col_name} not a column in data frame:\n{self.data}')

        df: pd.DataFrame = pd.DataFrame()
        for i, window_width in enumerate(window_widths):
            basis: pd.Series = self.data[self.basis_col_name].tolist()
            # Determine window sizes
            if step_size:
                starts: List[float] = list(np.arange(min(basis), max(basis) - step_size, step_size))
            else:
                if overlapping:
                    starts: List[float] = [x for x in basis if basis[-1] - x >= window_width]
                else:
                    starts: List[float] = list(np.arange(min(basis), max(basis) - window_width, window_width))

            for j, window_start in enumerate(starts):
                window_data: VectorSequence = self.slice(start_at=window_start, stop_at=window_start + window_width,
                                                         inplace=False)
                evaluation: Dict[str, Any] = function(window_data, *args, **kwargs)
                df = df.append({'window_width': window_width, 'window_start': window_start,
                                **evaluation}, ignore_index=True)

                if verbose:
                    logger.info(f"\nWindow width: {window_width} ({i + 1} of {len(window_widths)})\n"
                                f"Starting at: {window_start} ({j + 1} of {len(starts)})")

        return SlidingWindowResults(window_width_col_name='window_width', window_start_col_name='window_start',
                                    data=df)

    def repackage(self, instance: Any, sequence_attribute: str = 'series', basis_name: str = 'basis') -> Any:
        cache: Dict[str, Any] = instance.dict()
        cache['data'] = pd.DataFrame({x: input.__getattribute__(x) for x in [sequence_attribute, basis_name]})
        cache['basis_col_name'] = basis_name
        return self.__class__(**cache)


class OrderedSeries:
    def __init__(self, **data: Any):
        super().__init__(**data)
        raise TypeError("Notice: OrderedSeries class has been supplanted by VectorSequence, see repackage() method")
