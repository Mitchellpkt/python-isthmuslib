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


class SlidingWindowResults(Style, Rosetta):
    """ Results from OrderedSeries.sliding_window() """
    dataframe: pd.DataFrame
    window_width_field_name: str = 'window_width'
    window_start_field_name: str = 'window_start'
    name_root: str = None

    class Config:
        arbitrary_types_allowed: bool = True

    def plot(self, field_name: str, cumulative: bool = False, figsize: Any = None, title: str = None,
             legend_override: List[str] = None) -> plt.Figure:
        """
        Plot any series (based on the data frame column name)
        @param field_name: dataframe column to plot
        @param cumulative: plot cumulative values (default: False)
        @param figsize: specify a custom figure size if desired
        @param title: specify a custom title if desired
        @param legend_override: specify custom legend keys if desired
        @return: figure handle
        """
        if figsize is None:
            figsize = self.figsize

        if title is None:
            if self.name_root:
                title: str = f"{self.translate_time(self.name_root)}: {self.translate(field_name)}"
            else:
                title: str = self.translate(field_name)

        traces: List[mpl.collections.PathCollection] = []
        labels: List[str] = []
        figure_handle = plt.figure(figsize=figsize, facecolor=self.facecolor)

        # For each `window_width` add a trace to the plot
        group_by_window_width: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key in sorted(group_by_window_width.keys()):  # looping over group_by_window_width.items() gives wrong order
            value: pd.DataFrame = group_by_window_width[key]
            y_vec: List[float] = value[field_name]
            if cumulative:
                y_vec: List[float] = np.cumsum(y_vec)
            h: mpl.collections.PathCollection = plt.scatter(value[self.window_start_field_name], y_vec, self.markersize)
            traces.append(h)
            labels.append(f"Window width: {key}")
            plt.plot(value[self.window_start_field_name], y_vec, linewidth=self.linewidth)

        plt.xlabel('Window start', fontsize=self.label_fontsize)
        plt.ylabel(self.translate(field_name), fontsize=self.label_fontsize)
        plt.title(title, fontsize=self.title_fontsize)
        if legend_override:
            plt.legend(handles=traces, labels=legend_override, fontsize=self.legend_fontsize)
        else:
            plt.legend(handles=traces, labels=labels, fontsize=self.legend_fontsize)

        return figure_handle

    def plot_pdf(self, field_name: str, legend_override: List[str] = None, **kwargs) -> plt.Figure:
        figure_handle: plt.Figure = plt.figure(figsize=self.figsize, facecolor=self.facecolor)
        grouped: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key, value in grouped.items():
            plt.hist(value[field_name], **kwargs)
        plt.legend(grouped.keys())
        if legend_override:
            plt.legend(legend_override, fontsize=self.legend_fontsize)
        else:
            plt.legend([f"{x} weeks " for x in grouped], fontsize=self.legend_fontsize)
        plt.xlabel(self.translate(field_name), fontsize=self.label_fontsize)
        if ('cumulative', True) in kwargs.items():
            ylabel: str = 'cumulative counts'
        else:
            ylabel: str = 'counts'
        if ('density', True) in kwargs.items():
            ylabel += " (density)"
        plt.ylabel('Normalized cumulative counts', fontsize=self.label_fontsize)
        return figure_handle

    def heatmap_feature(self, field_name: str, **kwargs) -> plt.Figure:
        """
        Heatmap a feature as a function of window width and start time
        @param field_name: field to plot
        @param kwargs: sns heatmap keyword arguments
        @return: figure handle
        """
        df: pd.DataFrame = pd.DataFrame({
            self.window_width_field_name: self.dataframe[self.window_width_field_name],
            self.window_start_field_name: self.dataframe[self.window_start_field_name],
            field_name: self.dataframe[field_name]
        })
        pivoted: pd.DataFrame = df.pivot(self.window_width_field_name, self.window_start_field_name, field_name)
        figure_handle = plt.figure(figsize=self.figsize, facecolor=self.facecolor)
        sns.heatmap(pivoted, **kwargs)
        plt.xlabel(self.translate(self.window_start_field_name), fontsize=self.label_fontsize)
        plt.ylabel(self.translate(self.window_width_field_name), fontsize=self.label_fontsize)
        plt.title(self.translate(self.name_root), fontsize=self.title_fontsize)
        return figure_handle

    def group_by_window_width(self) -> Dict[float, pd.DataFrame]:
        """ Helper function that returns individual dataframes for each window width """
        return {x: self.dataframe[self.dataframe['window_width'] == x] for x in set(self.dataframe.window_width)}


class OrderedSeries(PickleUtils, Style, Rosetta):
    """ A well ordered series (for example a timeseries) with miscellaneous helper functions
        - the series vector is a list (of anything) that is related by some order. Does not need to be sortable.
        - the basis vector is a list (of the same length) describing said order. Must be sortable (e.g. int, str)
        The series is required, however the basis is optional """
    series: List[Any]
    basis: List[Any] = None  # optional
    name_root: str = None  # optional to use for labeling

    def from_dataframe(self, data_frame: pd.DataFrame, series_name: str, basis_name: str = None,
                       name_root: str = None):
        """
        Converts a dataframe to an OrderedSeries
        @param df: data frame to import
        @param series_name: column name for the series
        @param basis_name: column name for the basis
        @param name_root: optional name for the series
        @return: OrderedSeries
        """
        ordered_series: OrderedSeries = OrderedSeries(series=data_frame[series_name].tolist(), name_root=name_root)
        if basis_name:
            ordered_series.basis = data_frame[basis_name]
        else:
            return ordered_series

    def union_with(self, other, inplace: bool = False, sort: bool = True):
        """
        Combine two ordered series
        @param other: other OrderedSeries
        @param inplace: change values inplace or return the results?
        @param sort: sort the result?
        @return: if not inplace, returns OrderedSeries
        """
        # Check for extant and/or conflicting name_root values
        if self.name_root:
            if other.name_root:
                if self.name_root != other.name_root:
                    raise ValueError(f"Cannot union series, conflicting names: {self.name_root} and {other.name_root}")
                else:
                    use_name_root = self.name_root
            else:
                use_name_root = self.name_root
        else:
            use_name_root = other.name_root

        result_series: OrderedSeries = OrderedSeries(series=self.series + other.series, basis=self.basis + other.basis,
                                                     name_root=use_name_root)
        if sort:
            result_series.sort(inplace=True)
        if inplace:
            self.series = result_series.series
            self.basis = result_series.basis
        else:
            return result_series

    def __len__(self) -> int:
        """
        The length of an OrderedSeries is determined by its series attribute
        @return: length of the series
        """
        if self.basis and self.series:
            if len(self.basis) != len(self.series):
                raise Exception(f"length mismatch: { {x: len(self.__getattribute__(x)) for x in ['basis', 'series']} }")
        return len(self.series)

    def __add__(self, other):
        """ Magic method '+' wraps OrderedSeries.union_with() """
        return self.union_with(other, inplace=False)

    def to_csv(self, file_path: str = None) -> None:
        """
        Write an OrderedSeries out to CSV
        @param file_path: where to write the csv file
        """
        if file_path is None:
            f'default_filename_{time.time():.0f}.csv'
        df: pd.DataFrame = pd.DataFrame({'basis': self.basis, 'series': self.series})
        df.to_csv(file_path)

    def read_csv(self, file_path: str, basis_name: str = 'basis', series_name: str = 'series', inplace: bool = False):
        """
        Read an OrderedSeries from CSV
        @param file_path: file path to read
        @param inplace: if True load is place, otherwise return the result
        @return: OrderedSeries if inplace is False
        """
        df: pd.DataFrame = pd.read_csv(file_path)
        if inplace:
            self.basis, self.series = (df[x].tolist() for x in [basis_name, series_name])
        else:
            return OrderedSeries(series=df[series_name].tolist(), basis=df[basis_name].tolist())

    def to_df(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with the OrderedSeries data
        @return: pandas dataframe
        """
        return pd.DataFrame({'basis': self.basis, 'series': self.basis})

    def index_to_basis(self):
        """ Set index basis. In other contexts, feel free to override with relevant basis """
        self.basis = list(range(len(self.series)))

    def sort(self, inplace=True):
        """ Sorts the basis and rearranges the series according """
        sorted_series: List[Any] = [x for _, x in sorted(zip(self.basis, self.series))]
        if inplace:
            self.series = sorted_series
            self.basis = sorted(self.basis)
        else:
            return OrderedSeries(series=sorted_series, basis=sorted(self.basis))

    def slice(self, start_at: Any = None, stop_at: Any = None, inplace: bool = False):
        """
        Slices an OrderedSeries based on the basis
        @param start_at: Low end of the slice
        @param stop_at: High end of the slice
        @param inplace:
        @return: OrderedSeries if inplace is False
        """
        if start_at is None:
            start_at: Any = min(self.basis)
        if stop_at is None:
            stop_at: Any = max(self.basis)
        in_range: List[Tuple[Any, Any]] = [x for x in zip(self.basis, self.series) if start_at <= x[0] <= stop_at]
        new_series: OrderedSeries = OrderedSeries(basis=[x[0] for x in in_range], series=[x[1] for x in in_range])
        if inplace:
            self.series = new_series.series
            self.basis = new_series.basis
        else:
            return new_series

    def values(self, attribute: str, cumulative: bool = False) -> List[Any]:
        """
        Retrieves the values for a given attribute. Notably, can read from the top level or extract from Series objects
        @param attribute: attribute to retrieve
        @param cumulative: Apply a cumulative sum to the result?
        @return: the retrieved data
        """
        if attribute in ['series', 'basis']:
            values: List[Any] = self.__getattribute__(attribute)
        else:
            values: List[Any] = [x.__getattribute__(attribute) for x in self.series]
        if cumulative:
            return np.cumsum(values).tolist()
        return values

    def sliding_window(self, function: Callable, window_widths: List[float], args: Tuple[Any] = None,
                       kwargs: Dict[str, Any] = None, verbose: bool = False,
                       step_size: float = None, overlapping: bool = False) -> SlidingWindowResults:
        """
        Apply function in a sliding window over the series
        @param function: callable to be applied
        @param window_widths: list of window widths to use
        @param step_size: how far apart to space windows
        @param args: positional arguments for the function
        @param kwargs: keyword arguments for the function
        @param verbose: verbose output?
        @param overlapping: whether the windows should overlap
        @return: SlidingWindowResults (see the dataframe attribute for results)
        """
        if args is None:
            args: List[Any] = []
        if kwargs is None:
            kwargs: Dict[str, Any] = {}

        # Validation
        if self.basis is None:
            raise AttributeError('cannot apply sliding window analysis without basis attribute. See index_to_basis()')
        if len(self.basis) != len(self.series):
            raise ValueError('length of basis does not match length of series - something is wrong')

        df: pd.DataFrame = pd.DataFrame()
        for i, window_width in enumerate(window_widths):

            # Determine window sizes
            if step_size:
                starts: List[float] = list(np.arange(min(self.basis), max(self.basis) - step_size, step_size))
            else:
                if overlapping:
                    starts: List[float] = [x for x in self.basis if self.basis[-1] - x >= window_width]
                else:
                    starts: List[float] = list(np.arange(min(self.basis), max(self.basis) - window_width, window_width))

            for j, window_start in enumerate(starts):
                window_data: OrderedSeries = self.slice(start_at=window_start,
                                                        stop_at=window_start + window_width,
                                                        inplace=False)
                evaluation: Dict[str, Any] = function(window_data, *args, **kwargs)
                df = df.append({'window_width': window_width, 'window_start': window_start, **evaluation},
                               ignore_index=True)

                if verbose:
                    logger.info(f"\nWindow width: {window_width} ({i + 1} of {len(window_widths)})\n"
                                f"Starting at: {window_start} ({j + 1} of {len(starts)})")

        return SlidingWindowResults(window_width_field_name='window_width', window_start_field_name='window_start',
                                    dataframe=df)

    ################
    # Visualizations
    ################

    def visualize_x_y(self, y_name: str = 'series', x_name: str = 'basis', cumulative: str = '',
                      figsize: Any = None, title: str = None, log_axes: str = '',
                      types: Union[str, List[str]] = 'scatter') -> plt.Figure:
        """
        Creates a 2D (scatter or line) plot of x and y data
        @param y_name: field or subfield to plot on x-axis
        @param x_name: field or subfield to plot on y-axis
        @param cumulative: 'x' or 'y' or 'xy' to plot cumulative data on that axis
        @param figsize: figure size
        @param title: figure title (raw strings are translated)
        @param log_axes: 'x' or 'y' or 'xy' to plot that axis/axes on a log scale
        @param types: any subset of ['scatter, 'plot']
        @return: figure handle
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
        if 'scatter' in (styles_list := as_list(types)):
            plt.scatter(x_data, y_data, self.markersize, self.color)
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

    def scatter(self, **kwargs) -> plt.Figure:
        """
        Creates a 2D scatter plot of x and y data (wraps visualize_x_y)
        - param y_name: field or subfield to plot on x-axis
        - param x_name: field or subfield to plot on y-axis
        - param cumulative: 'x' or 'y' or 'xy' to plot cumulative data on that axis
        - param figsize: figure size
        - param title: figure title (raw strings are translated)
        - param log_axes: 'x' or 'y' or 'xy' to plot that axis/axes on a log scale
        - return: figure handle
        """
        return self.visualize_x_y(**kwargs, types='scatter')

    def plot(self, **kwargs) -> plt.Figure:
        """
        Creates a 2D line plot of x and y data (wraps visualize_x_y)
        - param y_name: field or subfield to plot on x-axis
        - param x_name: field or subfield to plot on y-axis
        - param cumulative: 'x' or 'y' or 'xy' to plot cumulative data on that axis
        - param figsize: figure size
        - param title: figure title (raw strings are translated)
        - param log_axes: 'x' or 'y' or 'xy' to plot that axis/axes on a log scale
        - return: figure handle
        """
        return self.visualize_x_y(**kwargs, types='plot')

    def hist(self, field_name: str = 'series', bins: Any = None, log_axes: str = '',
             cumulative: bool = False, density: bool = False, title: str = None) -> plt.Figure:
        """
        Plot a histogram of data
        @param field_name: name of field to plot
        @param bins: number of bins or bins edges
        @param log_axes: 'y' to plot log-scale y-axis
        @param cumulative: set True to plot cumulative distribution
        @param density: set True to plot density
        - param title: figure title (raw strings are translated)
        @return:
        """
        figure_handle: plt.Figure = plt.figure(facecolor=self.facecolor, figsize=self.figsize)
        plt.hist((data := self.values(field_name)), color=self.color, bins=bins, cumulative=cumulative, density=density)
        plt.xlabel(self.translate(field_name), fontsize=self.label_fontsize)
        if cumulative:
            ylabel: str = 'counts'
        else:
            ylabel: str = 'cumulative counts'
        if density:
            ylabel += " (density)"
        plt.ylabel("counts", fontsize=self.label_fontsize)

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
