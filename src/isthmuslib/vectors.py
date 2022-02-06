import time as time
from typing import List, Any, Tuple, Callable, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from .config import Style
from .utils import PickleUtils, Rosetta, make_dict
from .data_quality import basis_quality_checks, basis_quality_plots, fill_ratio
from copy import deepcopy
import statsmodels.api as sm
from .plotting import visualize_x_y, visualize_1d_distribution, visualize_surface, visualize_embedded_surface
import pathlib
from pydantic import BaseModel
from sklearn.feature_selection import SelectKBest, chi2


class SVD(BaseModel):
    u: Any
    s: Any
    vh: Any


class VectorMultiset(PickleUtils, Style, Rosetta):
    """ A set of vectors (which may or may not be ordered)"""
    data: pd.DataFrame = None
    name_root: str = None
    svd: SVD = None

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        """ The length of the vector set is the length of the data frame """
        return len(self.data)

    def values(self, feature: str, cumulative: bool = False, *args) -> List[Any]:
        """ Retrieves a particular data feature by attribute name. Additional args unpack deeper

        :param feature: name of feature to retrieve
        :param cumulative: whether to apply a cumulative sum (default = False)
        :return: Extracted data
        """
        values: pd.Series = self.data.loc[:, feature]
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

    def from_dataframe(self, data_frame: pd.DataFrame, inplace: bool = True, **kwargs) -> Union[Any, None]:
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

    def to_csv(self, file_path: Union[str, pathlib.Path] = None) -> None:
        """ Saves the data as a CSV file (note: this drops the name_root)

        :param file_path: path to write the file
        """
        if not file_path:
            file_path: str = f'default_filename_{time.time():.0f}.csv'
        self.data.to_csv(file_path)

    def read_csv(self, file_path: Union[str, pathlib.Path], inplace: bool = True, **kwargs) -> Union[Any, None]:
        """Reads data from a CSV file

        :param file_path: file to read
        :param inplace: import inplace (default) or return the result
        """
        data: pd.DataFrame = pd.read_csv(file_path)
        if inplace:
            self.data = data
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        else:
            return self.__class__(data=data, **kwargs)

    def to_feather(self, file_path: Union[str, pathlib.Path], **kwargs) -> None:
        """ Saves the data as a feather file (note: this drops the name_root). keyword

        :param file_path: path to write the file
        """
        self.data.to_feather(file_path, **kwargs)

    def read_feather(self, file_path: Union[str, pathlib.Path], inplace: bool = True, **kwargs) -> Union[Any, None]:
        """Reads data from a CSV file

        :param file_path: file to read
        :param inplace: import inplace (default) or return the result
        """
        data: pd.DataFrame = pd.read_feather(file_path)
        if inplace:
            self.data = data
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        else:
            return self.__class__(data=data, **kwargs)

    ################
    # Visualizations
    ################

    def visualize_x_y(self, x_name: str, y_name: str, cumulative: Union[str, List[str]] = '', **kwargs) -> plt.Figure:
        """ Visualize in two dimensions

        :param x_name: name of the x-axis data feature
        :param y_name: name of the y-axis data feature
        :param cumulative: Which (if any) dimensions should be cumulative, e.g. 'x' or ['x','y'] or 'xy'
        :param kwargs: additional kwargs for isthmuslib.visualize_x_y, passed through to matplotlib.pyplot.scatter()
        :return: figure handle for the plot
        """
        x_data: List[Any] = self.values(x_name, cumulative='x' in cumulative)
        y_data: List[Any] = self.values(y_name, cumulative='y' in cumulative)
        kwargs.setdefault('xlabel', self.translate(x_name))
        kwargs.setdefault('ylabel', self.translate(y_name))
        kwargs.setdefault('title', self.translate(self.name_root))
        return visualize_x_y(x_data, y_data, style=Style(**self.dict()), **kwargs)

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

    def hist(self, col_name: str, **kwargs) -> plt.Figure:
        """ Plot a histogram of data. Useful kwargs: [cumulative, density, bins]

        :param col_name: name of field to plot
        :return: figure handle
        """
        data: List[Any] = self.values(col_name)
        kwargs.setdefault('xlabel', self.translate(col_name))
        kwargs.setdefault('title', self.translate(self.name_root))
        return visualize_1d_distribution(data, style=Style(**self.dict()), **kwargs)

    def surface(self, x_name: str, y_name: str, z_name: str, **kwargs) -> plt.Figure:
        """ Plot a surface from the data. Useful kwargs: [cumulative, density, bins]

        :param x_name: name of the x-axis data feature
        :param y_name: name of the y-axis data feature
        :param z_name: name of the z-axis data feature (color)
        :param kwargs: additional keyword arguments for visualize_surface which passes to seaborn heatmap()
        :return: figure handle for the plot
        """
        if isinstance(self, VectorSequence) and (not x_name):
            x_name: str = self.basis_col_name
        kwargs.setdefault('xlabel', self.translate(x_name, missing_response='return_input'))
        kwargs.setdefault('ylabel', self.translate(y_name, missing_response='return_input'))
        kwargs.setdefault('title', self.translate(z_name, missing_response='return_input'))
        return visualize_surface(self.data.loc[:, x_name], self.data.loc[:, y_name], self.data.loc[:, z_name], **kwargs)

    def visualize_embedded_surface(self, x_col_name: str, y_col_name: str, z_col_name: str, **kwargs) -> plt.Figure:
        """ Plots a surface (2D embedded in 3D) based on vector features"""
        return visualize_embedded_surface(self.data.loc[:, x_col_name], self.data.loc[:, y_col_name],
                                          self.data.loc[:, z_col_name], **kwargs)

    ################
    # Statistics
    ################

    def correlation_matrix(self, **kwargs) -> pd.DataFrame:
        """
        Very thin wrapper around correlation matrix (using pandas corr() method)

        :param kwargs: additional keyword arguments for correlation_matrix (fed through to df.style.background_gradient)
        :return: styled pandas dataframe
        """
        return correlation_matrix(self.data, **kwargs)

    def singular_value_decomposition(self, cols: Union[str, List[str]] = None,
                                     cache_results: bool = False,
                                     **kwargs) -> SVD:
        """ Applies singular value decomposition

        :param cache_results: if True, saves u, s, and vh as the attribute 'svd' (an SVD class)
        :param cols: which data features to use
        :param kwargs: keyword arguments for SVD
        :return: u, s, vh arrays
        """
        if not cols:
            cols = self.data.keys().tolist()
        svd: SVD = singular_value_decomposition(self.data, cols, **kwargs)
        if cache_results:
            self.svd = svd
        return svd

    def to_surface_df(self, group_by_col_names: List[str], aggregation_method: str = 'max', **kwargs) -> pd.DataFrame:
        """ Thin wrapper that aggregates the data to a surface, using pandas methods like mean, median, max, etc

        :param group_by_col_names: axes for the surface (i.e. how to bin the data)
        :param aggregation_method: how to combine multiple data points in the same bin
        :param kwargs: additional keyword arguments for pandas groupby method
        :return: data frame with combined according `aggregation_method` into bins of `group_by_col_names`
        """
        return getattr(self.data.groupby(by=group_by_col_names, **kwargs), aggregation_method)().reset_index(drop=False)

    def plot_projection_to_surface(self, surface_axes_names: Union[List[str], Tuple[str, str]], z_axis_name: str,
                                   aggregation_method: str = 'max', **kwargs) -> plt.Figure:
        if len(surface_axes_names) != 2:
            raise ValueError(f"plot_projection is intended to be used with 2D surfaces")
        df_surface: pd.DataFrame = self.to_surface_df(group_by_col_names=surface_axes_names,
                                                      aggregation_method=aggregation_method,
                                                      **kwargs)
        return visualize_x_y(df_surface.loc[:, surface_axes_names[0]],
                             df_surface.loc[:, surface_axes_names[1]],
                             c=df_surface.loc[:, z_axis_name].tolist(),
                             xlabel=self.translate(surface_axes_names[0]),
                             ylabel=self.translate(surface_axes_names[1]),
                             colorbar_label=self.translate(z_axis_name))

    def feature_selection_univariate(self, target_feature_name: str, input_feature_names: List[str] = None,
                                     k_best: int = 3, normalize: bool = True, **kwargs) -> any:
        """ Feature selection of k best using univariate methods.

        :param target_feature_name: column name of the target feature
        :param input_feature_names: column names of the input features
        :param k_best: how many features to return
        :param normalize: if True, scales input features to unit mean
        :param kwargs: additional keyword arguments passed through to scikit-learn SelectKBest
        :return: trimmed input feature data set
        """
        if not input_feature_names:
            input_feature_names = [x for x in self.data.keys() if x != target_feature_name]
        input_feature_data = self.data.loc[:, input_feature_names]
        if normalize:
            for fieldname in input_feature_data.keys():
                std_dev: float = float(np.std(this_field_data := self.data.loc[:, fieldname]))
                input_feature_data[fieldname] = [x / std_dev for x in this_field_data]

        target_feature_data = self.data.loc[:, target_feature_name].to_numpy()
        return SelectKBest(chi2, k=k_best).fit_transform(input_feature_data, target_feature_data, **kwargs)


class SlidingWindowResults(VectorMultiset):
    """ Results from Sequence.sliding_window(), which is a VectorMultiset with extra context baked in """
    window_width_col_name: str = 'window_width'
    window_start_col_name: str = 'window_start'

    def group_by_window_width(self) -> Dict[float, pd.DataFrame]:
        """ Helper function that returns individual dataframes for each window width """
        return {x: self.data.loc[self.data.loc[:, 'window_width'] == x, :] for x in set(self.data.window_width)}

    def plot_results(self, col_name: str, legend_override: List[str] = None, **kwargs) -> plt.Figure:
        """ Plot any sequence (based on the data frame column name)

        :param col_name: dataframe column to plot
        :param legend_override: specify custom legend keys if desired
        :return: figure handle
        """
        x_arrays: List[Any] = []
        y_arrays: List[Any] = []
        labels: List[Any] = []

        # For each `window_width` add we'll add a trace to the plot
        group_by_window_width: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key in sorted(group_by_window_width.keys()):  # looping over group_by_window_width.items() gives wrong order
            value: pd.DataFrame = deepcopy(group_by_window_width[key])
            value.sort_values(by=self.window_start_col_name, ascending=True, ignore_index=True, inplace=True)
            x_arrays.append(value.loc[:, self.window_start_col_name])
            y_arrays.append(value.loc[:, col_name])
            labels.append(str(key))  # noqa: key is a string

        # Final settings and overrides
        if legend_override:
            labels: List[str] = legend_override
        kwargs.setdefault('types', ['scatter', 'plot'])
        kwargs.setdefault('xlabel', self.translate(self.translate('window_start')))
        kwargs.setdefault('ylabel', self.translate(col_name))
        kwargs.setdefault('title', self.translate(self.name_root))
        return visualize_x_y(x_arrays, y_arrays, legend_strings=labels, style=Style(**self.dict()), **kwargs)

    def plot_pdfs(self, col_name: str, legend_override: List[str] = None, **kwargs) -> plt.Figure:
        """ Plot the probability density function(s) of the sliding window results. Useful kwargs: [cumulative, density]

        :param col_name: which field should be plotted
        :param legend_override: custom legend labels if desired (must have same length as number of unique widths)
        :param kwargs: plt.hist() keyword arguments.
        :return: figure handle
        """
        grouped: Dict[float, pd.DataFrame] = self.group_by_window_width()
        data_sets: List[Any] = []
        labels: List[Any] = []
        for key, value in grouped.items():
            data_sets.append(value[col_name])
            labels.append(str(key))
        if legend_override:
            labels: List[str] = legend_override
        kwargs.setdefault('xlabel', self.translate(col_name))
        kwargs.setdefault('title', self.translate(self.name_root))
        return visualize_1d_distribution(data_sets, legend_strings=labels, **kwargs)

    def heatmap_feature(self, col_name: str, **kwargs) -> plt.Figure:
        return self.surface(self.window_start_col_name, self.window_width_col_name, col_name, **kwargs)


class InfoSurface(SlidingWindowResults):
    """ Wrapper for SlidingWindowResults that knows how to plot the infosurface """

    def plot_info_surface(self, singular_values: List[int] = None, **kwargs) -> List[plt.Figure]:
        """ Plot the info surface showing value of singular vectors as a function of window start and width

        :param singular_values: Which singular values to plot [1, 2, 3] by default
        :return: figure handles
        """
        figure_handles: List[plt.Figure] = []
        if not singular_values:
            singular_values: List[int] = [1, 2, 3]
        for s in singular_values:
            figure_handles.append(self.heatmap_feature(f"singular_value_{s}", title=f'Singular value # {s}', **kwargs))
        return figure_handles


class VectorSequence(VectorMultiset):
    """ Set of vectors in a sequence ordered according to some basis (e.g. height, wavelength, time) """

    # (Inherits `data` dataframe and `name_root` string from VectorMultiset)
    basis_col_name: str = 'basis'
    error_if_basis_quality_issues: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.data is not None:
            self.sort(by=self.basis_col_name, inplace=True, reset_index=True)
            if self.error_if_basis_quality_issues:
                is_ok, explanation = self.passes_basis_quality_checks()
                if not is_ok:
                    raise ValueError(f"Issues with basis:\n{explanation}\nSee: `error_if_basis_quality_issues`")

    def basis_quality_checks(self) -> Tuple[bool, str]:
        """ Checks basis data quality and returns both a True/False flag and a string with an explanation

        :return: True if OK, and a string with explanation either way
        """
        return basis_quality_checks(self.data.loc[:, self.basis_col_name])

    def passes_basis_quality_checks(self) -> bool:
        """ Super thin wrapper around basis_quality_checks that drops the explanation string

        :return: True if OK
        """
        result, _ = self.basis_quality_checks()
        return result

    def basis_quality_plots(self) -> List[plt.Figure]:
        """ Creates a series of plots showing the basis data quality (missing data, nans, timeseries gaps, etc)

        :return: figure handles for several plots
        """
        return basis_quality_plots(self.data.loc[:, self.basis_col_name])

    def fill_ratio(self) -> float:
        """ Checks for how many data points are observed for the expected number of data points in a uniform basis.
            Attempts to infer spacing from median of diffs. If used on non-uniform data the result is not meaningful.

        Use example: [10, 20, 40, 50] has 0.8 fill ratio because it is missing the 30 to complete the set

        :return: fractional amount of the array
        """
        return fill_ratio(self.data.loc[:, self.basis_col_name])

    def slice(self, start_at: Any = None, stop_at: Any = None, inplace: bool = False, reset_index: bool = True):
        """ Slices the VectorSequence according to the basis

        :param start_at: Start point (can be None)
        :param stop_at: Stop point (can be None)
        :param inplace: slice inplace or return the result (default --> return)
        :param reset_index: indicate whether or not to reset the data frame index (default --> True)
        :return: VectorSequence if not inplace
        """
        df: pd.DataFrame = deepcopy(self.data)
        df.sort_values(by=self.basis_col_name, ascending=True, inplace=True, ignore_index=True)
        if not start_at:
            start_at: Any = min(df.loc[:, self.basis_col_name])
        if not stop_at:
            stop_at: Any = max(df.loc[:, self.basis_col_name])
        in_range: pd.DataFrame = df.loc[df.loc[:, self.basis_col_name].between(start_at, stop_at), :]
        if reset_index:
            in_range.reset_index(drop=True, inplace=True)
        if inplace:
            self.data = in_range
        else:
            return VectorSequence(basis_col_name=self.basis_col_name, name_root=self.name_root, data=in_range)

    def index_to_basis(self) -> None:
        """  Adds (or overwrites) 'basis' column with a default index [0, 1, 2, ...] """
        self.data.loc[:, self.basis_col_name] = list(range(len(self.data)))

    def sort(self, by: str = None, inplace: bool = True, reset_index: bool = True):
        """ Sorts the data points by the basis

        :param inplace: import inplace (default) or return the result
        :param reset_index: indicate whether or not to reset the data frame index (default = True)
        :param by: which field to use for the sort (uses the basis by default)
        :return: sorted data frame, if inplace is False
        """
        if not by:
            by = self.basis_col_name
        result: pd.DataFrame = self.data.sort_values(by=by, ascending=True, inplace=False, ignore_index=False)
        if reset_index:
            result.reset_index(drop=True, inplace=True)
        if inplace:
            self.data = result
        else:
            return VectorSequence(basis_col_name=self.basis_col_name, name_root=self.name_root, data=result)

    def sliding_window(self, function: Callable[[Any, List[Any], Dict[str, Any]], Dict[str, Any]],
                       window_widths: List[float] = None, window_starts: List[Any] = None, step_size: float = None,
                       overlapping: bool = False, verbose: bool = False, *args, **kwargs) -> SlidingWindowResults:
        """ Apply function in a sliding window over the sequence

        :param function: callable to be applied
        :param window_widths: list of window widths to use
        :param window_starts: list of starting points for the windows
        :param step_size: how far apart to space windows
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        :param verbose: verbose output?
        :param overlapping: whether the windows should overlap
        :return: SlidingWindowResults (see the dataframe attribute for results)

        Note: `window_starts` overrides `step_size` overrides `overlapping`
        """

        # Validation
        if not self.basis_col_name:
            raise AttributeError('cannot apply sliding window analysis without knowing which field is the basis')
        if self.basis_col_name not in self.data.keys():
            raise AttributeError(f'basis_col_name {self.basis_col_name} not a column in data frame:\n{self.data}')

        df: pd.DataFrame = pd.DataFrame()
        basis: Tuple[float] = (self.data.loc[:, self.basis_col_name].tolist())

        if not window_widths:
            duration: float = max(self.data.loc[:, self.basis_col_name]) - min(self.data.loc[:, self.basis_col_name])
            window_widths: List[float] = [duration / x for x in range(20, 401, 20)]

        # Loop over window widths
        for i, window_width in enumerate(window_widths):
            # First, figure out where to place the windows:
            if not window_starts:
                if step_size:
                    window_starts: List[float] = list(np.arange(min(basis), max(basis) - window_width, step_size))  # !
                else:
                    if overlapping:
                        window_starts: List[float] = [x for x in basis if basis[-1] - x >= window_width]
                    else:
                        window_starts: List[float] = list(
                            np.arange(min(basis), max(basis) - window_width, window_width))

            # Loop over window starts
            for j, window_start in enumerate(window_starts):
                window_data: VectorSequence = self.slice(start_at=window_start, stop_at=window_start + window_width,
                                                         inplace=False, reset_index=True)
                evaluation: Dict[str, Any] = function(window_data, *args, **kwargs)
                new_entry: Dict[Any, Any] = {'window_width': window_width, 'window_start': window_start, **evaluation}
                df = pd.concat([df, pd.DataFrame(new_entry, index=[0])], ignore_index=True)

                if verbose:
                    logger.info(f"\nWindow width: {window_width} ({i + 1} of {len(window_widths)})\n"
                                f"Starting at: {window_start} ({j + 1} of {len(window_starts)})")

        return SlidingWindowResults(window_width_col_name='window_width', window_start_col_name='window_start',
                                    data=df, name_root=self.name_root)

    def repackage(self, instance: Any, sequence_attribute: str = 'series', basis_name: str = 'basis') -> Any:
        cache: Dict[str, Any] = instance.dict()
        cache['data'] = pd.DataFrame({x: input.__getattribute__(x) for x in [sequence_attribute, basis_name]})
        cache['basis_col_name'] = basis_name
        return self.__class__(**cache)

    def seasonal_decompose(self, col: str, period: float, **kwargs):
        """ Seasonal (weekly, monthly, etc) decomposition, must specify the period

        :param period: period of the data
        :param col: which data feature to use
        :param kwargs: additional kwargs for statsmodels.tsa.seasonal_decompose
        :return: sm.tsa.seasonal.DecomposeResult
        """
        x: pd.DataFrame = deepcopy(self.data.loc[:, col])
        try:
            return sm.tsa.seasonal_decompose(x, period=period, **kwargs)
        except ValueError as e:
            raise ValueError(f"Error - did you include a non-numeric column, or not specify 'cols'? See: {e}")

    def plot_decomposition(self, col: str, period: float, figsize: Tuple[float, float] = None,
                           which_plots: List[str] = None, xlabel: str = 'basis', ylabel: str = '[units]',
                           title: str = None, xlim: List[float] = None, **kwargs) -> List[plt.Figure]:

        """ Plot the seasonal (weekly, monthly, etc) decomposition. Must specify period

        :param period: period of the data
        :param col: which data feature to use
        :param figsize: size of each subfigure
        :param which_plots: which plots to make
        :param xlabel: how to label the x-axis
        :param ylabel: how to label the y-axis (e.g. units)
        :param title: title string (prefix) for the plots
        :param xlim: x-axis limits (Left, Right)
        :param kwargs: additional kwargs for statsmodels.tsa.seasonal_decompose
        :return: list of figure handles
        """
        if not figsize:
            figsize: List[float] = [8.0, 3.0]
        if not which_plots:
            which_plots: List[str] = ['observed', 'trend', 'seasonal', 'residual']
        if not title:
            title: str = self.translate(self.name_root)

        def add_labels(x_label: str, y_label: str) -> None:
            plt.xlabel(x_label, size=self.label_fontsize)
            plt.ylabel(y_label, size=self.label_fontsize)
            plt.legend(fontsize=self.legend_fontsize)

        decomposition = self.seasonal_decompose(col, period, **kwargs)
        figure_handles: List[plt.Figure] = []

        if 'observed' in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(self.data.loc[:, self.basis_col_name], decomposition.observed, '-', label='Sequence',
                     color='black')
            add_labels(xlabel, ylabel)
            plt.title(f'{title}observed', size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        if 'trend' in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(self.data.loc[:, self.basis_col_name], decomposition.trend, '-', label='Trend', color='green')
            add_labels(xlabel, ylabel)
            plt.title(f'{title}trend', size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        if 'seasonal' in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(self.data.loc[:, self.basis_col_name], decomposition.seasonal, '-', label='Seasonality',
                     color='darkslateblue')
            add_labels(xlabel, ylabel)
            plt.title(f"{title}seasonality with period {period}", size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        if 'residual' in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(self.data.loc[:, self.basis_col_name], decomposition.resid, '-', label='Residual', color='darkred')
            add_labels(xlabel, ylabel)
            plt.title(f'{title}residual', size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        return figure_handles

    def singular_value_decomposition(self, cols: Union[str, List[str]] = None, cache_results: bool = False,
                                     **kwargs) -> SVD:
        """ Applies singular value decomposition, skipping the basis column

        :param cache_results: if True, saves u, s, and vh as the attribute 'svd' (an SVD class)
        :param cols: which data features to use
        :param kwargs: keyword arguments for SVD
        :return: u, s, vh arrays
        """
        if not cols:
            cols = [x for x in self.data.keys().tolist() if x != self.basis_col_name]
        svd: SVD = singular_value_decomposition(self.data, cols, **kwargs)
        if cache_results:
            self.svd = svd
        return svd

    def calculate_info_surface(self, window_widths: List[float] = None, cols: Union[str, List[str]] = None, *args,
                               **kwargs) -> InfoSurface:
        """ Calculates the info surface by sliding the SVD function along the basis

        :param window_widths: window widths for the sliding window analysis
        :param cols: which data features to use in the svd
        :param args: args for sliding window analysis and eval function
        :param kwargs: kwargs for sliding window analysis and eval function
        :return: InfoSurface object
        """
        result: SlidingWindowResults = self.sliding_window(info_surface_slider, window_widths=window_widths, cols=cols,
                                                           *args, **kwargs)
        return InfoSurface(**result.dict())

    def plot_info_surface(self, window_widths: List[float] = None, cols: Union[str, List[str]] = None,
                          singular_values: List[int] = None, style: Style = None, *args, **kwargs) -> List[plt.Figure]:
        """ Calculates and plots the info surface: value of singular vectors as a function of window start and width

        :param window_widths: window widths for the sliding window analysis
        :param cols: which data features to use in the svd
        :param singular_values: singular values for which to plot the surface (default: [1, 2, 3])
        :param args: args for sliding window analysis and eval function
        :param kwargs: kwargs for sliding window analysis and eval function
        :param style: isthmuslib Style object for the colormap
        :return: list of figure handles
        """
        # Set style. Overrides: kwargs > style input > Style() defaults

        config: Style = Style(**{**Style().dict(), **make_dict(style), **make_dict(kwargs)})
        svd_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k not in config.dict()}
        style_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in config.dict()}

        result: InfoSurface = self.calculate_info_surface(window_widths=window_widths, cols=cols, *args, **svd_kwargs)
        return result.plot_info_surface(singular_values=singular_values, **{**make_dict(style), **make_dict(kwargs)})

    def correlation_matrix(self, exclude_basis: bool = True, **kwargs) -> pd.DataFrame:
        """
        Very thin wrapper around correlation matrix (using pandas corr() method)

        :param exclude_basis: whether to exclude the basis column from correlation analysis
        :param kwargs: additional keyword arguments for correlation_matrix (fed through to df.style.background_gradient)
        :return: styled pandas dataframe
        """
        if exclude_basis:
            if kwargs_in := kwargs.get('exclude_cols'):
                if self.basis_col_name not in kwargs_in:
                    kwargs['exclude_cols'] = kwargs_in + [self.basis_col_name]
            else:
                kwargs: Dict[str, Any] = {'exclude_cols': self.basis_col_name}
        return correlation_matrix(self.data, **kwargs)


################
# Helpers
################


def auto_extract_from_text(input_string: str, return_type: str = 'dataframe', record_delimiter: str = '[@@@]',
                           left: str = '[<<', key_value_delimiter: str = '=', right: str = '>>]',
                           basis_col_name: str = None) -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
    """
    Extracts a data frame from a string

    Input: "the [@@@] quick [<<x='foo'>>] brown[<<y=9>>] [@@@]fox [<<y=93>>]"
    Output: (data frame)
                   x   y
            0  'foo'   9
            1    NaN  93

    Hint, write your logs using python fstrings like f"Foo [<<{x=}>>] and [<<{y=}>>]"

    :param input_string: string to be parsed
    :param record_delimiter: substring between rows
    :param left: left side of a record
    :param key_value_delimiter: marker between name and
    :param right: right side of a record
    :param basis_col_name: optional, specify a basis column name for a VectorSequence (otherwise ignored)
    :param return_type: what format to return the data in
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """
    df_output: pd.DataFrame = pd.DataFrame()
    # Loop over rows
    for i, chunk in enumerate(input_string.split(record_delimiter)[1:]):
        chunk_buffer: Dict[str, Any] = dict()
        raw_breaks: List[str] = chunk.split(left)[1:]
        middle: List[str] = [x.split(right)[0] for x in raw_breaks]
        # Loop over possible tokens
        for entry in middle:
            if key_value_delimiter not in entry:
                continue  # Do not attempt to parse if the delimiter is not present
            substrings: List[str] = entry.split(key_value_delimiter)
            key: str = substrings[0]
            if not key:
                continue  # Do not continue to parse if the kee is known
            value: str = ''.join([x + key_value_delimiter for x in substrings[1:]])[:-1]
            chunk_buffer.setdefault(key, value)  # Add the value to this row
        if chunk_buffer:
            df_output = pd.concat([df_output, pd.DataFrame(chunk_buffer, index=[-1])], ignore_index=True)
    if 'dataframe' in return_type.lower():
        return df_output
    elif 'multiset' in return_type.lower():
        return VectorMultiset(data=df_output)
    elif 'sequence' in return_type.lower():
        return VectorSequence(data=df_output, basis_col_name=basis_col_name)


def auto_extract_from_file(file_path: Union[str, pathlib.Path], record_delimiter: str = '[@@@]', left: str = '[<<',
                           key_value_delimiter: str = '=', right: str = '>>]', basis_col_name: str = None,
                           return_type: str = 'dataframe') -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
    """
    Extracts a data frame from a file

    Input: "the [@@@] quick [<<x='foo'>>] brown[<<y=9>>] [@@@]fox [<<y=93>>]"
    Output: (data frame)
                   x   y
            0  'foo'   9
            1    NaN  93

    Hint, write your logs using python fstrings like f"Foo [<<{x=}>>] and [<<{y=}>>]"

    :param file_path: file to read
    :param record_delimiter: substring between rows
    :param left: left side of a record
    :param key_value_delimiter: marker between name and
    :param right: right side of a record
    :param basis_col_name: optional, specify a basis column name for a VectorSequence (otherwise ignored)
    :param return_type: what format to return the data in
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """
    with open(file_path, 'r') as f:
        input_string: str = f.read()
    return auto_extract_from_text(input_string=input_string, return_type=return_type, record_delimiter=record_delimiter,
                                  left=left, key_value_delimiter=key_value_delimiter, right=right,
                                  basis_col_name=basis_col_name)


def extract_text_to_dataframe(input_string: str, tokens_dictionary: Dict[str, Tuple[str, str]],
                              record_delimiter: str = '[@@@]') -> pd.DataFrame:
    """ Extracts a pandas dataframe from a string

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :return: Vector data set extracted from the input string
    """
    df_output: pd.DataFrame = pd.DataFrame()
    for i, chunk in enumerate(input_string.split(record_delimiter)):
        chunk_buffer: Dict[str, Any] = dict()
        for key, (before_token, after_token) in tokens_dictionary.items():
            if before_token in chunk:
                target_chunk = chunk.split(before_token)[1]
                if after_token in target_chunk:
                    chunk_buffer.setdefault(key, target_chunk.split(after_token)[0])
        if chunk_buffer:
            df_output = pd.concat([df_output, pd.DataFrame(chunk_buffer, index=[-1])], ignore_index=True)
    return df_output


def extract_text_to_vector(input_string: str, tokens_dictionary: Dict[str, Tuple[str, str]],
                           record_delimiter: str = '[@@@]',
                           basis_col_name: str = None) -> Union[VectorMultiset, VectorSequence]:
    """ Extracts a VectorMultiset from a string (or a VectorSequence if you specify `basis_col_name`)

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :return: Vector data set extracted from the input string
    """
    df_output: pd.DataFrame = extract_text_to_dataframe(input_string=input_string, record_delimiter=record_delimiter,
                                                        tokens_dictionary=tokens_dictionary)
    if basis_col_name:
        return VectorSequence(data=df_output, basis_col_name=basis_col_name)
    else:
        return VectorMultiset(data=df_output)


def extract_file_to_vector(file_path: Union[str, pathlib.Path], record_delimiter: str,
                           tokens_dictionary: Dict[str, Tuple[str, str]],
                           basis_col_name: str = None) -> Union[VectorSequence, VectorMultiset]:
    """ Wrapper for extract_from_text that pulls vector data out of a file, such as raw log output

    :param file_path: file to read
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :return: Vector data set extracted from the input string
    """
    with open(file_path, 'r') as f:
        return extract_text_to_vector(input_string=f.read(), record_delimiter=record_delimiter,
                                      tokens_dictionary=tokens_dictionary, basis_col_name=basis_col_name)


def correlation_matrix(dataframe: pd.DataFrame, use_cols: List[str] = None, exclude_cols: List[str] = None,
                       correlation_method: str = 'pearson', style: Style = None, **kwargs) -> pd.DataFrame:
    """
    Returns a styled pandas data frame with correlation coefficients (Pearson by default), wraps pandas.corr()

    :param dataframe: dataframe to analyze
    :param use_cols: which columns to use
    :param exclude_cols: which columns to exclude (NB: exclude overrides include)
    :param correlation_method: Method of correlation {‘pearson’, ‘kendall’, ‘spearman’} or Callable
    :param style: isthmuslib Style object for the colormap
    :param kwargs: additional keyword arguments for df.style.background_gradient()
    :return: styled pandas dataframe
    """
    if not use_cols:
        use_cols: List[str] = dataframe.keys().tolist()
    if exclude_cols:
        use_cols: List[str] = [x for x in use_cols if x not in exclude_cols]
    if style:
        kwargs.setdefault('cmap', style.cmap)
    else:
        kwargs.setdefault('cmap', Style().cmap)
    corr: pd.DataFrame = dataframe.loc[:, use_cols].corr(method=correlation_method)
    return corr.style.background_gradient(**kwargs)  # noqa: misinterprets type


def info_surface_slider(vs: VectorSequence, *args, **kwargs) -> Dict[str, Any]:
    """ Helper function for sliding window SVD analysis

    :param vs: vector sequence
    :param args: args for svd
    :param kwargs: kwargs for svd
    :return: dictionary with singular values
    """
    svd: SVD = vs.singular_value_decomposition(*args, **kwargs)
    return {f"singular_value_{i + 1}": value for i, value in enumerate(svd.s)}


def singular_value_decomposition(df: pd.DataFrame, cols: Union[str, List[str]] = None, **kwargs) -> SVD:
    """ Helper function that wraps numpy svd to feed in a subset of data features (see kwarg: 'full_matrices')

    :param df: pandas dataframe to analyze
    :param cols: which data features to use
    :param kwargs: keyword arguments for SVD
    :return: u, s, vh arrays
    """
    u, s, vh = np.linalg.svd(deepcopy(df.loc[:, cols]), **kwargs)
    return SVD(u=u, s=s, vh=vh)


class OrderedSeries:
    def __init__(self, **data: Any):
        super().__init__(**data)
        raise TypeError("Notice: OrderedSeries class has been supplanted by VectorSequence, see repackage() method")
