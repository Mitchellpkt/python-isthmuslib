import time as time
from typing import List, Any, Tuple, Callable, Dict, Union, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from matplotlib.patches import Rectangle
from .config import Style
from .utils import (
    PickleUtils,
    Rosetta,
    make_dict,
    get_num_workers,
    process_queue,
    dict_pretty,
)
from .data_quality import basis_quality_checks, basis_quality_plots, fill_ratio
from copy import deepcopy
import statsmodels.api as sm
from .plotting import (
    visualize_x_y,
    visualize_1d_distribution,
    visualize_surface,
    visualize_embedded_surface,
    visualize_hist2d,
    visualize_x_y_input_interpreter,
)
import pathlib
from pydantic import BaseModel
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.auto import tqdm
import matrixprofile
import math


class SVD(BaseModel):
    u: Any
    s: Any
    vh: Any


class VectorMultiset(PickleUtils, Style, Rosetta):
    """A set of vectors (which may or may not be ordered)"""

    data: Any = None  # needs to be dataframe or path/str to CSV file
    name_root: str = None
    svd: SVD = None

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        """The length of the vector set is the length of the data frame"""
        return len(self.data)

    def __add__(self, other):
        """Adding items derived from MultiSets concatenates their dataframes (keeping metadat from left side)"""
        to_return = deepcopy(self)
        to_return.data = pd.concat([self.data, other.data])
        return to_return

    def head(self, *args, **kwargs) -> pd.DataFrame:
        """For convenience, alias to pandas method DataFrame.head()"""
        return self.data.head(*args, **kwargs)

    def tail(self, *args, **kwargs) -> pd.DataFrame:
        """For convenience, alias to pandas method DataFrame.tail()"""
        return self.data.tail(*args, **kwargs)

    def values(self, feature: str, cumulative: bool = False, *args) -> List[Any]:
        """Retrieves a particular data feature by attribute name. Additional args unpack deeper

        :param feature: name of feature to retrieve
        :param cumulative: whether to apply a cumulative sum (default = False)
        :return: Extracted data
        """
        if feature not in self.data.keys():
            raise ValueError(
                f"Feature {feature} not in known keys: {self.data.keys().tolist()}"
            )
        values: pd.Series = self.data.loc[:, feature]
        if args:
            for unpack_next in args:
                values: List[Any] = [x.__getattribute__(unpack_next) for x in values]
        if cumulative:
            return np.cumsum(values).tolist()
        return values.tolist()

    def dict_pretty(self, max_length: int = 32, previews: bool = False) -> str:
        return dict_pretty(
            self.dict(),
            max_length=self.dict_pretty_max_length,
            previews=self.dict_pretty_previews,
        )

    ################
    # I/O
    ################

    def to_df(self) -> pd.DataFrame:
        """Trivial helper function

        :return: hands back the self.data data frame
        """
        return self.data

    def from_dataframe(
        self, data_frame: pd.DataFrame, inplace: bool = True, **kwargs
    ) -> Union[Any, None]:
        """Makes an VectorMultiset from a pandas data frame

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
        """Saves the data as a CSV file (note: this drops the name_root)

        :param file_path: path to write the file
        """
        if not file_path:
            file_path: str = f"default_filename_{time.time():.0f}.csv"
        self.data.to_csv(file_path)

    def read_csv(
        self, file_path: Union[str, pathlib.Path], inplace: bool = True, **kwargs
    ) -> Union[Any, None]:
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
        """Saves the data as a feather file (note: this drops the name_root). keyword

        :param file_path: path to write the file
        """
        self.data.to_feather(file_path, **kwargs)

    def read_feather(
        self, file_path: Union[str, pathlib.Path], inplace: bool = True, **kwargs
    ) -> Union[Any, None]:
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

    def visualize_x_y(
        self,
        x: Union[str, List[Any], Any],
        y: Union[str, List[Any], Any],
        cumulative: Union[str, List[str]] = "",
        **kwargs,
    ) -> plt.Figure:
        """Visualize in two dimensions

        :param x: name of the x-axis data feature (or the data itself)
        :param y: name of the y-axis data feature (or the data itself)
        :param cumulative: Which (if any) dimensions should be cumulative, e.g. 'x' or ['x','y'] or 'xy'
        :param kwargs: additional kwargs for isthmuslib.visualize_x_y, passed through to matplotlib.pyplot.scatter()
        :return: figure handle for the plot
        """
        if isinstance(x, str):
            x_data: List[Any] = self.values(x, cumulative="x" in cumulative)
            kwargs.setdefault("xlabel", self.translate(x))
        else:
            x_data: Any = x

        if isinstance(y, str):
            y_data: List[Any] = self.values(y, cumulative="y" in cumulative)
            kwargs.setdefault("ylabel", self.translate(y))
        else:
            y_data: Any = y

        kwargs.setdefault("title", self.translate(self.name_root))
        return visualize_x_y(x_data, y_data, **kwargs)

    def viz2d(self, *args, **kwargs) -> plt.Figure:
        """
        For convenience, allow direct access to visualize_x_y_input_interpreter

        :param args: visualize_x_y_input_interpreter positional inputs
        :param kwargs: visualize_x_y_input_interpreter keyword argument inputs
        :return: figure handle
        """
        kwargs.setdefault("style", Style(**self.dict()))
        return visualize_x_y_input_interpreter(*args, **kwargs)

    def scatter(self, *args, **kwargs) -> plt.Figure:
        """Creates a 2D scatter plot of x and y data (wraps visualize_x_y)"""
        if isinstance(self, VectorSequence) and (len(args) == 1):
            args: tuple = (
                self.basis_col_name,
                args[0],
            )  # sequences we can infer the basis for the x-axis
        return self.visualize_x_y(types="scatter", *args, **kwargs)

    def plot(self, *args, **kwargs) -> plt.Figure:
        """Creates a 2D line plot of x and y data (wraps visualize_x_y)"""
        if isinstance(self, VectorSequence) and (len(args) == 1):
            args: tuple = (
                self.basis_col_name,
                args[0],
            )  # sequences we can infer the basis for the x-axis
        if isinstance(self, SlidingWindowResults) and (len(args) == 1):
            raise ValueError(
                f"Plot requires x & y. Hint: You might want SlidingWindowResults.plot_results()"
            )
        return self.visualize_x_y(types="plot", *args, **kwargs)

    def hist(self, col_name: str, **kwargs) -> plt.Figure:
        """Plot a histogram of data. Useful kwargs: [cumulative, density, bins]

        :param col_name: name of field to plot
        :return: figure handle
        """
        data: List[Any] = self.values(col_name)
        kwargs.setdefault("xlabel", self.translate(col_name))
        kwargs.setdefault("title", self.translate(self.name_root))
        return visualize_1d_distribution(data, style=Style(**self.dict()), **kwargs)

    def hist2d(self, *args, **kwargs) -> plt.Figure:
        """Creates a 2D histogram plot of x and y data (wraps visualize_hist2d)"""
        if isinstance(self, VectorSequence) and (len(args) == 1):
            args: tuple = (
                self.basis_col_name,
                args[0],
            )  # sequences we can infer the basis for the x-axis

        # If the first argument is a string, replace it with column data
        if isinstance(args[0], str):
            kwargs.setdefault("xlabel", self.translate(args[0]))
            args = self.data.loc[:, args[0]].tolist(), *args[1:]

        # If the second argument is a string, replace it with column data
        if isinstance(args[1], str):
            kwargs.setdefault("ylabel", self.translate(args[1]))
            extra_args = args[2:] if len(args) > 0 else tuple()
            args = args[0], self.data.loc[:, args[1]].tolist(), *extra_args

        kwargs.setdefault("title", self.name_root)
        return visualize_hist2d(*args, **kwargs)

    def surface(self, x_name: str, y_name: str, z_name: str, **kwargs) -> plt.Figure:
        """Plot a surface from the data. Useful kwargs: [cumulative, density, bins]

        :param x_name: name of the x-axis data feature
        :param y_name: name of the y-axis data feature
        :param z_name: name of the z-axis data feature (color)
        :param kwargs: additional keyword arguments for visualize_surface which passes to seaborn heatmap()
        :return: figure handle for the plot
        """
        if isinstance(self, VectorSequence) and (not x_name):
            x_name: str = self.basis_col_name
        kwargs.setdefault(
            "xlabel", self.translate(x_name, missing_response="return_input")
        )
        kwargs.setdefault(
            "ylabel", self.translate(y_name, missing_response="return_input")
        )
        kwargs.setdefault(
            "title", self.translate(z_name, missing_response="return_input")
        )
        return visualize_surface(
            self.data.loc[:, x_name],
            self.data.loc[:, y_name],
            self.data.loc[:, z_name],
            **kwargs,
        )

    def visualize_embedded_surface(
        self, x_col_name: str, y_col_name: str, z_col_name: str, **kwargs
    ) -> plt.Figure:
        """Plots a surface (2D embedded in 3D) based on vector features"""
        return visualize_embedded_surface(
            self.data.loc[:, x_col_name],
            self.data.loc[:, y_col_name],
            self.data.loc[:, z_col_name],
            **kwargs,
        )

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

    def singular_value_decomposition(
        self, cols: Union[str, List[str]] = None, cache_results: bool = False, **kwargs
    ) -> SVD:
        """Applies singular value decomposition

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

    def to_surface_df(
        self, group_by_col_names: List[str], aggregation_method: str = "max", **kwargs
    ) -> pd.DataFrame:
        """Thin wrapper that aggregates the data to a surface, using pandas methods like mean, median, max, etc

        :param group_by_col_names: axes for the surface (i.e. how to bin the data)
        :param aggregation_method: how to combine multiple data points in the same bin
        :param kwargs: additional keyword arguments for pandas groupby method
        :return: data frame with combined according `aggregation_method` into bins of `group_by_col_names`
        """
        return getattr(
            self.data.groupby(by=group_by_col_names, **kwargs), aggregation_method
        )().reset_index(drop=False)

    def plot_projection_to_surface(
        self,
        surface_axes_names: Union[List[str], Tuple[str, str]],
        z_axis_name: str,
        aggregation_method: str = "max",
        **kwargs,
    ) -> plt.Figure:
        if len(surface_axes_names) != 2:
            raise ValueError(f"plot_projection is intended to be used with 2D surfaces")
        df_surface: pd.DataFrame = self.to_surface_df(
            group_by_col_names=surface_axes_names,
            aggregation_method=aggregation_method,
            **kwargs,
        )
        return visualize_x_y(
            df_surface.loc[:, surface_axes_names[0]],
            df_surface.loc[:, surface_axes_names[1]],
            c=df_surface.loc[:, z_axis_name].tolist(),
            xlabel=self.translate(surface_axes_names[0]),
            ylabel=self.translate(surface_axes_names[1]),
            colorbar_label=self.translate(z_axis_name),
        )

    def feature_selection_univariate(
        self,
        target_feature_name: str,
        input_feature_names: List[str] = None,
        k_best: int = 3,
        normalize: bool = True,
        **kwargs,
    ) -> any:
        """Feature selection of k best using univariate methods.

        :param target_feature_name: column name of the target feature
        :param input_feature_names: column names of the input features
        :param k_best: how many features to return
        :param normalize: if True, scales input features to unit mean
        :param kwargs: additional keyword arguments passed through to scikit-learn SelectKBest
        :return: trimmed input feature data set
        """
        if not input_feature_names:
            input_feature_names = [
                x for x in self.data.keys() if x != target_feature_name
            ]
        input_feature_data = self.data.loc[:, input_feature_names]
        if normalize:
            for fieldname in input_feature_data.keys():
                std_dev: float = float(
                    np.std(this_field_data := self.data.loc[:, fieldname])
                )
                input_feature_data[fieldname] = [x / std_dev for x in this_field_data]

        target_feature_data = self.data.loc[:, target_feature_name].to_numpy()
        return SelectKBest(chi2, k=k_best).fit_transform(
            input_feature_data, target_feature_data, **kwargs
        )

    def cast_to_numeric(
        self,
        columns: Union[List[str], str] = None,
        errors: str = "ignore",
        inplace: bool = True,
    ) -> Union[None, Any]:
        """
        Helper function that converts the data to numeric types

        :param columns: a string or list of strings of column names. If not provided, will try on all columns
        :param errors: passed to `pandas.to_numeric()`, options are: {‘ignore’, ‘raise’, ‘coerce’}
        :param inplace: whether to transform the data inplace (default) or return a fresh object
        """
        if not columns:
            columns = list(self.data.keys())
        if isinstance(columns, str):
            columns = [columns]

        df: pd.DataFrame = deepcopy(self.data)
        for col in columns:
            df[col] = pd.to_numeric(df.loc[:, col], errors=errors)

        if inplace:
            self.data = df
        else:
            return self.__class__(
                data=df, **{k: v for k, v in self.dict().items() if k != "data"}
            )

    def __init__(self, **kwargs):

        # Intercept if 'data' is a path instead of a dataframe
        if isinstance(kwargs.get("data", None), (str, pathlib.Path)):
            data_input: Union[str, pathlib.Path] = kwargs.get("data")
            if ".csv" in str(data_input).lower():
                if pathlib.Path(data_input).exists():
                    kwargs["data"] = pd.read_csv(data_input)
                else:
                    raise ValueError(
                        f"Input data string interpreted as a path does not exist:\n{data_input}"
                    )
            else:
                raise ValueError(
                    f"Input data appears to be a {type(data_input)} but is not a file with .csv extension"
                )

        super().__init__(**kwargs)

        if not kwargs.get("disable_auto_conversion_to_numeric", False) and (
            self.data is not None
        ):
            self.cast_to_numeric()

    def drop_col_types(
        self, drop_types: Union[type, List[type]], inplace: bool = True
    ) -> Union[Any, None]:
        """
        Helper function to drop columns of a particular type(s)

        :param drop_types: the types to be dropped
        :param inplace: whether the columns should be dropped inplace or if a new instance should be returned
        :return: None if inplace=True, otherwise a copy of the object with dropped columns
        """
        if not isinstance(drop_types, (list, tuple)):
            drop_types = [drop_types]
        target_cols: List[str] = [
            k
            for k in self.data.keys()
            if isinstance(self.data.loc[:, k][0], drop_types)
        ]
        if inplace:
            self.data.drop(columns=target_cols, inplace=True)
            return None
        else:
            to_return: Any = deepcopy(self)
            to_return.drop_col_types(drop_types=drop_types, inplace=True)
            return to_return

    def matrix_profile_univariate(
        self,
        col_names: Union[str, Iterable[str]],
        auto_locf: bool = False,
        suppress_nan_warning: bool = False,
        **kwargs,
    ) -> List[Tuple[Any, List[plt.Figure]]]:
        """
        Very thin wrapper around matrixprofile with some warnings and opt-in preprocessing automation

        :param col_names: a column name (or list of column names) to be profiled
        :param auto_locf:
        :param kwargs: keyword arguments passed through to matrixprofile.analyze()
        :return: List of outputs from matrixprofile.analyze()
        """
        queue: List[str] = [col_names] if isinstance(col_names, str) else col_names
        results: List[Tuple[Any, List[plt.Figure]]] = []
        try:
            for q in queue:
                ts: List[float] = self.data.loc[:, q].astype(float).tolist()

                if not kwargs.get("preprocessing_kwargs") and any(
                    np.isnan(x) for x in ts
                ):
                    if hasattr(self, "auto_locf_params"):
                        auto_locf_params: Dict[str, Any] = self.auto_locf_params
                    else:
                        auto_locf_params: Dict[str, Any] = {
                            "window": 3,
                            "impute_method": "median",
                            "impute_direction": "forward",
                            "add_noise": False,
                        }
                    if auto_locf:
                        kwargs.setdefault("preprocessing_kwargs", auto_locf_params)
                    else:
                        if not suppress_nan_warning:
                            print(
                                """
    Warning: matrix profiling data with NaNs and no preprocessing specified
             (potential anomalies or errors ahead)\n
    > To suppress this warning in the future, specify `suppress_nan_warning=True`.
    > One way to address NaNs is specifying a 'preprocessing_kwargs' dictionary, like:
        {'window': 3, 'impute_method': 'median', 'impute_direction': 'forward', 'add_noise': False}
    > Alternatively, you can specify `auto_locf=True` to use the config presets:\n"""
                                + "".join(
                                    [
                                        f"       ~ {k}={v}\n"
                                        for k, v in auto_locf_params.items()
                                    ]
                                )
                            )
                results.append(matrixprofile.analyze(ts, **kwargs))
        except TypeError as e:
            s: str = f"""
            Encountered a TypeError in matrixprofile.analyze(). Hint: this might happen if you are
            trying to use a custom parameter like 'use_right_edge' with the standard matrixprofile
            package. If that is the case, try running:\n
              pip uninstall matrixprofile && pip install git+https://github.com/mitchellpkt/matrixprofile
            \nOriginal error: {e}"""
            print(s)
            raise TypeError(s)

        return results

    def calculate_stumpy_profile_univariate(
        self, col_name: str, window_size: int, **kwargs
    ) -> np.ndarray:
        """
        Very thin wrapper around stumpy's matrix profile method. NB: for use in MultiSet, user is responsible for order

        :param col_name: column name to profile
        :param window_size: sliding window size
        :param kwargs: keyword arguments for stumpy.stump()
        :return: numpy array with the profile
        """
        return stumpy.stump(self.data.loc[:, col_name].tolist(), window_size, **kwargs)

    def stumpy_profile_univariate(
        self, col_name: str, window_size: int, annotate_motif: bool = True, **kwargs
    ) -> plt.Figure:
        """
        Plot of data & matrix profile (with optional annotation). NB: for use in MultiSet, user is responsible for order

        :param col_name: name of the column to plot
        :param window_size: sliding window size
        :param annotate_motif: whether to highlight the main motif and its nearest neighbor
        :param kwargs: keyword arguments for plots and stumpy.stump()
        :return: figure handle of the plot
        """
        # Make the plot and profile
        figsize: Any = kwargs.pop(
            "figsize", self.figsize
        )  # typically a 2-element tuple or list
        title: str = kwargs.pop("title", self.translate(self.name_root))
        profile: np.ndarray = self.calculate_stumpy_profile_univariate(
            col_name, window_size, **kwargs
        )
        minimum: float = float(np.nanmin(self.data.loc[:, col_name].tolist()))
        maximum: float = float(np.nanmax(self.data.loc[:, col_name].tolist()))
        fig, axs = plt.subplots(
            2,
            sharex="all",
            gridspec_kw={"hspace": 0},
            figsize=figsize,
            facecolor=self.facecolor,
        )
        plt.suptitle(title, fontsize=self.title_fontsize)
        axs[0].plot(
            self.data.loc[:, col_name].tolist(), color=kwargs.get("color", self.color)
        )
        axs[0].set_ylabel(self.translate(col_name), fontsize=self.label_fontsize)
        axs[1].set_xlabel(self.translate("Index"), fontsize=self.label_fontsize)
        axs[1].set_ylabel("Matrix Profile", fontsize=self.label_fontsize)
        axs[1].plot(profile[:, 0], color=kwargs.get("color", self.color))

        # Annotate the main motif (and its nearest neighbor) if desired
        if annotate_motif:
            motif_idx = np.argsort(profile[:, 0])[0]
            nearest_neighbor_idx = profile[motif_idx, 1]
            rect = Rectangle(
                (motif_idx, minimum), window_size, maximum, facecolor="palegoldenrod"
            )
            axs[0].add_patch(rect)
            rect = Rectangle(
                (nearest_neighbor_idx, minimum),
                window_size,
                maximum,
                facecolor="palegoldenrod",
            )
            axs[0].add_patch(rect)
            axs[1].axvline(x=motif_idx, linestyle="dashed", color="k")
            axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed", color="k")

        return fig

    # Group the multiset by a given row
    def group_by_col(self, by: str) -> Dict[Any, pd.DataFrame]:
        """Helper function that returns a dictionary of individual dataframes separated by 'by'"""
        return {
            x: self.data.loc[self.data.loc[:, by] == x, :]
            for x in set(self.data.loc[:, by])
        }


class SlidingWindowResults(VectorMultiset):
    """Results from Sequence.sliding_window(), which is a VectorMultiset with extra context baked in"""

    window_width_col_name: str = "window_width"
    window_start_col_name: str = "window_start"

    def group_by_window_width(self) -> Dict[float, pd.DataFrame]:
        """Helper function that returns individual dataframes for each window width"""
        return self.group_by_col(by=self.window_width_col_name)

    def plot_results(
        self, col_name: str, legend_override: List[str] = None, **kwargs
    ) -> plt.Figure:
        """Plot any sequence (based on the data frame column name)

        :param col_name: dataframe column to plot
        :param legend_override: specify custom legend keys if desired
        :return: figure handle
        """
        x_arrays: List[Any] = []
        y_arrays: List[Any] = []
        labels: List[Any] = []

        # For each `window_width` add we'll add a trace to the plot
        group_by_window_width: Dict[float, pd.DataFrame] = self.group_by_window_width()
        for key in sorted(
            group_by_window_width.keys()
        ):  # looping over group_by_window_width.items() gives wrong order
            value: pd.DataFrame = deepcopy(group_by_window_width[key])
            value.sort_values(
                by=self.window_start_col_name,
                ascending=True,
                ignore_index=True,
                inplace=True,
            )
            x_arrays.append(value.loc[:, self.window_start_col_name])
            y_arrays.append(value.loc[:, col_name])
            labels.append(str(key))  # noqa: key is a string

        # Final settings and overrides
        if legend_override:
            labels: List[str] = legend_override
        kwargs.setdefault("types", ["scatter", "plot"])
        kwargs.setdefault("xlabel", self.translate(self.translate("window_start")))
        kwargs.setdefault("ylabel", self.translate(col_name))
        kwargs.setdefault("title", self.translate(self.name_root))
        return visualize_x_y(
            x_arrays,
            y_arrays,
            legend_strings=labels,
            style=Style(**self.dict()),
            **kwargs,
        )

    def plot_pdfs(
        self, col_name: str, legend_override: List[str] = None, **kwargs
    ) -> plt.Figure:
        """
        Plot the probability density function(s) of the sliding window results. Useful kwargs: [cumulative, density]

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
        kwargs.setdefault("xlabel", self.translate(col_name))
        kwargs.setdefault("title", self.translate(self.name_root))
        return visualize_1d_distribution(data_sets, legend_strings=labels, **kwargs)

    def heatmap_feature(self, col_name: str, **kwargs) -> plt.Figure:
        """
        Returns a heatmap of some feature as a function of window start time and start width

        :param col_name: which column to plot
        :param kwargs: additional keyword arguments for surface method
        :return: figure handle of the plot
        """
        return self.surface(
            self.window_start_col_name, self.window_width_col_name, col_name, **kwargs
        )


class InfoSurface(SlidingWindowResults):
    """Wrapper for SlidingWindowResults that knows how to plot the infosurface"""

    def plot_info_surface(
        self, singular_values: List[int] = None, **kwargs
    ) -> List[plt.Figure]:
        """Plot the info surface showing value of singular vectors as a function of window start and width

        :param singular_values: Which singular values to plot [1, 2, 3] by default
        :return: figure handles
        """
        figure_handles: List[plt.Figure] = []
        if not singular_values:
            singular_values: List[int] = [1, 2, 3]
        for s in singular_values:
            figure_handles.append(
                self.heatmap_feature(
                    f"singular_value_{s}", title=f"Singular value # {s}", **kwargs
                )
            )
        return figure_handles


class VectorSequence(VectorMultiset):
    """Set of vectors in a sequence ordered according to some basis (e.g. height, wavelength, time)"""

    # (Inherits `data` dataframe and `name_root` string from VectorMultiset)
    basis_col_name: str = "basis"
    error_if_basis_quality_issues: bool = False
    info_surface: InfoSurface = None

    def __init__(self, skip_vector_sequence_init: bool = False, **kwargs: Any):
        super().__init__(**kwargs)

        # Cast to numeric if possible
        if not kwargs.get("disable_auto_conversion_to_numeric", False) and (
            self.data is not None
        ):
            self.cast_to_numeric()

        # Handle the other processing specific to ordered data
        if (
            (self.data is not None)
            and len(self.data)
            and (not skip_vector_sequence_init)
        ):
            if (not self.basis_col_name) or (
                self.basis_col_name not in self.data.keys()
            ):
                raise ValueError(
                    f"{self.basis_col_name=} not in known keys: {self.data.keys().tolist()=}\n"
                    + "Pass skip_vector_sequence_init=True to suppress"
                )
            self.sort(by=self.basis_col_name, inplace=True, reset_index=True)
            if self.error_if_basis_quality_issues:
                is_ok, explanation = self.passes_basis_quality_checks()
                if not is_ok:
                    raise ValueError(
                        f"Issues with basis:\n{explanation}\nSee: `error_if_basis_quality_issues`"
                    )

    def basis(self) -> List[Any]:
        """Thin wrapper that returns the basis as a list"""
        return self.data.loc[:, self.basis_col_name].tolist()

    def basis_quality_checks(self) -> Tuple[bool, str]:
        """Checks basis data quality and returns both a True/False flag and a string with an explanation

        :return: True if OK, and a string with explanation either way
        """
        return basis_quality_checks(self.basis())

    def passes_basis_quality_checks(self) -> bool:
        """Super thin wrapper around basis_quality_checks that drops the explanation string

        :return: True if OK
        """
        result, _ = self.basis_quality_checks()
        return result

    def basis_quality_plots(self, **kwargs) -> List[plt.Figure]:
        """Creates a series of plots showing the basis data quality (missing data, nans, timeseries gaps, etc)

        :return: figure handles for several plots
        """
        return basis_quality_plots(self.basis(), **kwargs)

    def fill_ratio(self) -> float:
        """Checks for how many data points are observed for the expected number of data points in a uniform basis.
            Attempts to infer spacing from median of diffs. If used on non-uniform data the result is not meaningful.

        Use example: [10, 20, 40, 50] has 0.8 fill ratio because it is missing the 30 to complete the set

        :return: fractional amount of the array
        """
        return fill_ratio(self.data.loc[:, self.basis_col_name])

    def slice(
        self,
        start_at: Any = None,
        stop_at: Any = None,
        inplace: bool = False,
        reset_index: bool = True,
        return_type: str = None,
        **kwargs,
    ) -> Union[None, pd.DataFrame, Tuple[Iterable, Iterable], Any]:
        """Slices the VectorSequence according to the basis

        # TODO have slice return self type for convenience

        :param start_at: Start point (can be None)
        :param stop_at: Stop point (can be None)
        :param inplace: slice inplace or return the result (default --> return)
        :param return_type: how to return the data ('VectorMultiset', 'VectorSequence', 'dataframe', etc)
        :param reset_index: indicate whether or not to reset the data frame index (default --> True)
        :param kwargs: keyword arguments passed through on init of non-dataframe objects
        :return: VectorSequence if not inplace
        """
        df: pd.DataFrame = deepcopy(self.data)
        df.sort_values(
            by=self.basis_col_name, ascending=True, inplace=True, ignore_index=True
        )
        if not start_at:
            start_at: Any = min(df.loc[:, self.basis_col_name])
        if not stop_at:
            stop_at: Any = max(df.loc[:, self.basis_col_name])
        in_range: pd.DataFrame = df.loc[
            df.loc[:, self.basis_col_name].between(start_at, stop_at), :
        ]
        if reset_index:
            in_range.reset_index(drop=True, inplace=True)
        if inplace:
            self.data = in_range
        else:
            if not return_type:
                return self.__class__(
                    basis_col_name=self.basis_col_name,
                    name_root=self.name_root,
                    data=in_range,
                    **kwargs,
                )
            else:
                if "sequence" in (return_type_lower := return_type.lower()):
                    return VectorSequence(
                        basis_col_name=self.basis_col_name,
                        name_root=self.name_root,
                        data=in_range,
                        **kwargs,
                    )
                elif "dataframe" in return_type_lower:
                    return in_range
                elif "timeseries" in return_type_lower:
                    return Timeseries(
                        basis_col_name=self.basis_col_name,
                        name_root=self.name_root,
                        data=in_range,
                        **kwargs,
                    )
                else:
                    raise ValueError(f"Unknown return type: {return_type}")

    def split(self, split_at: Union[float, str], gap: float = 0, **kwargs) -> tuple:
        """
        Splits the sequence into two subsequences (with an optional gap, which is placed after sthe split point)

        :param split_at: basis value to split
        :param gap: whether to leave a gap between the end of the pre-split and the start of the post-split (optional)
        :param kwargs: additional keyword arguments passed to the slice method (e.g. reset_index, return_type)
        :return: two objects separate at the split point
        """
        kwargs["inplace"] = False  # Cannot split one object in place
        pre_split = self.slice(stop_at=split_at, **kwargs)
        post_split = self.slice(start_at=split_at + gap, **kwargs)
        return pre_split, post_split

    def index_to_basis(self) -> None:
        """Adds (or overwrites) 'basis' column with a default index [0, 1, 2, ...]"""
        self.data.loc[:, self.basis_col_name] = list(range(len(self.data)))

    def sort(self, by: str = None, inplace: bool = True, reset_index: bool = True):
        """Sorts the data points by the basis

        :param inplace: import inplace (default) or return the result
        :param reset_index: indicate whether or not to reset the data frame index (default = True)
        :param by: which field to use for the sort (uses the basis by default)
        :return: sorted data frame, if inplace is False
        """
        if not by:
            by = self.basis_col_name
        result: pd.DataFrame = self.data.sort_values(
            by=by, ascending=True, inplace=False, ignore_index=False
        )
        if reset_index:
            result.reset_index(drop=True, inplace=True)
        if inplace:
            self.data = result
        else:
            return VectorSequence(
                basis_col_name=self.basis_col_name,
                name_root=self.name_root,
                data=result,
            )

    def downsample(
        self, interval: Union[int, float], method: str = "by_basis", inplace=True
    ):
        """
        Downsamples a VectorSequence

        :param interval: interval by which to downsample (either by N rows, or by some basis interval)
        :param method: 'by_basis' or 'by_row'
        :param inplace: whether to update the dataframe data attribute in place or return an updated copy of self
        :return:
        """
        result_vector: Any = deepcopy(self)
        result_vector.sort(inplace=True, reset_index=True)

        if "by_row" in method.lower():
            if not isinstance(interval, int):
                raise ValueError(
                    f"downsample with method='by_row' requires integer interval"
                )
            keep_indices: List[int] = list(range(0, len(result_vector.data), interval))

        elif "by_basis" in method:
            keep_indices: List[int] = []
            last: float = result_vector.basis()[0]
            for i, basis_value in enumerate(self.basis()):
                if basis_value >= last + interval:
                    keep_indices.append(i)
                    last = basis_value

        else:
            raise ValueError(
                f"Unknown downsample method {method}; try 'by_row' or 'by_basis'."
            )

        # Downsample and record or return the result
        result_vector.data = result_vector.data.iloc[keep_indices, :]
        if inplace:
            self.data = result_vector.data
        else:
            return result_vector

    def evaluate_over_window(
        self,
        function: Callable,
        start_at: float,
        window_width: float,
        args: Tuple,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Helper function that evaluates a function over a given window

        :param function: function to be applied to the window
        :param start_at: start time for the window
        :param window_width: width of the window
        :param args: positional arguments for the function (NB: args, not *args)
        :param kwargs: keyword arguments for the function (NB: kwargs, not **kwargs)
        :return: The evaluation of the function (along with a note of the window start and width)
        """
        window_data: VectorSequence = self.slice(
            start_at=start_at,
            stop_at=start_at + window_width,
            inplace=False,
            reset_index=True,
        )
        try:
            return {
                **function(window_data, *args, **kwargs),
                "window_start": start_at,
                "window_width": window_width,
            }
        except IndexError as e:
            window_description: str = f"{start_at=}\n{window_width=}\n"
            if not len(window_data):
                raise IndexError(
                    "\nCaught IndexError.\n\nHint: there were no data points in this window:\n"
                    + f"{window_description}\n(The best way to resolve this is probably to tweak\n"
                    + f"your eval function to gracefully handle empty slices)\n\nOriginal error:\n{e}"
                )
            else:
                raise IndexError(
                    f"Eval function had IndexError at window:\n{window_description}\nOriginal error:\n{e}"
                )
        except TypeError as e:
            raise TypeError(
                f"Caught type error (below). Hint: your eval function needs to return a dictionary.\n\n{e}"
            )

    #                                       vv sequence vv       vv args vv          vv kwargs vv
    def sliding_window(
        self,
        function: Callable[
            [Any, Union[Tuple[Any], List[Any]], Dict[str, Any]], Dict[str, Any]
        ],
        window_widths: List[float] = None,
        window_starts: List[Any] = None,
        step_size: float = None,
        parallelize_sliding_window: Union[bool, int] = True,
        allow_shorter_windows: bool = False,
        disable_sliding_window_progress_bar=None,
        limit: int = None,
        *args,
        **kwargs,
    ) -> SlidingWindowResults:
        """Apply function in a sliding window over the sequence

        :param function: callable to be applied
        :param window_widths: list of window widths to use (or a single value if only one width desired)
        :param window_starts: list of starting points for the windows
        :param step_size: how far apart to space windows
        :param allow_shorter_windows: whether to allow windows that are not the full length (disabled by default)
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        :param parallelize_sliding_window: whether to use multiprocessing for the sliding window
        :param disable_sliding_window_progress_bar: pass True to disable progress bar (only applies to serial mode)
        :param limit: set a limit for how many windows to evaluate, takes the first N
        :return: SlidingWindowResults (see the dataframe attribute for results)
        """

        # Validation
        if not self.basis_col_name:
            raise AttributeError(
                "cannot apply sliding window analysis without knowing which field is the basis"
            )
        if self.basis_col_name not in self.data.keys():
            raise AttributeError(
                f"basis_col_name {self.basis_col_name} not a column in data frame:\n{self.data}"
            )

        basis: Tuple[float] = self.data.loc[:, self.basis_col_name].tolist()

        if isinstance(window_widths, (float, int)):
            window_widths = [window_widths]

        if not window_widths:
            duration: float = max(self.data.loc[:, self.basis_col_name]) - min(
                self.data.loc[:, self.basis_col_name]
            )
            window_widths: List[float] = [
                duration / x for x in range(20, 401, 20)
            ]  # TODO: move vals to vars

        # First, figure out where to place the windows based on the window widths and/or steps size:
        list_of_start_and_width_tuples: List[Tuple[float, float]] = []
        for i, window_width in enumerate(window_widths):
            if not window_starts:
                if step_size:
                    if allow_shorter_windows:
                        window_starts: List[float] = list(
                            np.arange(min(basis), max(basis), step_size)
                        )
                    else:
                        window_starts: List[float] = list(
                            np.arange(min(basis), max(basis) - window_width, step_size)
                        )
                else:
                    window_starts: List[float] = list(
                        np.arange(min(basis), max(basis) - window_width, window_width)
                    )
            list_of_start_and_width_tuples += [
                (start_time, window_width) for start_time in window_starts
            ]

        if not list_of_start_and_width_tuples:
            raise ValueError(
                "No windows for sliding analysis - check your sizes and start times"
            )

        if limit and limit < len(list_of_start_and_width_tuples):
            list_of_start_and_width_tuples = list_of_start_and_width_tuples[:limit]

        # If parallelize_sliding_window != False, run in parallel using starmap() from multiprocessing library `Pool`
        num_workers: int = get_num_workers(parallelize_sliding_window)
        if parallelize_sliding_window and (num_workers > 1):
            evaluations: List[Dict[Any, Any]] = process_queue(
                func=self.evaluate_over_window,
                iterable=[
                    (function, start, width, args, kwargs)
                    for start, width in list_of_start_and_width_tuples
                ],
                pool_function="starmap",
                batching=False,
                num_workers=num_workers,
            )

        # If parallelize_sliding_window == False (or None) use `for` loop in list comprehension -> serial processing
        else:
            evaluations: List[Dict[Any, Any]] = [
                self.evaluate_over_window(
                    function=function,
                    start_at=start,
                    window_width=width,
                    args=args,
                    kwargs=kwargs,
                )
                for start, width in tqdm(
                    list_of_start_and_width_tuples,
                    disable=disable_sliding_window_progress_bar,
                )
            ]

        return SlidingWindowResults(
            window_width_col_name="window_width",
            window_start_col_name="window_start",
            data=pd.DataFrame(evaluations),
            name_root=self.name_root,
        )

    def repackage(
        self,
        instance: Any,
        sequence_attribute: str = "series",
        basis_name: str = "basis",
    ) -> Any:
        """
        Helper function for repackaging similar (BaseModel-like) objects into the top class

        :param instance: thing to be repackaged
        :param sequence_attribute: the name of the sequence (read: values) attribute
        :param basis_name: the name of the basis (read: time-like) attribute
        :return: repackaged instance with new class (if applicable)
        """
        cache: Dict[str, Any] = instance.dict()
        cache["data"] = pd.DataFrame(
            {x: input.__getattribute__(x) for x in [sequence_attribute, basis_name]}
        )
        cache["basis_col_name"] = basis_name
        return self.__class__(**cache)

    def seasonal_decompose(self, col: str, period: float, **kwargs):
        """Seasonal (weekly, monthly, etc) decomposition, must specify the period

        :param period: period of the data
        :param col: which data feature to use
        :param kwargs: additional kwargs for statsmodels.tsa.seasonal_decompose
        :return: sm.tsa.seasonal.DecomposeResult
        """
        x: pd.DataFrame = deepcopy(self.data.loc[:, col])
        try:
            return sm.tsa.seasonal_decompose(x, period=period, **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Error - did you include a non-numeric column, or not specify 'cols'? See: {e}"
            )

    def plot_decomposition(
        self,
        col: str,
        period: float,
        figsize: Tuple[float, float] = None,
        which_plots: List[str] = None,
        xlabel: str = "basis",
        ylabel: str = "[units]",
        title: str = None,
        xlim: List[float] = None,
        **kwargs,
    ) -> List[plt.Figure]:

        """Plot the seasonal (weekly, monthly, etc) decomposition. Must specify period

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
            which_plots: List[str] = ["observed", "trend", "seasonal", "residual"]
        if not title:
            title: str = self.translate(self.name_root)

        def add_labels(x_label: str, y_label: str) -> None:
            plt.xlabel(x_label, size=self.label_fontsize)
            plt.ylabel(y_label, size=self.label_fontsize)
            plt.legend(fontsize=self.legend_fontsize)

        decomposition = self.seasonal_decompose(col, period, **kwargs)
        figure_handles: List[plt.Figure] = []

        if "observed" in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(
                self.data.loc[:, self.basis_col_name],
                decomposition.observed,
                "-",
                label="Sequence",
                color="black",
            )
            add_labels(xlabel, ylabel)
            plt.title(f"{title}observed", size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        if "trend" in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(
                self.data.loc[:, self.basis_col_name],
                decomposition.trend,
                "-",
                label="Trend",
                color="green",
            )
            add_labels(xlabel, ylabel)
            plt.title(f"{title}trend", size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        if "seasonal" in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(
                self.data.loc[:, self.basis_col_name],
                decomposition.seasonal,
                "-",
                label="Seasonality",
                color="darkslateblue",
            )
            add_labels(xlabel, ylabel)
            plt.title(
                f"{title}seasonality with period {period}", size=self.title_fontsize
            )
            if xlim:
                plt.xlim(xlim)

        if "residual" in which_plots:
            figure_handles.append(plt.figure(figsize=figsize, facecolor=self.facecolor))
            plt.plot(
                self.data.loc[:, self.basis_col_name],
                decomposition.resid,
                "-",
                label="Residual",
                color="darkred",
            )
            add_labels(xlabel, ylabel)
            plt.title(f"{title}residual", size=self.title_fontsize)
            if xlim:
                plt.xlim(xlim)

        return figure_handles

    def singular_value_decomposition(
        self, cols: Union[str, List[str]] = None, cache_results: bool = False, **kwargs
    ) -> SVD:
        """Applies singular value decomposition, skipping the basis column

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

    def calculate_info_surface(
        self,
        window_widths: List[float] = None,
        cols: Union[str, List[str]] = None,
        *args,
        save_result: bool = True,
        **kwargs,
    ) -> InfoSurface:
        """Calculates the info surface by sliding the SVD function along the basis

        :param window_widths: window widths for the sliding window analysis
        :param cols: which data features to use in the svd
        :param args: args for sliding window analysis and eval function
        :param save_result: whether to save attach the infosurface to the object once calculated
        :param kwargs: kwargs for sliding window analysis and eval function
        :return: InfoSurface object
        """
        result: SlidingWindowResults = self.sliding_window(
            info_surface_slider, window_widths=window_widths, cols=cols, *args, **kwargs
        )
        info_surface: InfoSurface = InfoSurface(**result.dict())
        if save_result:
            self.info_surface = info_surface
        return info_surface

    def plot_info_surface(
        self,
        window_widths: List[float] = None,
        cols: Union[str, List[str]] = None,
        singular_values: List[int] = None,
        style: Style = None,
        use_cache: bool = False,
        *args,
        **kwargs,
    ) -> List[plt.Figure]:
        """Calculates and plots the info surface: value of singular vectors as a function of window start and width

        :param window_widths: window widths for the sliding window analysis
        :param cols: which data features to use in the svd
        :param singular_values: singular values for which to plot the surface (default: [1, 2, 3])
        :param use_cache: uses cached info surface if available, otherwise calculates a fresh one
        :param args: args for sliding window analysis and eval function
        :param kwargs: kwargs for sliding window analysis and eval function
        :param style: isthmuslib Style object for the colormap
        :return: list of figure handles
        """
        # Set style. Overrides: kwargs > style input > Style() defaults

        config: Style = Style(
            **{**Style().dict(), **make_dict(style), **make_dict(kwargs)}
        )
        svd_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in config.dict()
        }
        style_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in config.dict()
        }

        if use_cache and self.info_surface:
            info_surface: InfoSurface = self.info_surface
        else:
            info_surface: InfoSurface = self.calculate_info_surface(
                window_widths=window_widths, cols=cols, *args, **svd_kwargs
            )
        return info_surface.plot_info_surface(
            singular_values=singular_values,
            **{**make_dict(style), **make_dict(style_kwargs)},
        )

    def correlation_matrix(self, exclude_basis: bool = True, **kwargs) -> pd.DataFrame:
        """
        Very thin wrapper around correlation matrix (using pandas corr() method)

        :param exclude_basis: whether to exclude the basis column from correlation analysis
        :param kwargs: additional keyword arguments for correlation_matrix (fed through to df.style.background_gradient)
        :return: styled pandas dataframe
        """
        if exclude_basis:
            if kwargs_in := kwargs.get("exclude_cols"):
                if self.basis_col_name not in kwargs_in:
                    kwargs["exclude_cols"] = kwargs_in + [self.basis_col_name]
            else:
                kwargs: Dict[str, Any] = {"exclude_cols": self.basis_col_name}
        return correlation_matrix(self.data, **kwargs)

    def downsample_and_plot_matrix_profile_univariate(
        self,
        downsample_interval: float,
        col_names: Union[str, Iterable[str]],
        diff: bool = False,
        downsample_method: str = "by_basis",
        **kwargs,
    ) -> List[Tuple[Any, List[plt.Figure]]]:
        """
        Downsamples and optionally diffs data before applying the matrix profile (to which kwargs are passed through)

        :param downsample_interval: interval by which to downsample (either by N rows, or by some basis interval)
        :param downsample_method: 'by_basis' or 'by_row'
        :param col_names: a column name (or list of column names) to be profiled
        :param diff: whether to take the diff of the rows (coming soon - fractional differentiation??)
        :param kwargs: keyword arguments passed through to matrixprofile.analyze()
        :return: List of outputs from matrixprofile.analyze()
        """
        downsampled = self.downsample(
            downsample_interval, method=downsample_method, inplace=False
        )  # self type
        if diff:
            downsampled.data = downsampled.data.diff()
            downsampled.data.dropna(inplace=True)
        downsampled.data.reset_index(inplace=True)
        return downsampled.matrix_profile_univariate(col_names, **kwargs)

    def plot_fluss_semantic_segmentation(
        self,
        col_name: str = None,
        fluss_width: int = 15,
        n_regimes: int = 2,
        excl_factor: float = 1,
        **kwargs,
    ) -> plt.Figure:
        """
        Wrapper for STUMPY's FLUSS algonithm + plots

        :param col_name: name of the column to plot
        :param fluss_width: FLUSS window size
        :param n_regimes: number of regimes in the data
        :param excl_factor: factor to correct for the start and end of the arc curve
        :param kwargs: additional keyword arguments for stumpy and plots (figsize, title, ...)
        :return: figure handle showing the data and the analysis results
        """
        figsize: Any = kwargs.pop(
            "figsize", self.figsize
        )  # typically a 2-element tuple or list
        title: str = kwargs.pop("title", self.translate(self.name_root))

        # If column name not specified and only one non-basis column, infer that we should use that one
        if not col_name:
            if len(self.data.keys()) == 2:
                col_name: str = [
                    x for x in self.data.keys() if x != self.basis_col_name
                ][0]
            else:
                raise ValueError(
                    f"For fluss, please specify `col_name=X` with X from: {list(self.data.keys())}"
                )

        # Calculate the matrix profile and FLUSS results
        data_array: np.ndarray = self.data.loc[:, col_name].to_numpy()
        mp: np.ndarray = stumpy.stump(data_array, m=fluss_width, **kwargs)
        cac, regime_locations = stumpy.fluss(
            mp[:, 1], L=fluss_width, n_regimes=n_regimes, excl_factor=excl_factor
        )

        # Plot the results
        f, axs = plt.subplots(
            2,
            sharex=True,  # noqa: bool is in fact allowed
            gridspec_kw={"hspace": 0},
            figsize=figsize,
            facecolor=self.facecolor,
        )
        plt.suptitle(title, fontsize=self.title_fontsize)
        axs[0].plot(range(data_array.shape[0]), data_array, color=self.color)
        axs[0].axvline(x=regime_locations[0], linestyle="dashed")
        axs[1].plot(range(cac.shape[0]), cac, color="k")
        axs[1].axvline(x=regime_locations[0], linestyle="dashed")
        return f

    def human_time_start_and_stop(self, **kwargs) -> Tuple[str, str]:
        """
        Helper function that returns the min and max value of the basis column converted to human-readable string

        :param kwargs: keyword arguments for human_time()
        :return: tuple of strings like ('2025-03-21', '2028-07-04')
        """
        return self.human_time(
            min(self.data.loc[:, self.basis_col_name].astype(float).dropna().tolist()),
            **kwargs,
        ), self.human_time(
            max(self.data.loc[:, self.basis_col_name].astype(float).dropna().tolist()),
            **kwargs,
        )

    def human_timeframe(
        self, prefix: str = None, between: str = None, suffix: str = None, **kwargs
    ) -> str:
        """
        Formatting helper function for ensuring consistency across timeframe descriptions, for example

            ``From 2025-03-21 to 2028-07-04``

        :param prefix: text to precede a time/date range
        :param between: text to go between the times/dates
        :param suffix: text to follow the time/date range
        :param kwargs: additional keyword arguments for human_time, e.g. formatter = '%Y-%m-%d %H:%M:%S'
        :return: a single string describing the timeframe
        """
        kwargs.setdefault("formatter", self.formatter)
        if prefix is None:
            prefix: str = self.timeframe_prefix
        if between is None:
            between: str = self.timeframe_between
        if suffix is None:
            suffix: str = self.timeframe_suffix
        start_time_string, stop_time_string = self.human_time_start_and_stop(**kwargs)
        return f"{prefix}{start_time_string}{between}{stop_time_string}{suffix}"


class Timeseries(VectorSequence):
    """Thin wrapper for VectorSequence in the context of time"""

    basis_col_name: str = "timestamp"

    # x_axis_human_tick_labels: bool = True

    def calc_weights(
        self,
        lowest_weight: float = 0.5,
        decay_timeframe_sec: float = None,
        show_plots: bool = False,
        method: str = "linear",
        post_window_weight: float = None,
        base: float = np.e,
    ) -> List[float]:
        """
        Calculates a vector of weights from an exponential or linear decay

        :param lowest_weight: what should be the minimum weight? (i.e. at the end of the decay timeframe)
        :param decay_timeframe_sec: how long should the decay window be?
        :param post_window_weight: what should weights be after the decay window? Set to 0 to forget anything older.
        :param show_plots: whether to show plots of the weights
        :param method: whether the decay should be linear or exponential
        :param base: the base for the decay, but I don't think it matters
        :return: a list of weights
        """
        min_timestamp: float = min(self.data.loc[:, self.basis_col_name])
        max_timestamp: float = max(self.data.loc[:, self.basis_col_name])

        # If decay_timeframe_sec not specified, apply to the whole data set
        if decay_timeframe_sec is None:
            decay_timeframe_sec = max_timestamp - min_timestamp

        # If post_window_weight not specified, carry the last weight forward
        if post_window_weight is None:
            post_window_weight = lowest_weight

        # Make sure that we don't collide on column name
        age_col_name: str = "age"
        i: int = 0
        while age_col_name in self.data.keys():
            age_col_name += "_"
            i += 1
            if (
                i > 10
            ):  # this 10 is not a magic number, I just picked age_, age__, age___, up to 10 as excessive
                raise NotImplementedError(
                    f"Having {i} age columns with trailing underscores is not allowed..."
                )

        # Calculate age in a copy of the data frame
        df: pd.DataFrame = deepcopy(self.data)
        df[age_col_name] = max_timestamp - df.loc[:, self.basis_col_name]

        # Calculate the weights (not taking into account the decay timeframe)
        if "lin" in method.lower():  # 'lin' or 'linear'
            #  y = m * x + b
            slope: float = (lowest_weight - 1) / decay_timeframe_sec
            weights_all: List[float] = [slope * x + 1 for x in df.loc[:, age_col_name]]
        elif "exp" in method.lower():  # 'exp' or 'exponential'
            #  y = b ^ (-k * x) where k is the decay constant and x is the age
            decay_constant: float = (
                -1 * math.log(lowest_weight, base) / decay_timeframe_sec
            )
            weights_all: List[float] = [
                base ** (-1 * decay_constant * tau) for tau in df.loc[:, age_col_name]
            ]
        else:
            raise ValueError(
                f"Unknown method {method}, please try 'linear' or 'exponential'"
            )

        # Replace any out-of-timeframe data points with the post_window_weight
        ages_and_weights: zip = zip(df.loc[:, age_col_name], weights_all)
        weights_raw: List[float] = [
            w if a < decay_timeframe_sec else post_window_weight
            for a, w in ages_and_weights
        ]
        normalization_factor = sum(weights_raw)
        final_weights: List[float] = [w / normalization_factor for w in weights_raw]

        # Show plots of weights for the curious user
        if show_plots:
            self.scatter(
                df.loc[:, self.basis_col_name].tolist(),
                final_weights,
                xlabel=self.translate(self.basis_col_name),
                ylabel="weights",
                title="weights vs timestamps",
            )

            self.scatter(
                df.loc[:, age_col_name].tolist(),
                final_weights,
                xlabel="age",
                ylabel="weights",
                title="weights vs ages",
            )
            plt.show()

        return final_weights

    def weighted_mean(self, col_name: str, **kwargs) -> float:
        """
        Calculates the weighted mean of the provided column name. kwargs are passed to calc_weights

        :param col_name: which column to calculate the weight for
        :param kwargs: kwargs for calc_weights, e.g. lowest_weight, decay_timeframe_sec, post_window_weight, method
        :return: weighted mean of that column
        """

        weights = self.calc_weights(**kwargs)
        return sum(
            [weight * val for weight, val in zip(weights, self.data.loc[:, col_name])]
        )


################
# Helpers
################


def correlation_matrix(
    dataframe: pd.DataFrame,
    use_cols: List[str] = None,
    exclude_cols: List[str] = None,
    correlation_method: str = "pearson",
    style: Style = None,
    **kwargs,
) -> pd.DataFrame:
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
        kwargs.setdefault("cmap", style.cmap)
    else:
        kwargs.setdefault("cmap", Style().cmap)
    corr: pd.DataFrame = dataframe.loc[:, use_cols].corr(method=correlation_method)
    return corr.style.background_gradient(**kwargs)  # noqa: misinterprets type


def info_surface_slider(vs: VectorSequence, *args, **kwargs) -> Dict[str, Any]:
    """
     Helper function for sliding window SVD analysis

    :param vs: vector sequence
    :param args: args for svd
    :param kwargs: kwargs for svd
    :return: dictionary with singular values
    """
    svd: SVD = vs.singular_value_decomposition(*args, **kwargs)
    return {f"singular_value_{i + 1}": value for i, value in enumerate(svd.s)}


def singular_value_decomposition(
    df: pd.DataFrame, cols: Union[str, List[str]] = None, **kwargs
) -> SVD:
    """
    Helper function that wraps numpy svd to feed in a subset of data features (see kwarg: 'full_matrices')

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
        raise TypeError(
            "Notice: OrderedSeries class has been supplanted by VectorSequence, see repackage() method"
        )
