import pickle as pickle
from typing import Any
from pydantic import BaseModel
from typing import Dict, Union, List, Tuple
from datetime import datetime
import pytz
from dateutil import parser
import pandas as pd
import pathlib
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import cpu_count


class PickleUtils(BaseModel):
    """ Pickle IO - adds pickle import & export helper functions to any class that imports these utils  """

    def to_pickle(self, file_path: Union[str, pathlib.Path]) -> None:
        """Exports self as a pickle
        @param file_path: where to write the file
        """
        with open(file_path, 'wb') as outfile:
            pickle.dump(self, outfile)

    def read_pickle(self, file_path: Union[str, pathlib.Path]) -> Any:  # noqa: static, placed here for organization
        """ Imports a pickle. Note: this is _not_ inplace; the pickle contents are _returned_ by this method
        @param file_path: file to import
        """
        with open(file_path, 'rb') as infile:
            return pickle.load(infile)


class PickleJar(PickleUtils):
    """ Class with to_pickle() and from_pickle(), supporting arbitrary data in `contents` attribute """
    contents: Any = None

    class Config:
        arbitrary_types_allowed = True

    def from_pickle(self, file_path: Union[str, pathlib.Path]) -> None:
        """ Imports a pickle (similar to read_pickle(), but this method is inplace)
        @param file_path: file to import
        """
        self.contents = self.read_pickle(file_path).contents


class Rosetta(BaseModel):
    """ Rules and methods for converting timestamps and raw labels to human-readable formats """
    timezone: str = 'US/Pacific'
    formatter: str = '%Y-%m-%d %H:%M:%S'
    stone: Dict[str, str] = {
        # input string : human-readable output
        "input": "The Input (formatted)",
        "window_start": "Window start time",
        "window_width": "Window width"
        # ... (fill out the rest of the human-readable names)
    }
    default_missing_response: str = 'return_input'

    def translate(self, key: str, missing_response: str = None) -> str:
        """ Main function that wraps dictionary access with robust handling for missing inputs, missing keys, etc.
        Consider using for cases like axis.label(self.translate('timestamp_sec')) --> axis.label("Timestamp (sec)")
        @param key: input to be translated
        @param missing_response: how to handle a missing entry ("return_input" or "error")
        @return: human-readable output
        """
        if not key:
            return ''
        if key in self.stone:
            return self.stone[key]
        if not missing_response:
            missing_response: str = self.default_missing_response
        if missing_response == "return_input":
            return key
        elif missing_response == "error":
            raise KeyError(f"\nNOT FOUND: {key}\n in mappings:\n{self.stone}")
        else:
            raise ValueError(f"Unknown missing_response parameter: {missing_response}")

    def translate_time(self, key: Union[str, float, int], include_timezone: bool = True) -> str:
        """ Convert a timestamp (seconds) into human readable string """
        return human_time(key, formatter=self.formatter, timezone=self.timezone, include_timezone=include_timezone)

    def add_entry(self, key: str, value: str, silent_overwrite: bool = True) -> None:
        """ Adds an entry to the mappings
        @param key: input string
        @param value: human-readable output
        @param silent_overwrite: if True, overwrites existing entries. If false, raises KeyError
        """
        if (key in self.stone) and (not silent_overwrite):
            raise KeyError(f"Key {key} is already in mappings, and silent_overwrite is False")
        self.stone[key] = value

    def human_time(self, *args, **kwargs) -> str:
        return human_time(*args, **kwargs)

    def machine_time(self, *args, **kwargs) -> float:
        return machine_time(*args, **kwargs)

    def __add__(self, other):
        """ Rosetta objects can be combined. Note: if both stones have the same key with a different value,
            the output will reflect the value from the second (`other`) stone """
        return {**self.stone, **other.stone}

    def __iadd__(self, other):
        """ Rosetta objects can be combined. Note: if both stones have the same key with a different value,
            the output will reflect the value from the second (`other`) stone """
        self.stone = {**self.stone, **other.stone}

    def merge(self, string: str, **kwargs) -> str:
        """
        Helper function that replaces merge fields in a string

        :param string: string to be modified, e.g. "Records from [[start_date]] for [[first_name]]"
        :return: merged string, e.g. "Records from 2011-05-06 for Foobar
        """
        return object_string_merge(string=string, values_from=self.stone, **kwargs)

    def risky_cast(self, x: Any) -> Any:  # noqa: it is static, attached here only for convenience.
        """ Reckless helper function that tries to cast the input to a number (float) or boolean """
        return risky_cast(x)


def human_time(timestamp_sec: Union[float, str, int], formatter: str = '%Y-%m-%d %H:%M:%S',
               timezone: str = 'US/Pacific', include_timezone: bool = True) -> str:
    """ Converts timestamp to human readable time, taking into account time zone (US/Pacific by default)
        To see other time zone options, see `pytz.common_timezones` """

    # If input is a string, try to parse it as a float or int
    if isinstance(timestamp_sec, str):
        if timestamp_sec.replace('.', '', 1).isdigit():
            timestamp_sec: float = float(timestamp_sec)
        else:
            raise ValueError(f"Could not interpret string as a numeric timestamp: {timestamp_sec})")
    datetime_string: str = datetime.fromtimestamp(timestamp_sec, pytz.timezone(timezone)).strftime(formatter)
    if include_timezone:
        datetime_string += f" ({timezone})"
    return datetime_string


def machine_time(time_or_times: Union[str, Any], units: str = 'seconds',
                 disable_progress_bar: bool = True, **kwargs) -> Union[float, List[float]]:
    """
    Convert a string to a timestamp
    @param time_or_times: datetime string to parse (or a list of them)
    @param units: seconds or milliseconds
    @param disable_progress_bar: setting to False will activate a tqdm progress bar for conversions over lists
    @return: unix timestamp
    """

    # Get the units multiplier
    if units in ['s', 'sec', 'second', 'seconds']:
        multiplier: int = 1
    elif units in ['ms', 'millisecond', 'milliseconds']:
        multiplier: int = 1000
    else:
        raise ValueError(f"Unknown units: {units}")

    # If just one string, drop it into a list
    if isinstance(time_or_times, str):
        return float(multiplier * parser.parse(time_or_times).timestamp())
    else:
        return [float(multiplier * parser.parse(x).timestamp()) for x in
                tqdm(time_or_times, disable=disable_progress_bar, **kwargs)]


def as_list(anything: Union[Any, List[Any]]) -> List[Any]:
    """
    If it's not a list, stick it in a list. If it's already a list, return it.
    @param anything: anything that you want to ensure is in a list
    @return: anything, but in list form
    """
    if isinstance(anything, list):
        return anything
    return [anything]


def looks_like_list_of_lists(input_var: Any) -> bool:
    """ Does this look like a list of lists? Wraps a pandas util

    :param input_var: input to assess
    :return: True if it looks like a list of lists
    """
    return pd.api.types.is_list_like(input_var) and pd.api.types.is_list_like(input_var[0])


def margin_calc(margin: float, span: Tuple[float, float], scale: str) -> float:
    """ Helper function for ascertaining watermark or hud placements (in both linear and log environments)

    :param margin: a fractional placement (e.g. 0.05 --> 5% from the left of the pane)
    :param span: range being spanned
    :param scale: 'linear' or 'log'
    :return: float with the coordinate value
    """
    if scale == 'linear':
        return span[0] + margin * (span[1] - span[0])
    elif scale == 'log':
        return 10 ** (np.log10(span[0]) + margin * (np.log10(span[1]) - np.log10(span[0])))
    else:
        raise ValueError(f"Unexpected {scale=}")


def to_list_if_other_array(array: Any) -> List[Any]:
    """ Helper function that casts 1D data frames and numpy ndarrays to lists (important for inputs of core viz code)

    :param array: An array-like (1D) object with numeric values
    :return: a list
    """
    if isinstance(array, pd.DataFrame) and (l := len(array.keys()) > 1):
        raise ValueError(f"Instead of array, received data frame with {l} features / columns")
    if isinstance(array, (np.ndarray, pd.Series)):
        return array.tolist()
    return array


def zero_mean_unit_deviation(array: any) -> List[float]:
    """ Helper function that maps a vector to zero mean and unit standard deviation

    :param array: anything that looks like an array
    :return: list with the normalized values
    """
    std_dev: float = float(np.std(array))
    mean: float = float(np.mean(array))
    return [(x - mean) / std_dev for x in array]


def make_dict(d: Union[Dict, object, None]) -> Dict[Any, Any]:
    """
    Converts inputs to a dictionary if possible (or provides an empty one if None

    :param d: either a dictionary or None
    :return: passes through input, or an empty dictionary if input is None
    """
    if not d:
        return dict()
    elif isinstance(d, object) and ('dict' in dir(d)):
        return d.dict()  # noqa
    else:
        return d


def get_num_workers(parallelize_arg: Union[bool, int]) -> int:
    """
    This is a helper function to ascertain the number of workers based on the parallelize_sliding_window input

    :param parallelize_arg: Whether to use multiprocessing for the sliding window. If True, uses # CPU cores.
    :return: number of parallel workers to instantiate
    """
    if parallelize_arg and (cpu_count() > 1):  # (if only have one core, no benefit from multiprocessing)
        if isinstance(parallelize_arg, bool):
            return cpu_count()
        else:
            return min(cpu_count(), parallelize_arg)
    else:
        return 1


def determine_load_per_worker(num_tasks: int, num_workers: int) -> List[int]:
    """
    This is a helper function to split up tasks across workers for batching. Example:
        Input: 83 tasks and 8 workers
        Output: [11, 11, 11, 10, 10, 10, 10, 10]

    :param num_tasks: Number of tasks to be tackled
    :param num_workers: Number of workers available in the pool (usually = number of CPU cores)
    :return: List designating how many tasks each worker should take
    """
    remainder: int = num_tasks % num_workers
    return remainder * [1 + (num_tasks // num_workers)] + (num_workers - remainder) * [num_tasks // num_workers]


def divvy_workload(num_workers: int, tasks: List[Any]) -> List[List[Any]]:
    """
    Helper function that divvies up some tasks among a specific number of workers

    :param num_workers: number of workers available in the pool (usually = number of workers)
    :param tasks: task list to be carried out
    :return: task list broken up into num_workers segments
    """
    load_per_worker: List[int] = determine_load_per_worker(num_tasks=len(tasks), num_workers=num_workers)
    i: int = 0
    task_list_all: List[List[Any]] = []
    for load_amount in load_per_worker:
        task_list_all.append(tasks[i:i + load_amount])
        i += load_amount
    return task_list_all


def object_string_merge(string: str, values_from: Any, left_merge_token: str = '[[',
                        right_merge_token: str = ']]') -> str:
    """
    Helper function that replaces merge fields in a string


    :param string: string to be modified, e.g. "Records from [[start_date]] for [[first_name]]"
    :param values_from: source of values, can be dictionary or BaseModel-like object with dict() method.
    :param left_merge_token: token to indicate left end of a merge field
    :param right_merge_token: token to indicate right end of a merge field
    :return: merged string, e.g. "Records from 2011-05-06 for Foobar
    """
    if isinstance(values_from, dict):
        values_dict: Dict[str, Any] = values_from
    elif hasattr(values_from, 'dict'):
        values_dict: Dict[str, Any] = values_from.dict()
    else:
        raise ValueError(f"Unsure how to interpret `values_from`. Expecting a dictionary or BaseModel-like object.")

    for k, v in values_dict.items():
        if k in string:
            string = string.replace(f"{left_merge_token}{k}{right_merge_token}", f"{v}")
    return string


def risky_cast(x: Any) -> Any:
    """
    Reckless helper function that tries to cast the input to a dictionary or number (float) or boolean.

    """

    # If not a string, send it back
    if not isinstance(x, str):
        return x

    # Try to parse model representations
    if all(substr in x for substr in ['=', ',']):
        try:
            return {k[0]: risky_cast(k[1]) for k in [j.split('=') for j in x.split(', ')]}  # "a=5, b=6"
        except:
            pass

    # Float?
    if x.replace('.', '', 1).isdigit():
        if '.' in x:
            try:
                return float(x)
            except:
                pass

        # Integer?
        else:
            try:
                return int(x)
            except:
                pass

    # Bool?
    str_to_bool_mapper: Dict[str, Any] = {'true': True, 'false': False}
    if x.lower() in str_to_bool_mapper:
        return str_to_bool_mapper[x.lower()]

    # Give up
    return x
