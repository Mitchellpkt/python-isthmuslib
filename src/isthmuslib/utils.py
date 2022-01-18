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

    def __add__(self, other):
        """ Rosetta objects can be combined. Note: if both stones have the same key with a different value,
            the output will reflect the value from the second (`other`) stone """
        return {**self.stone, **other.stone}

    def __iadd__(self, other):
        """ Rosetta objects can be combined. Note: if both stones have the same key with a different value,
            the output will reflect the value from the second (`other`) stone """
        self.stone = {**self.stone, **other.stone}


def human_time(timestamp_sec: Union[float, str, int], formatter: str = '%Y-%m-%d %H:%M:%S',
               timezone: str = 'US/Pacific', include_timezone: bool = True) -> str:
    """ Converts timestamp to human readable time, taking into account time zone (US/Pacific by default)
        To see other time zone options, see `pytz.common_timezones` """

    # If input is a string, try to parse it as a float or int
    if isinstance(timestamp_sec, str):
        if timestamp_sec.replace('.', '', 1).isdigit():
            timestamp_sec: float = int(timestamp_sec)
        else:
            raise ValueError(f"Could not interpret string as a numeric timestamp: {timestamp_sec})")
    datetime_string: str = datetime.fromtimestamp(timestamp_sec, pytz.timezone(timezone)).strftime(formatter)
    if include_timezone:
        datetime_string += f" ({timezone})"
    return datetime_string


def machine_time(time_or_times: Union[str, Any], units: str = 'seconds',
                 disable_progress_bar: bool = True) -> Union[float, List[float]]:
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
                tqdm(time_or_times, disable=disable_progress_bar)]


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
