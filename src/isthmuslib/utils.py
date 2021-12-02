import pickle as pickle
from typing import Any
from pydantic import BaseModel
from typing import Dict, Union, List
from datetime import datetime
import pytz
from dateutil import parser


class PickleUtils(BaseModel):
    """ Pickle IO - adds pickle import & export helper functions to any class that imports these utils  """

    def to_pickle(self, file_path: str) -> None:
        """Exports self as a pickle
        @param file_path: where to write the file
        """
        with open(file_path, 'wb') as outfile:
            pickle.dump(self, outfile)

    def read_pickle(self, file_path: str) -> Any:  # noqa: could be static, placed here for convenient organization
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

    def from_pickle(self, file_path: str) -> None:
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
        # ... (fill out the rest of the human-readable names)
    }

    def translate(self, key: str, missing_response: str = "return_input") -> str:
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
    if isinstance(timestamp_sec, str):
        if timestamp_sec.replace('.', '', 1).isdigit():
            timestamp_sec: float = int(timestamp_sec)
        else:
            raise ValueError(f"Could not interpret string as a numeric timestamp: {timestamp_sec})")
    datetime_string: str = datetime.fromtimestamp(timestamp_sec, pytz.timezone(timezone)).strftime(formatter)
    if include_timezone:
        datetime_string += f" ({timezone})"
    return datetime_string


def machine_time(time: str, units: str = 'seconds') -> float:
    """
    Convert a string to a timestamp
    @param time: datetime string to parse
    @param units: seconds or milliseconds
    @return: unix timestamp
    """
    unix_time_sec: float = parser.parse(time).timestamp()
    if units in ['s', 'sec', 'second', 'seconds']:
        return unix_time_sec
    elif units in ['ms', 'millisecond', 'milliseconds']:
        return 1000 * unix_time_sec
    else:
        raise ValueError(f"Unknown units: {units}")


def as_list(anything: Union[Any, List[Any]]) -> List[Any]:
    """
    If it's not a list, stick it in a list. If it's already a list, return it.
    @param anything: Anything that you want to ensure is in a list
    @return: Anything, but in list form
    """
    if isinstance(anything, list):
        return anything
    return [anything]
