import pickle as pickle
import time
from pydantic import BaseModel
from datetime import datetime
import pytz
from dateutil import parser
import pandas as pd
import pathlib
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import cpu_count, Pool, current_process
import itertools as itertools
from typing import Iterable, List, Tuple, Dict, Any, Union, Callable
import math
import random


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
        if timestamp_sec == 'now':
            timestamp_sec = time.time()
        else:
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

    # Get the units multiplier (todo - make this cleaner with a mapper)
    if units in ['s', 'sec', 'second', 'seconds']:
        multiplier: int = 1
    elif units in ['ms', 'milli', 'millisecond', 'milliseconds']:
        multiplier: int = 1_000
    elif units in ['us', 'micro', 'microsecond', 'microseconds']:
        multiplier: int = 1_000_000
    elif units in ['ns', 'nano', 'nanosecond', 'nanoseconds']:
        multiplier: int = 1_000_000_000
    else:
        raise ValueError(f"Unknown units: {units}")

    # If just one string, drop it into a list
    if isinstance(time_or_times, str):
        return float(multiplier * parser.parse(time_or_times).timestamp())
    else:
        return [float(multiplier * parser.parse(x).timestamp()) for x in
                tqdm(time_or_times, disable=disable_progress_bar, **kwargs)]


def grid(individual_parameter_values: Dict[str, Iterable]) -> List[Dict[str, Any]]:
    """
    Implements the grid search approach to hyperparameter optimization & tuning. This function creates a list of
    parameter sets that map out a discrete exhaustive search over arbitrary dimensions of arbitrary type.

    Example:
    Input:  {'temperature': [150, 160], 'circulation': [True, False]}
    Output: [{'temperature': 150, 'circulation': True}, {'temperature': 150, 'circulation': False},
             {'temperature': 160, 'circulation': True}, {'temperature': 160, 'circulation': False}]

    :param individual_parameter_values: dictionary, key = parameter name, value = iterables of sample points
    :return: multiple input parameter sets spanning the space
    """
    keys_list: Tuple[str] = tuple(individual_parameter_values.keys())
    values_list: Tuple[Iterable] = tuple(individual_parameter_values.values())
    combinations = itertools.product(*values_list)
    return [{keys_list[i]: value for i, value in enumerate(combination)} for combination in combinations]


def neighborhood_grid(starting_point: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    Helper function that produces a grid around a coordinate described by a dictionary

    :param starting_point: dictionary describing the coordinate
    :param kwargs: additional keywords for neighborhood functions (below) and numpy linspace / logspace
    :return: a list where each element is a dictionary with a point in the neighborhood of the starting point
    """
    return grid(neighborhood_multivariate(starting_point, **kwargs))


def neighborhood_multivariate(starting_point: Dict[str, Any], errors: str = 'passthrough', fields: List[str] = None,
                              keep_other_vals: bool = True, **kwargs) -> Dict[str, List[float]]:
    """
    Helper function that wraps the univariate helper function for dictionaries with multiple fields

    :param starting_point: initial point
    :param errors: whether non-numeric rows should be passed through (default), dropped, or raised
    :param kwargs: additional keyword arguments passed to neighborhood_univariate (and into numpy linspace / logspace)
    :param fields: which fields (keys) to expand into neighborhoods
    :param keep_other_vals: whether to keep the other values (whose keys are not in fields)
    :return: a dictionary (with keys matching the input) whose values contain the neighborhoods
    """

    if not fields:
        fields = list(starting_point.keys())

    if keep_other_vals:
        return_dictionary: Dict[str, List[float]] = {k: [v] for k, v in starting_point.items()}
    else:
        return_dictionary: Dict[str, List[float]] = dict()

    for key, value in [x for x in starting_point.items() if x[0] in fields]:
        try:
            if isinstance(value, bool) or (not isinstance(value, (int, float))):
                raise TypeError(f'The variable {key} has a value that is not a float or integer: {value}')
            return_dictionary[key] = neighborhood_univariate(value, **kwargs)
        except TypeError as e:
            if 'passthrough' in errors.lower():
                return_dictionary[key] = [value]
            elif 'drop' in errors.lower():
                pass
            elif 'raise' in errors.lower():
                raise ValueError(f"The value of {key} ({value}) does not appear to be numeric. Lower error: {e}")
            else:
                raise ValueError(f"Unknown error handling method {errors}. Try 'passthrough', 'drop', or 'raise'.")

    return return_dictionary


def neighborhood_univariate(starting_point: float, width_prct: float = 50, num_samples: int = 5,
                            spacing: str = 'linear', placement: str = 'center',
                            width_temperature_prct: float = None, **kwargs) -> List[float]:
    """
    Helper function that samples the area around a point (just a wrapper for numpy linspace and logspace)

    :param starting_point: the value for which we want the neighborhood
    :param width_prct: the percentage _total_ width of the bin (so 5% with starting point 10 --> [7.5, 12.5])
    :param num_samples: how many samples to include
    :param spacing: whether the spacing should be linear or log
    :param placement: whether the starting point should be at the left edge, center, or right edge of the neighborhood
    :param width_temperature_prct: applies a +/- perturbation to the window width itself (not the absolute value)
    :param kwargs: additional keyword arguments passed through to numpy linspace / logspace
    :return: the points to sample for the neighborhood
    """
    width: float = starting_point * width_prct / 100
    if width_temperature_prct:
        width: float = width * (1 + random.uniform(-1 * abs(width_temperature_prct), abs(width_temperature_prct)) / 100)

    if placement.lower() == 'left_edge':
        left, right = starting_point, starting_point + width
    elif placement.lower() in 'centered':  # 'center' or 'centered' is fine
        left, right = starting_point - width / 2, starting_point + width / 2
    elif placement.lower() == 'right_edge':
        left, right = starting_point - width, starting_point
    else:
        raise ValueError(f"Unknown placement: {placement}. Try 'left_edge', 'center', or 'right_edge'.")

    if spacing.lower() == 'linear':
        return list(np.linspace(left, right, num_samples, **kwargs))
    elif 'log' in spacing.lower():  # 'log' or 'logarithmic' is fine
        return list(np.logspace(np.log10(left), np.log10(right), num_samples, **kwargs))
    elif spacing.lower() == 'inverse_hyperbolic_transform':
        raise NotImplementedError(f"Not implemented ... yet ... spoiler alert")
    else:
        raise ValueError(f"Unknown spacing: {spacing}. Try 'linear' or 'log'.")


class MaxTimeException(KeyboardInterrupt):
    pass


def recursive_batch_evaluation(func: Callable,
                               initial_input: Union[Dict[str, Any], Any],
                               selection_method: Callable = None,
                               batch_generator: Callable = None,
                               batch_generator_kwargs: Dict[str, Any] = None,
                               max_deep: int = None,
                               max_time_sec: int = None,
                               return_input_and_value_tuple: bool = False,
                               print_progress: bool = False,
                               print_current_value: bool = False,
                               print_current_inputs: bool = False,
                               evaluate_initial_inputs: bool = True,
                               *_, **kwargs) -> Union[Any, Tuple[Any, Any]]:
    """
    Helper function that applies f recursively in batches

    :param func: some evaluation function that outputs a sortable object (e.g. a float or int)
    :param initial_input: initial inputs (typically a dictionary for standard methods, but is flexible)
    :param batch_generator: how to generate the next batch from a given seed
    :param batch_generator_kwargs: additional keyword arguments for the batch generator
    :param return_input_and_value_tuple: if True, returns a (inputs, value) tuple; otherwise only returns the inputs
    :param max_time_sec: max time to run in seconds
    :param max_deep: how many layers deep to go
    :param selection_method: how to select one element from the outputs to use for seeding the next batch
    :param print_progress: whether to log information like depth and timing
    :param print_current_value: whether to print stepwise value (might be ok for int or str, but avoid if big / complex)
    :param print_current_inputs: whether to print stepwise spot (might be ok for int or str, but avoid if big / complex)
    :param evaluate_initial_inputs: whether to evaluate the inputs before beginning the main cycle
    :param kwargs: additional kwargs for process_queue, which are passed through to Pool's map() and starmap()
    :return: the best inputs (or if return_input_and_value_tuple=True, returns the value too)
    """

    # Handle defaults
    if batch_generator is None:
        batch_generator = neighborhood_grid
    if batch_generator_kwargs is None:
        batch_generator_kwargs = dict()
    if selection_method is None:
        selection_method = max

    # Initialize
    current_best_input = initial_input
    if evaluate_initial_inputs:
        current_best_value = func(current_best_input)
    else:
        current_best_value = None
    func_inputs_iterable: List[Dict[str, Any]] = batch_generator(current_best_input, **batch_generator_kwargs)

    # Begin recursively applying
    start_time = time.perf_counter()
    counter: int = 0
    try:
        while (max_deep is None) or (counter < max_deep):
            tic: float = time.perf_counter()
            kwargs.setdefault('pool_function', 'map')
            output_vals: List[Any] = process_queue(func, func_inputs_iterable, **kwargs)
            current_best_value = selection_method(output_vals)
            current_best_input = [i for i, v in zip(func_inputs_iterable, output_vals) if v == current_best_value][0]
            func_inputs_iterable = batch_generator(current_best_input, **batch_generator_kwargs)
            counter += 1
            if print_progress:
                print(f"Completed cycle #{counter} in {time.perf_counter() - tic:.6f} seconds")
            if print_current_value:
                print(f"... Current best value: {current_best_value}")
            if print_current_inputs:
                print(f"... Current best input is: {current_best_input}")
            if max_time_sec:
                if time.perf_counter() > start_time + max_time_sec:
                    raise MaxTimeException
    except KeyboardInterrupt as e:
        if isinstance(e, MaxTimeException):
            print(f"Breaking after {counter} cycles for max time allowance ({max_time_sec / 60:.2f} minutes)")
        print(f"Breaking for keyboard interrupt after {counter} cycles.")
    except Exception as e:
        print(f"After {counter} cycles, encountered exception {e}")

    # Return the results
    if return_input_and_value_tuple:
        return current_best_input, current_best_value
    else:
        return current_best_input


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


def get_num_workers(parallelize_arg: Union[bool, int, None]) -> int:
    """
    This is a helper function to ascertain the number of workers based on the parallelize_sliding_window input

    :param parallelize_arg: Whether to use multiprocessing for the sliding window. If True, uses # CPU cores.
    :return: number of parallel workers to instantiate
    """
    # (Daemonic processes are not allowed to have children)
    if current_process().name != 'MainProcess':
        return 1

    # return CPU count if not specified
    if parallelize_arg is None:
        return cpu_count()

    # return CPU count if True
    if isinstance(parallelize_arg, bool):
        if parallelize_arg:
            return cpu_count()
        else:
            return 1

    # Otherwise return numeric inputs (but no more than the number of cores available)
    return min(cpu_count(), parallelize_arg)


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


def benchmark_process_queue(*args, worker_counts: List[int] = None, verbose: bool = True,
                            disable_benchmark_progress_bar: bool = None, **kwargs) -> Dict[int, float]:
    """
    Helper function that wraps multiprocessing and measures how number of workers impacts execution time

    :param args: args for multiprocessing (e.gs func, iterable)
    :param worker_counts: list of counts to try (if not specified, uses powers of two up to core count)
    :param verbose: whether the benchmarking should be verbose
    :param disable_benchmark_progress_bar: pass true to disable benchmarking bar
    :param kwargs: kwargs for multiprocessing (such as pool_function)
    :return: dictionary with worker counts for keys and performance time in seconds for the values
    """
    if not worker_counts:
        worker_counts = [2 ** x for x in range(int(math.log2(cpu_count())) + 1)][::-1]

    benchmarks: Dict[int, float] = dict()
    try:
        for num_workers in tqdm(worker_counts, disable=disable_benchmark_progress_bar):
            if verbose:
                print(f'Beginning benchmark with {num_workers} workers...')
            tic: float = time.perf_counter()
            process_queue(*args, num_workers=num_workers, **kwargs)
            benchmarks[num_workers] = (duration := time.perf_counter() - tic)
            if verbose:
                print(f'... completed in {duration:.4f} seconds')
    except KeyboardInterrupt:
        pass  # allow early break out of the loop if desired

    return benchmarks


def multiprocess(*args, suppress_multiprocess_notice: bool = False, **kwargs) -> List[Any]:
    """ Legacy name wrapper for process_queue with a warning that can be silenced """
    if not suppress_multiprocess_notice:
        print("'multiprocess' is now 'process_queue'. Update or pass suppress_multiprocess_notice=True to silence.")
    return process_queue(*args, **kwargs)


def recursive_process(func: Callable, initial_inputs: Any, max_deep: int = None,
                      print_progress: bool = False, print_current_value: bool = False, *args, **kwargs) -> Any:
    """
    Helper function for processing recursive functions

    :param func: recursive function
    :param initial_inputs: initial inputs to the function
    :param max_deep: max iterations through the loop
    :param print_progress: whether to log information like depth and timing
    :param print_current_value: whether to print stepwise value (might be ok for int or str, but avoid if big / complex)
    :param args: positional arguments for the function
    :param kwargs: keyword arguments for the function
    :return: latest state at time of keyboard interrupt or maxing out the depth
    """
    counter: int = 0
    current: Any = initial_inputs
    try:
        while (max_deep is None) or (counter < max_deep):
            tic: float = time.perf_counter()
            current = func(current, *args, **kwargs)
            counter += 1
            if print_progress:
                print(f"Completed cycle #{counter} in {time.perf_counter() - tic:.6f} seconds")
            if print_current_value:
                print(f"... Current value: {current}")
    except KeyboardInterrupt:
        print(f"Breaking for keyboard interrupt after {counter} cycles.")
    except Exception as e:
        print(f"After {counter} cycles, encountered exception {e}")
    return current


def process_queue(func: Callable, iterable: List[Any], pool_function: str = None,
                  batching: bool = False, num_workers: int = None, cuts_per_worker: int = 3,
                  serial_progress_bar: bool = True, *_, **kwargs) -> List[Any]:
    """
    Convenience wrapper for Pool.map and Pool.starmap that offers manual batching and automatic flattening

    :param func: function to be evaluated in parallel
    :param iterable: can be map or starmap style iterable of inputs to func
    :param pool_function: whether to use 'map' or 'starmap'. Will try to infer if not provided.
    :param batching: whether to handle batching manually
    :param num_workers: number of processes to run in parallel (should not exceed CPU core count)
    :param cuts_per_worker: if batching manually, can break into extra chunks to avoid idle time even if runtimes vary
    :param serial_progress_bar: whether to show the progress bar if processing in serial
    :param kwargs: additional keyword arguments for 'map' or 'starmap' (for example the max chunk size)
    :return: the results of func evaluated over the iterable
    """
    # Get num workers if not supplied
    if not num_workers:
        num_workers = get_num_workers(num_workers)

    # Attempt to infer the pool function to be used, if not specified
    # This is NOT infallible! Suppose you want to 'map' over a list of tuples; it would infer that 'starmap' is desired.
    if not pool_function:
        if isinstance(iterable[0], Iterable) and (not isinstance(iterable[0], str)):
            pool_function: str = 'starmap'
        else:
            pool_function: str = 'map'

    # If only 1 worker, do the work in serial since we don't need the pool and its overhead
    if num_workers == 1:
        return [func(i) for i in tqdm(iterable, disable=None if serial_progress_bar else True)]

    # Break the task list into batches if desired
    if batching:
        global batched_func  # ... TODO: how to make this pickleable without making it global??

        def batched_func(tasks: List[Any]) -> List[Any]:  # noqa:
            return [func(args) for args in tasks]

        if pool_function == 'starmap':
            raise NotImplementedError(f"batching and starmap are mutually exclusive in the current implementation")

        map_style_inputs: List[List[Any]] = divvy_workload(num_workers=num_workers * cuts_per_worker, tasks=iterable)
        use_function: Callable = batched_func
    else:
        # No batching = batches of one (let the pool methods handle chunksize)
        map_style_inputs: List[Any] = iterable
        use_function = func

    # Run the pool
    with Pool(min(num_workers, len(map_style_inputs))) as pool:
        if pool_function.lower() == 'starmap':
            result = pool.starmap(func=use_function, iterable=map_style_inputs, **kwargs)
        elif pool_function.lower() == 'map':
            result = pool.map(func=use_function, iterable=map_style_inputs, **kwargs)
        else:
            raise ValueError(f"Pool function should be 'map' or 'starmap' but received unknown type: {pool_function}")

    # Flatten if necessary
    if batching:
        result = [item for sublist in result for item in sublist]

    return result  # noqa: results prouced in lines above


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
    """ Reckless helper function that tries to cast the input to a dictionary or number (float) or boolean or None. """

    # If not a string, send it back
    if not isinstance(x, str):
        return x

    # Try to parse model representations
    if all(substr in x for substr in ['=', ',']):
        try:
            return {k[0]: risky_cast(k[1]) for k in [j.split('=') for j in x.split(', ')]}  # "a=5, b=6"
        except:
            pass

    if x.replace('.', '', 1).isdigit() or (x[0] == '-' and x[1:].replace('.', '', 1).isdigit()):
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

    # Bool? None?
    mapper: Dict[str, Any] = {'true': True, 'false': False, 'none': None}
    if x.lower() in mapper:
        return mapper[x.lower()]

    # Give up
    return x


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Helper function that flattens a list, nothing fancy (only works 1 level deep)
    :param nested_list: a list of lists of <something>
    :return: a list of <something>
    """
    return [item for sublist in nested_list for item in sublist]
