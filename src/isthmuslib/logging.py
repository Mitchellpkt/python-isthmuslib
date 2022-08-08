from typing import Union, Dict, Any, List, Tuple
import pandas as pd
from tqdm.auto import tqdm
from .utils import divvy_workload, get_num_workers, process_queue
from pydantic import BaseModel
from multiprocessing import Pool
from copy import deepcopy
import json
from .vectors import VectorSequence, VectorMultiset
import pathlib


class LogIO(BaseModel):
    record_delimiter: str = "[@@@]"
    left_token: str = "[<<"
    key_value_delimiter: str = "="
    right_token: str = ">>]"
    parallelize_imports: bool = True
    embedded_json_line_prefix: str = "[[EMBEDDED_JSON_LINE]]"
    embedded_csv_line_prefix: str = "[[EMBEDDED_CSV_LINE]]"  # spoiler alert
    log_formatter: str = "\n@@ {time:x} AT: {time} | LEVEL: {level} | IN: {name}.{function}\n\n{message} |\n"  # TODO: unify this object's string fields with Config?? # noqa

    def single_feature_to_log(self, key: str, value: str) -> str:
        return (
            f"{self.left_token}{key}{self.key_value_delimiter}{value}{self.right_token}"
        )

    def dict_to_log(
        self, key_value_dict: Dict[str, Any], include_delimiter: bool = False
    ) -> str:
        s: str = self.record_delimiter if include_delimiter else ""
        for key, value in key_value_dict.items():
            s += f"{self.single_feature_to_log(key=key, value=value)}"
        return s


def parse_string_with_manual_tokens(
    input_string: str,
    tokens_dictionary: Dict[str, Tuple[str, str]],
    limit: int = None,
    record_delimiter: str = "[@@@]",
    disable_progress_bar: bool = False,
    parallelize_processing: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Extracts a pandas dataframe from a string

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param limit: maximum number of rows to process
    :return: Vector data set extracted from the input string
    """

    # Initialize, split the string, and trim the list if longer than 'limit'
    row_buffers_list: List[Dict[str, Any]] = []
    raw_row_texts_list: List[str] = input_string.split(record_delimiter)
    if limit and (limit < len(raw_row_texts_list)):
        raw_row_texts_list = raw_row_texts_list[:limit]

    # Loop over rows
    p = tqdm(raw_row_texts_list, disable=disable_progress_bar)
    if not disable_progress_bar:
        p.set_description("Reading and parsing")

    for raw_row_text in p:
        row_buffer: Dict[str, Any] = dict()

        # Look for each key in this row
        for key, (before_token, after_token) in tokens_dictionary.items():
            if before_token in raw_row_text:
                target_substring = raw_row_text.split(before_token)[1]
                if after_token in target_substring:
                    row_buffer.setdefault(key, target_substring.split(after_token)[0])
        if row_buffer:
            row_buffers_list.append(row_buffer)

    # Convert the list of dictionaries to a dataframe (note: this is a slow step that I plan to optimize later)
    return list_of_dict_to_dataframe(
        data_point_dicts=row_buffers_list,
        disable_progress_bar=disable_progress_bar,
        parallelize_processing=parallelize_processing,
    )


def parse_file_with_manual_tokens(path: str, *args, **kwargs):
    """Wrapper for above function (parse_string_with_manual_tokens) that applies it to a file"""
    with open(path, "r") as f:
        return parse_string_with_manual_tokens(f.read(), *args, **kwargs)


def parse_string_with_key_value_delimiters(
    input_string: str,
    left_token: str = None,
    key_value_delimiter: str = None,
    right_token: str = None,
    record_delimiter: str = None,
    parallelize_read: Union[bool, int] = False,
    parallelize_processing: Union[bool, int] = False,
    limit: int = None,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
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
    :param left_token: left side of a record
    :param key_value_delimiter: marker between the variable name and its value
    :param right_token: right side of a record
    :param parallelize_read: whether to parallelize reading. Can be an integer (# workers) or bool True / False
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param limit: maximum number of rows to process
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """

    # Use LogIO defaults for unspecified tokens
    if not left_token:
        left_token = LogIO().left_token
    if not right_token:
        right_token = LogIO().right_token
    if not key_value_delimiter:
        key_value_delimiter = LogIO().key_value_delimiter
    if not record_delimiter:
        record_delimiter = LogIO().record_delimiter

    # Initialize, split the string, and trim the list if longer than 'limit'
    row_buffers_list: List[Dict[str, Any]] = []
    raw_row_texts_list: List[str] = input_string.split(record_delimiter)
    if limit and (limit < len(raw_row_texts_list)):
        raw_row_texts_list = raw_row_texts_list[:limit]

    # Process the chunks
    num_workers: int = get_num_workers(parallelize_arg=parallelize_read)
    if parallelize_read and (num_workers > 1):
        batches: List[List[Any]] = divvy_workload(
            num_workers=num_workers, tasks=raw_row_texts_list
        )
        i: List[Tuple[List[str], str, str, str]] = [
            (b, left_token, key_value_delimiter, right_token) for b in batches
        ]
        with Pool(num_workers) as pool:
            chunk_buffers_nested: List[List[Dict[str, Any]]] = pool.map(
                func=multi_key_value_extraction_lambda, iterable=i
            )
        row_buffers_list: List[Dict[str, Any]] = [
            item for sublist in chunk_buffers_nested for item in sublist
        ]

    else:
        # Serial processing
        # Loop over rows
        p1 = tqdm(raw_row_texts_list, disable=disable_progress_bar)
        if not disable_progress_bar:
            p1.set_description("Scanning text")

        for chunk in p1:
            row_buffers_list.append(
                key_value_extraction_lambda(
                    chunk,
                    left_token=left_token,
                    right_token=right_token,
                    key_value_delimiter=key_value_delimiter,
                )
            )

    return list_of_dict_to_dataframe(
        row_buffers_list,
        disable_progress_bar=disable_progress_bar,
        parallelize_processing=parallelize_processing,
    )


def parse_file_with_key_value_delimiters(path: str, *args, **kwargs):
    """Wrapper for above function (parse_string_with_key_value_delimiters) that applies it to a file"""
    with open(path, "r") as f:
        return parse_string_with_key_value_delimiters(f.read(), *args, **kwargs)


def parse_string_with_embedded_json_unprocessed_dicts(
    input_string: str,
    embedded_json_line_prefix: str = None,
    limit: int = None,
    ignore_decode_errors: bool = False,
    end_of_line: str = "\n",
) -> List[Dict[Any, Any]]:
    """
    Extracts embedded json from log files

    :param input_string: string (e.g. logs file) to be parsed
    :param embedded_json_line_prefix: marker string at the start of every embedded json line
    :param ignore_decode_errors: if True, silently ignores rows that don't parse (e.g. if pulling from incomplete row)
    :param end_of_line: substring between rows
    :param limit: maximum number of rows to process
    """

    # Use LogIO defaults for unspecified tokens
    if not embedded_json_line_prefix:
        embedded_json_line_prefix = LogIO().embedded_json_line_prefix

    # Initialize, split the string, and trim the list if longer than 'limit'
    if embedded_json_line_prefix not in input_string:
        return []

    # Pull out and clean the embedded data
    embedded_with_trailing_data = input_string.split(embedded_json_line_prefix)[1:]
    if limit and (limit < len(embedded_with_trailing_data)):
        embedded_with_trailing_data = embedded_with_trailing_data[:limit]
    embedded_clean = [
        x.split(end_of_line)[0] if end_of_line in x else x
        for x in embedded_with_trailing_data
    ]

    return_list: List[Dict[Any, Any]] = []
    for s in embedded_clean:
        try:
            return_list.append(json.loads(s))
        except json.JSONDecodeError as e:
            if not ignore_decode_errors:
                raise e
    return return_list


def parse_string_with_embedded_json(
    input_string: str,
    embedded_json_line_prefix: str = None,
    parallelize_processing: Union[bool, int] = False,
    disable_progress_bar: bool = False,
    limit: int = None,
    end_of_line: str = "\n",
) -> pd.DataFrame:
    """
    Extracts embedded json from log files

    :param input_string: string (e.g. logs file) to be parsed
    :param embedded_json_line_prefix: marker string at the start of every embedded json line
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param end_of_line: substring between rows
    :param limit: maximum number of rows to process
    """
    extracted_json_dicts: List[
        Dict[str, Any]
    ] = parse_string_with_embedded_json_unprocessed_dicts(
        input_string=input_string,
        embedded_json_line_prefix=embedded_json_line_prefix,
        limit=limit,
        end_of_line=end_of_line,
    )
    return list_of_dict_to_dataframe(
        extracted_json_dicts,
        disable_progress_bar=disable_progress_bar,
        parallelize_processing=parallelize_processing,
    )


def parse_file_with_embedded_json(path: str, *args, **kwargs) -> pd.DataFrame:
    """Wrapper for above (parse_string_with_embedded_json) function that applies it to a file"""
    with open(path, "r") as f:
        return parse_string_with_embedded_json(f.read(), *args, **kwargs)


def key_value_extraction_lambda(
    input_chunk: str,
    left_token: str = None,
    key_value_delimiter: str = None,
    right_token: str = None,
) -> Dict[str, Any]:
    """Helper lambda so we can do this in parallel or series"""
    output_chunk_dict: Dict[str, Any] = dict()
    raw_breaks: List[str] = input_chunk.split(left_token)[1:]
    middle: List[str] = [x.split(right_token)[0] for x in raw_breaks]
    # Loop over possible tokens
    for entry in middle:
        if key_value_delimiter not in entry:
            continue  # Do not attempt to parse if the delimiter is not present
        substrings: List[str] = entry.split(key_value_delimiter)
        key: str = substrings[0]
        if not key:
            continue  # Do not continue to parse if the key is known
        value: str = "".join([x + key_value_delimiter for x in substrings[1:]])[:-1]
        output_chunk_dict.setdefault(key, value)  # Add the value to this row
    return output_chunk_dict


def multi_key_value_extraction_lambda(
    args: Tuple[List[str], str, str, str]
) -> List[Dict[str, Any]]:
    """Helper function that works through a queue of chunks (for parallelization)"""
    input_chunks, left_token, key_value_delimiter, right_token = args
    return [
        key_value_extraction_lambda(
            x,
            left_token=left_token,
            key_value_delimiter=key_value_delimiter,
            right_token=right_token,
        )
        for x in input_chunks
    ]


def data_frame_init_wrapper(data: Any) -> pd.DataFrame:
    """
    Temp thin helper function to wrap pandas init (this only exists to be handy for the next function)

    :param data: data for the dataframe, e.g. a list of dictionaries
    :return: the pandas dataframe
    """
    return pd.DataFrame(data, index=range(len(data)))


def list_of_dict_to_dataframe(
    data_point_dicts,
    parallelize_processing: Union[bool, int] = True,
    disable_progress_bar: bool = False,
    pandas_automatic: bool = True,
    suppress_deprecation_warning: bool = False,
) -> pd.DataFrame:
    """
     Extracts a data frame from a string

    Input: "the [@@@] quick [<<x='foo'>>] brown[<<y=9>>] [@@@]fox [<<y=93>>]"
    Output: (data frame)
                   x   y
            0  'foo'   9
            1    NaN  93

    Hint, write your logs using python fstrings like f"Foo [<<{x=}>>] and [<<{y=}>>]"

    :param data_point_dicts: list of dictionaries to be added to a dataframe
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param pandas_automatic: Throw everything into pan
    :param suppress_deprecation_warning: Nudge users to adopt more efficient method, pass "True" to silence
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """
    num_reshape_workers: int = get_num_workers(parallelize_arg=parallelize_processing)

    if pandas_automatic:
        if not parallelize_processing:
            return pd.DataFrame(data_point_dicts)
        if parallelize_processing:
            if not disable_progress_bar:
                print(
                    f"Reshaping data (step 1 of 2): processing in parallel with {num_reshape_workers} workers"
                )
            batches_of_dictionaries: List[List[Any]] = divvy_workload(
                num_workers=num_reshape_workers, tasks=data_point_dicts
            )
            list_of_dataframes: List[pd.DataFrame] = process_queue(
                data_frame_init_wrapper,
                batches_of_dictionaries,
                pool_function="map",
                batching=False,  # here, we batch manually above rather than inside multiprocess
                num_workers=num_reshape_workers,
            )
            if not disable_progress_bar:
                print("Reshaping data (step 2 of 2): attaching subframes")
            return pd.concat(list_of_dataframes, ignore_index=True)

    if not suppress_deprecation_warning:
        print(f"You are using a very slow method for reshaping the data")
        print(
            f"Recommendation: use list_of_dict_to_dataframe(..., pandas_automatic=True)"
        )
        print(
            f"To suppress this warning, pass: use list_of_dict_to_dataframe(..., suppress_deprecation_warning=True)"
        )

    if parallelize_processing and (num_reshape_workers > 1):
        batches: List[List[Any]] = divvy_workload(
            num_workers=num_reshape_workers, tasks=data_point_dicts
        )
        if not disable_progress_bar:
            print(
                f"Reshaping data: processing in parallel with {num_reshape_workers} workers"
            )
        with Pool(num_reshape_workers) as pool:
            dataframes: List[pd.DataFrame] = pool.map(
                func=dicts_to_dataframe, iterable=batches
            )
        df: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    else:
        df: pd.DataFrame = pd.DataFrame()
        # THIS LOOP IS SO SLOW - you should use pandas_automatic=True
        for chunk_buffer in (
            p2 := tqdm(data_point_dicts, disable=disable_progress_bar)
        ):
            p2.set_description("Reshaping data:")
            if chunk_buffer:
                df: pd.DataFrame = pd.concat(
                    [df, pd.DataFrame(chunk_buffer, index=[-1])], ignore_index=True
                )
    return df


def dicts_to_dataframe(dictionaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    NB: THIS IS AN INEFFICIENT FUNCTION THAT SCALES AWFULLY, PLEASE USE list_of_dict_to_dataframe instead
    Helper function that converts a list of dictionaries into a dataframe (each dictionary = 1 row)

    :param dictionaries: list of dictionaries, with one value per key per row
    :return: dataframe representation
    """
    df: pd.DataFrame = pd.DataFrame()
    for dictionary in [d for d in dictionaries if d]:
        df = pd.concat([df, pd.DataFrame(dictionary, index=[-1])], ignore_index=True)
    return df


def batch_dicts_to_dataframe(
    dictionaries: List[Dict[str, Any]], batch_size: int = 1000
) -> pd.DataFrame:
    """
    NB: THIS IS AN INEFFICIENT FUNCTION THAT SCALES AWFULLY, PLEASE USE list_of_dict_to_dataframe instead
    Helper function that converts a list of dictionaries into a dataframe (each dictionary = 1 row)

    :param dictionaries: list of dictionaries, with one value per key per row
    :param batch_size: how many rows to batch together
    :return: dataframe representation
    """

    batches: List[pd.DataFrame] = []
    df_in_progress: pd.DataFrame = pd.DataFrame()
    for i, dictionary in enumerate([d for d in dictionaries if d]):
        df_in_progress = pd.concat(
            [df_in_progress, pd.DataFrame(dictionary, index=[-1])], ignore_index=True
        )
        if i % batch_size == 0:
            batches.append(deepcopy(df_in_progress))
            df_in_progress = pd.DataFrame()

    return pd.concat(batches, ignore_index=True)


#####################################
# EVERYTHING BELOW THIS LINE IS
# LEGACY AND SHOULD BE DEPRECATED
# (PLEASE INSTEAD USE ABOVE FUNCTIONS)
#####################################


# LEGACY FUNCTION - New use of this function is not recommended, however it works just fine
def auto_extract_from_text(
    input_string: str,
    return_type: str = "dataframe",
    left_token: str = None,
    key_value_delimiter: str = None,
    right_token: str = None,
    basis_col_name: str = None,
    record_delimiter: str = None,
    parallelize_read: Union[bool, int] = False,
    parallelize_processing: Union[bool, int] = False,
    limit: int = None,
    disable_progress_bar: bool = False,
) -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
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
    :param left_token: left side of a record
    :param key_value_delimiter: marker between the variable name and its value
    :param right_token: right side of a record
    :param basis_col_name: optional, specify a basis column name for a VectorSequence (otherwise ignored)
    :param return_type: what format to return the data in
    :param parallelize_read: whether to parallelize reading. Can be an integer (# workers) or bool True / False
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param limit: maximum number of rows to process
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """

    if not left_token:
        left_token = LogIO().left_token
    if not right_token:
        right_token = LogIO().right_token
    if not key_value_delimiter:
        key_value_delimiter = LogIO().key_value_delimiter
    if not record_delimiter:
        record_delimiter = LogIO().record_delimiter

    record_chunks: List[str] = input_string.split(record_delimiter)[1:]
    if limit and (limit < len(record_chunks)):
        record_chunks = record_chunks[:limit]
    chunk_buffers: List[Dict[str, Any]] = []

    # Process the chunks
    num_workers: int = get_num_workers(parallelize_arg=parallelize_read)
    if parallelize_read and (num_workers > 1):
        batches: List[List[Any]] = divvy_workload(
            num_workers=num_workers, tasks=record_chunks
        )
        i: List[Tuple[List[str], str, str, str]] = [
            (b, left_token, key_value_delimiter, right_token) for b in batches
        ]
        with Pool(num_workers) as pool:
            chunk_buffers_nested: List[List[Dict[str, Any]]] = pool.map(
                func=multi_key_value_extraction_lambda, iterable=i
            )
        chunk_buffers: List[Dict[str, Any]] = [
            item for sublist in chunk_buffers_nested for item in sublist
        ]

    else:
        # Serial processing
        for chunk in (p1 := tqdm(record_chunks, disable=disable_progress_bar)):
            p1.set_description("Scanning file (step 1 of 2)")
            chunk_buffers.append(
                key_value_extraction_lambda(
                    chunk,
                    left_token=left_token,
                    right_token=right_token,
                    key_value_delimiter=key_value_delimiter,
                )
            )

    df: pd.DataFrame = list_of_dict_to_dataframe(
        chunk_buffers,
        disable_progress_bar=disable_progress_bar,
        parallelize_processing=parallelize_processing,
    )

    if "dataframe" in return_type.lower():
        return df
    elif "multiset" in return_type.lower():
        return VectorMultiset(data=df)
    elif "sequence" in return_type.lower():
        return VectorSequence(data=df, basis_col_name=basis_col_name)


# LEGACY FUNCTION - New use of this function is not recommended, however it works just fine
def auto_extract_from_file(
    file_path: Union[str, pathlib.Path],
    record_delimiter: str = None,
    right_token: str = None,
    left_token: str = None,
    key_value_delimiter: str = None,
    return_type: str = "dataframe",
    basis_col_name: str = None,
    disable_progress_bar: bool = False,
    parallelize_read: Union[bool, int] = False,
    parallelize_processing: Union[bool, int] = False,
    **kwargs,
) -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
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
    :param left_token: left side of a record
    :param key_value_delimiter: marker between name and
    :param right_token: right side of a record
    :param basis_col_name: optional, specify a basis column name for a VectorSequence (otherwise ignored)
    :param return_type: what format to return the data in
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param parallelize_read: whether to parallelize reading. Can be an integer (# workers) or bool True / False
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """

    if not left_token:
        left_token = LogIO().left_token
    if not right_token:
        right_token = LogIO().right_token
    if not key_value_delimiter:
        key_value_delimiter = LogIO().key_value_delimiter
    if not record_delimiter:
        record_delimiter = LogIO().record_delimiter

    # Read in the file
    with open(file_path, "r") as f:
        input_string: str = f.read()

    return auto_extract_from_text(
        input_string=input_string,
        return_type=return_type,
        record_delimiter=record_delimiter,
        left_token=left_token,
        parallelize_read=parallelize_read,
        parallelize_processing=parallelize_processing,
        key_value_delimiter=key_value_delimiter,
        right_token=right_token,
        basis_col_name=basis_col_name,
        disable_progress_bar=disable_progress_bar,
        **kwargs,
    )


# LEGACY FUNCTION - New use of this function is not recommended, however it works just fine
def extract_text_to_dataframe(
    input_string: str,
    tokens_dictionary: Dict[str, Tuple[str, str]],
    limit: int = None,
    record_delimiter: str = "[@@@]",
    disable_progress_bar: bool = False,
    parallelize_processing: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Extracts a pandas dataframe from a string

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param disable_progress_bar: pass anything True to silence the progress bar
    :param parallelize_processing: whether to parallelize flattening. Can be an integer (# workers) or bool True / False
    :param limit: maximum number of rows to process
    :return: Vector data set extracted from the input string
    """
    # df_output: pd.DataFrame = pd.DataFrame()
    chunk_buffers_list: List[Dict[str, Any]] = []
    record_chunks: List[str] = input_string.split(record_delimiter)
    if limit and (limit < len(record_chunks)):
        record_chunks = record_chunks[:limit]
    for chunk in (p := tqdm(record_chunks, disable=disable_progress_bar)):
        p.set_description("Reading and parsing (step 1 of 2)")
        chunk_buffer: Dict[str, Any] = dict()
        for key, (before_token, after_token) in tokens_dictionary.items():
            if before_token in chunk:
                target_chunk = chunk.split(before_token)[1]
                if after_token in target_chunk:
                    chunk_buffer.setdefault(key, target_chunk.split(after_token)[0])
        if chunk_buffer:
            chunk_buffers_list.append(chunk_buffer)

    return list_of_dict_to_dataframe(
        data_point_dicts=chunk_buffers_list,
        disable_progress_bar=disable_progress_bar,
        parallelize_processing=parallelize_processing,
    )


# LEGACY FUNCTION - New use of this function is not recommended, however it works just fine
def extract_text_to_vector(
    input_string: str,
    tokens_dictionary: Dict[str, Tuple[str, str]],
    record_delimiter: str = "[@@@]",
    disable_progress_bar: bool = False,
    basis_col_name: str = None,
    **kwargs,
) -> Union[VectorMultiset, VectorSequence]:
    """
    Extracts a VectorMultiset from a string (or a VectorSequence if you specify `basis_col_name`)

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: Vector data set extracted from the input string
    """
    df_output: pd.DataFrame = extract_text_to_dataframe(
        input_string=input_string,
        record_delimiter=record_delimiter,
        tokens_dictionary=tokens_dictionary,
        disable_progress_bar=disable_progress_bar,
        **kwargs,
    )
    if basis_col_name:
        return VectorSequence(data=df_output, basis_col_name=basis_col_name)
    else:
        return VectorMultiset(data=df_output)


# LEGACY FUNCTION - New use of this function is not recommended, however it works just fine
def extract_file_to_vector(
    file_path: Union[str, pathlib.Path],
    record_delimiter: str,
    tokens_dictionary: Dict[str, Tuple[str, str]],
    disable_progress_bar: bool = False,
    basis_col_name: str = None,
    **kwargs,
) -> Union[VectorSequence, VectorMultiset]:
    """
    Wrapper for extract_from_text that pulls vector data out of a file, such as raw log output.

    :param file_path: file to read
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: Vector data set extracted from the input string
    """
    with open(file_path, "r") as f:
        return extract_text_to_vector(
            input_string=f.read(),
            record_delimiter=record_delimiter,
            tokens_dictionary=tokens_dictionary,
            basis_col_name=basis_col_name,
            disable_progress_bar=disable_progress_bar,
            **kwargs,
        )
