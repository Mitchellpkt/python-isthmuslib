import pathlib
from typing import Union, Dict, Any, List, Tuple
import pandas as pd
from tqdm.auto import tqdm
from .vectors import VectorSequence, VectorMultiset
from .utils import divvy_workload, get_num_workers
from pydantic import BaseModel
from multiprocessing import Pool
from copy import deepcopy


class LogIO(BaseModel):
    record_delimiter: str = '[@@@]'
    left_token: str = '[<<'
    key_value_delimiter: str = '='
    right_token: str = '>>]'
    parallelize_imports: bool = True

    def single_feature_to_log(self, key: str, value: str) -> str:
        return f"{self.left_token}{key}{self.key_value_delimiter}{value}{self.right_token}"

    def dict_to_log(self, key_value_dict: Dict[str, Any], include_delimiter: bool = False) -> str:
        s: str = self.record_delimiter if include_delimiter else ''
        for key, value in key_value_dict.items():
            s += f"{self.single_feature_to_log(key=key, value=value)}"
        return s

    def auto_extract_from_text(self, input_string: str, return_type: str = 'dataframe', left_token: str = None,
                               key_value_delimiter: str = None, right_token: str = None,
                               basis_col_name: str = None, disable_progress_bar: bool = None,
                               record_delimiter: str = None) -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
        """
        Extracts a data frame from a string (wrapper for standalone function below)

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
        :param disable_progress_bar: pass anything True to silence the progress bar
        :return: pandas.DataFrame or VectorMultiset or VectorSequence
        """
        return auto_extract_from_text(
            input_string=input_string, return_type=return_type,
            basis_col_name=basis_col_name,
            disable_progress_bar=disable_progress_bar,
            left_token=left_token if left_token else self.left_token,
            right_token=right_token if right_token else self.right_token,
            key_value_delimiter=key_value_delimiter if key_value_delimiter else self.key_value_delimiter,
            record_delimiter=record_delimiter if record_delimiter else self.record_delimiter
        )

    def auto_extract_from_file(self, file_path: Union[str, pathlib.Path], record_delimiter: str = None,
                               left_token: str = None, key_value_delimiter: str = None, right_token: str = None,
                               basis_col_name: str = None, disable_progress_bar: bool = None,
                               return_type: str = 'dataframe') -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
        """
        Extracts a data frame from a file (wrapper for standalone function below)

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
        :return: pandas.DataFrame or VectorMultiset or VectorSequence
        """

        # Read in the file
        with open(file_path, 'r') as f:
            input_string: str = f.read()

        return self.auto_extract_from_text(input_string=input_string, return_type=return_type,
                                           record_delimiter=record_delimiter, left_token=left_token,
                                           key_value_delimiter=key_value_delimiter, right_token=right_token,
                                           basis_col_name=basis_col_name, disable_progress_bar=disable_progress_bar)


def chunk_processor_lambda(input_chunk: str, left_token: str = None, key_value_delimiter: str = None,
                           right_token: str = None) -> Dict[str, Any]:
    """ Helper lambda so we can do this in parallel or series """
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
        value: str = ''.join([x + key_value_delimiter for x in substrings[1:]])[:-1]
        output_chunk_dict.setdefault(key, value)  # Add the value to this row
    return output_chunk_dict


def multi_chunk_processor_lambda(args: Tuple[List[str], str, str, str]) -> List[Dict[str, Any]]:
    """ Helper function that works through a queue of chunks (for parallelization) """
    input_chunks, left_token, key_value_delimiter, right_token = args
    return [chunk_processor_lambda(x, left_token=left_token, key_value_delimiter=key_value_delimiter,
                                   right_token=right_token) for x in input_chunks]


def auto_extract_from_text(input_string: str, return_type: str = 'dataframe', left_token: str = None,
                           key_value_delimiter: str = None, right_token: str = None, basis_col_name: str = None,
                           record_delimiter: str = None, parallelize_read: Union[bool, int] = False,
                           parallelize_processing: Union[bool, int] = False,
                           disable_progress_bar: bool = None) -> Union[pd.DataFrame, VectorSequence, VectorMultiset]:
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
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """
    times_ns: List[int] = []
    # # Use the default tokens if not specified
    # for token in ['left_token', 'right_token', 'key_value_delimiter', 'record_delimiter']:
    #     if not locals().get(token):
    #         locals()[token] = getattr(LogIO(), token)
    # TODO: Troubleshoot the above later. Meanwhile:
    if not left_token:
        left_token = LogIO().left_token
    if not right_token:
        right_token = LogIO().right_token
    if not key_value_delimiter:
        key_value_delimiter = LogIO().key_value_delimiter
    if not record_delimiter:
        record_delimiter = LogIO().record_delimiter

    record_chunks: List[str] = input_string.split(record_delimiter)[1:]
    chunk_buffers: List[Dict[str, Any]] = []

    # Process the chunks
    num_workers: int = get_num_workers(parallelize_arg=parallelize_read)
    if parallelize_read and (num_workers > 1):
        batches: List[List[Any]] = divvy_workload(num_workers=num_workers, tasks=record_chunks)
        i: List[Tuple[List[str], str, str, str]] = [(b, left_token, key_value_delimiter, right_token) for b in batches]
        with Pool(num_workers) as pool:
            chunk_buffers_nested: List[List[Dict[str, Any]]] = pool.map(func=multi_chunk_processor_lambda, iterable=i)
        chunk_buffers: List[Dict[str, Any]] = [item for sublist in chunk_buffers_nested for item in sublist]

    else:
        # Serial processing
        for chunk in (p1 := tqdm(record_chunks, disable=disable_progress_bar)):
            p1.set_description('Scanning file (step 1 of 2)')
            chunk_buffers.append(chunk_processor_lambda(chunk, left_token=left_token, right_token=right_token,
                                                        key_value_delimiter=key_value_delimiter))

    # # I would expect this to be the cleaner way to implement flattening, but the performance of the next block wins.
    # # However I might experiment with this later to parallelize the flattening specifically, to see if that helps.
    # all_keys_nested: List[List[str]] = [list(buffer.keys()) for buffer in chunk_buffers]
    # all_keys_flat: List[str] = [item for sublist in all_keys_nested for item in sublist]
    # # (this next line is the one that is very slow at scale)
    # reshaped: Dict[str, List[Any]] = {key: [chunk.get(key, None) for chunk in chunk_buffers] for key in all_keys_flat}
    # df_output: pd.DataFrame = pd.DataFrame(reshaped)

    num_reshape_workers: int = get_num_workers(parallelize_arg=parallelize_processing)
    if parallelize_processing and (num_reshape_workers > 1):
        batches: List[List[Any]] = divvy_workload(num_workers=num_reshape_workers, tasks=chunk_buffers)
        with Pool(num_reshape_workers) as pool:
            dataframes: List[pd.DataFrame] = pool.map(func=dicts_to_dataframe, iterable=batches)
        df: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    else:
        df: pd.DataFrame = pd.DataFrame()
        for chunk_buffer in (p2 := tqdm(chunk_buffers, disable=disable_progress_bar)):
            p2.set_description('Reshaping data (step 2 of 2)')
            if chunk_buffer:
                df: pd.DataFrame = pd.concat([df, pd.DataFrame(chunk_buffer, index=[-1])], ignore_index=True)

    if 'dataframe' in return_type.lower():
        return df
    elif 'multiset' in return_type.lower():
        return VectorMultiset(data=df)
    elif 'sequence' in return_type.lower():
        return VectorSequence(data=df, basis_col_name=basis_col_name)


def dicts_to_dataframe(dictionaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Helper function that converts a list of dictionaries into a dataframe (each dictionary = 1 row)

    :param dictionaries: list of dictionaries, with one value per key per row
    :return: dataframe representation
    """
    df: pd.DataFrame = pd.DataFrame()
    for dictionary in [d for d in dictionaries if d]:
        df = pd.concat([df, pd.DataFrame(dictionary, index=[-1])], ignore_index=True)
    return df


def batch_dicts_to_dataframe(dictionaries: List[Dict[str, Any]], batch_dicts_to_dataframe: int = 1000) -> pd.DataFrame:
    """
    Helper function that converts a list of dictionaries into a dataframe (each dictionary = 1 row)

    :param dictionaries: list of dictionaries, with one value per key per row
    :param batch_dicts_to_dataframe: how many rows to batch together
    :return: dataframe representation
    """

    batches: List[pd.DataFrame] = []
    df_in_progress: pd.DataFrame = pd.DataFrame()
    for i, dictionary in enumerate([d for d in dictionaries if d]):
        df_in_progress = pd.concat([df_in_progress, pd.DataFrame(dictionary, index=[-1])], ignore_index=True)
        if i % batch_dicts_to_dataframe == 0:
            batches.append(deepcopy(df_in_progress))
            df_in_progress = pd.DataFrame()

    return pd.concat(batches, ignore_index=True)


def auto_extract_from_file(file_path: Union[str, pathlib.Path], record_delimiter: str = None,
                           left_token: str = None, key_value_delimiter: str = None, right_token: str = None,
                           basis_col_name: str = None, disable_progress_bar: bool = None,
                           parallelize_read: Union[bool, int] = False, parallelize_processing: Union[bool, int] = False,
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

    # # Use the default tokens if not specified
    # for token in ['left_token', 'right_token', 'key_value_delimiter', 'record_delimiter']:
    #     if not locals().get(token):
    #         locals()[token] = getattr(LogIO(), token)
    # TODO: Troubleshoot the above later. Meanwhile:
    if not left_token:
        left_token = LogIO().left_token
    if not right_token:
        right_token = LogIO().right_token
    if not key_value_delimiter:
        key_value_delimiter = LogIO().key_value_delimiter
    if not record_delimiter:
        record_delimiter = LogIO().record_delimiter

    # Read in the file
    with open(file_path, 'r') as f:
        input_string: str = f.read()

    return auto_extract_from_text(input_string=input_string, return_type=return_type,
                                  record_delimiter=record_delimiter, left_token=left_token,
                                  parallelize_read=parallelize_read, parallelize_processing=parallelize_processing,
                                  key_value_delimiter=key_value_delimiter, right_token=right_token,
                                  basis_col_name=basis_col_name, disable_progress_bar=disable_progress_bar)


def extract_text_to_dataframe(input_string: str, tokens_dictionary: Dict[str, Tuple[str, str]],
                              record_delimiter: str = '[@@@]', disable_progress_bar: bool = None) -> pd.DataFrame:
    """
    Extracts a pandas dataframe from a string

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: Vector data set extracted from the input string
    """
    df_output: pd.DataFrame = pd.DataFrame()
    for chunk in tqdm(input_string.split(record_delimiter), disable=disable_progress_bar):
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
                           record_delimiter: str = '[@@@]', disable_progress_bar: bool = None,
                           basis_col_name: str = None) -> Union[VectorMultiset, VectorSequence]:
    """
    Extracts a VectorMultiset from a string (or a VectorSequence if you specify `basis_col_name`)

    :param input_string: string (e.g. logs file) to be parsed
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: Vector data set extracted from the input string
    """
    df_output: pd.DataFrame = extract_text_to_dataframe(input_string=input_string, record_delimiter=record_delimiter,
                                                        tokens_dictionary=tokens_dictionary,
                                                        disable_progress_bar=disable_progress_bar)
    if basis_col_name:
        return VectorSequence(data=df_output, basis_col_name=basis_col_name)
    else:
        return VectorMultiset(data=df_output)


def extract_file_to_vector(file_path: Union[str, pathlib.Path], record_delimiter: str,
                           tokens_dictionary: Dict[str, Tuple[str, str]], disable_progress_bar: bool = None,
                           basis_col_name: str = None) -> Union[VectorSequence, VectorMultiset]:
    """
    Wrapper for extract_from_text that pulls vector data out of a file, such as raw log output

    :param file_path: file to read
    :param record_delimiter: The string that should be used to chunk up the string into observations / rows
    :param tokens_dictionary: Extraction rules. Key = label for column, value = (before token, after token)
    :param basis_col_name: Optional - if specified returns a VectorSequence
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: Vector data set extracted from the input string
    """
    with open(file_path, 'r') as f:
        return extract_text_to_vector(input_string=f.read(), record_delimiter=record_delimiter,
                                      tokens_dictionary=tokens_dictionary, basis_col_name=basis_col_name,
                                      disable_progress_bar=disable_progress_bar)
