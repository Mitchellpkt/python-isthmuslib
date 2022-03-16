import pathlib
from typing import Union, Dict, Any, List, Tuple
import pandas as pd
from tqdm.auto import tqdm
from src.isthmuslib.vectors import VectorSequence, VectorMultiset
from src.isthmuslib.utils import divvy_workload, get_num_workers
from pydantic import BaseModel


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


def auto_extract_from_text(input_string: str, return_type: str = 'dataframe', left_token: str = None,
                           key_value_delimiter: str = None, right_token: str = None, basis_col_name: str = None,
                           record_delimiter: str = None, parallelize: Union[bool, int] = False,
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
    :param parallelize: whether to parallelize calculations. Can be an integer (# workers) or bool True / False
    :param disable_progress_bar: pass anything True to silence the progress bar
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """

    # Use the default tokens if not specified
    for token in ['left_token', 'right_token', 'key_value_delimiter', 'record_delimiter']:
        if not locals().get(token):
            locals()[token] = getattr(LogIO(), token)

    df_output: pd.DataFrame = pd.DataFrame()
    record_chunks: List[str] = input_string.split(record_delimiter)[1:]
    chunk_buffers: List[Dict[str, Any]] = []

    def chunk_processor_lambda(chunk: str):
        chunk_buffer: Dict[str, Any] = dict()
        raw_breaks: List[str] = chunk.split(left_token)[1:]
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
            chunk_buffer.setdefault(key, value)  # Add the value to this row
        return chunk_buffer

    # Process the chunks
    num_workers: int = get_num_workers(parallelize_arg=parallelize)
    if parallelize and (num_workers > 1):
        # Parallel processing
        pass
    else:
        # Serial processing
        for chunk in tqdm(record_chunks, disable=disable_progress_bar):
            chunk_buffers.append(chunk_processor_lambda(chunk))

    # Convert the buffers to a dataframe
    for chunk_buffer in chunk_buffers:
        if chunk_buffer:
            df_output = pd.concat([df_output, pd.DataFrame(chunk_buffer, index=[-1])], ignore_index=True)

    if 'dataframe' in return_type.lower():
        return df_output
    elif 'multiset' in return_type.lower():
        return VectorMultiset(data=df_output)
    elif 'sequence' in return_type.lower():
        return VectorSequence(data=df_output, basis_col_name=basis_col_name)


def auto_extract_from_file(file_path: Union[str, pathlib.Path], record_delimiter: str = None,
                           left_token: str = None, key_value_delimiter: str = None, right_token: str = None,
                           basis_col_name: str = None, disable_progress_bar: bool = None,
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
    :return: pandas.DataFrame or VectorMultiset or VectorSequence
    """

    # Use the default tokens if not specified
    for token in ['left_token', 'right_token', 'key_value_delimiter', 'record_delimiter']:
        if not locals().get(token):
            locals()[token] = getattr(LogIO(), token)

    # Read in the file
    with open(file_path, 'r') as f:
        input_string: str = f.read()

    return auto_extract_from_text(input_string=input_string, return_type=return_type,
                                  record_delimiter=record_delimiter, left_token=left_token,
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
