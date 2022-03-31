import pytest
from src.isthmuslib.utils import risky_cast
from typing import Tuple, Any


def test_risky_cast():
    def val_and_type(input_value: Any) -> Tuple[Any, type]:
        return type(r := risky_cast(input_value)), r

    assert (int, 5) == val_and_type('5')
    assert (float, 4.55) == val_and_type('4.55')
    assert (str, 'foo') == val_and_type('foo')
    assert (dict, {'a': 5, 'b': 6}) == val_and_type("a=5, b=6")
    assert (bool, False) == val_and_type('False')
    assert (bool, True) == val_and_type('True')
    assert (dict, {'a': 5, 'b': True}) == val_and_type("a=5, b=True")
