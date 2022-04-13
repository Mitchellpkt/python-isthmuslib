import pytest
from src.isthmuslib.utils import risky_cast, multiprocess
from src.isthmuslib.logging import parse_string_with_embedded_json
from typing import Tuple, Any, List


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


def test_extract_json():
    s: str = """The quick
    [[EMBEDDED_JSON_LINE]]{"color": "red", "value": "#f00"}
    Brown fox jumped 
    [[EMBEDDED_JSON_LINE]]{"color": "green", "value": "#0f0"}
    [[EMBEDDED_JSON_LINE]]{"color": "blue", "value": "#00f"}
    Over the foobar
    [[EMBEDDED_JSON_LINE]]{"color": "cyan", "value": "#0ff"}
    [[EMBEDDED_JSON_LINE]]{"color": "black", "value": "#000"}
    """
    extracted = parse_string_with_embedded_json(s)
    assert extracted.iloc[1, 0] == 'green'
    assert extracted.iloc[1, 1] == '#0f0'


def foobar(x: int) -> int:
    return x * 10


def starbar(x: int, y: int) -> str:
    return f"x={x} and y={y} so {x*y=}"


def test_multiprocess():
    inputs: List[int] = list(range(25))
    star_inputs: List[Tuple[int, int]] = [(x, x + 2) for x in list(range(10))]

    result = multiprocess(foobar, inputs, num_workers=5, cuts_per_worker=1, batching=False)
    assert result[:3] == [0, 10, 20]
    result = multiprocess(foobar, inputs, num_workers=5, cuts_per_worker=1, batching=True)
    assert result[:3] == [0, 10, 20]
    result = multiprocess(starbar, star_inputs, num_workers=5, cuts_per_worker=1, batching=False)
    assert result[:3] == ['x=0 and y=2 so x*y=0', 'x=1 and y=3 so x*y=3', 'x=2 and y=4 so x*y=8']
    try:
        print(multiprocess(starbar, star_inputs, num_workers=5, cuts_per_worker=1, batching=True))
    except NotImplementedError as e:
        print(f"... not implemented yet")
