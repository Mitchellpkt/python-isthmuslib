from typing import Tuple, Any, List, Dict

import matplotlib.pyplot as plt

from isthmuslib import VectorMultiset
from src.isthmuslib.logging import parse_string_with_embedded_json, parse_string_with_manual_tokens
from src.isthmuslib.utils import risky_cast, process_queue, recursive_batch_evaluation, return_best_input


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


def test_process_queue():
    inputs: List[int] = list(range(25))
    star_inputs: List[Tuple[int, int]] = [(x, x + 2) for x in list(range(10))]

    result = process_queue(foobar, inputs, num_workers=5, cuts_per_worker=1, batching=False)
    assert result[:3] == [0, 10, 20]
    result = process_queue(foobar, inputs, num_workers=5, cuts_per_worker=1, batching=True)
    assert result[:3] == [0, 10, 20]
    result = process_queue(starbar, star_inputs, num_workers=5, cuts_per_worker=1, batching=False)
    assert result[:3] == ['x=0 and y=2 so x*y=0', 'x=1 and y=3 so x*y=3', 'x=2 and y=4 so x*y=8']
    try:
        print(process_queue(starbar, star_inputs, num_workers=5, cuts_per_worker=1, batching=True))
    except NotImplementedError as e:
        print(f"... not implemented yet")


def test_parsing():
    log: str = """
    log-2018-08-31-20-07-29:2018-08-31 20:07:24.748	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49156/1651224 (0.282567 sec, 353.898367 blocks/sec), 99.548294 MB [49255:o][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:25.040	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49256/1651224 (0.291497 sec, 343.056704 blocks/sec), 99.046715 MB [49355:][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:25.280	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49356/1651224 (0.239763 sec, 417.078532 blocks/sec), 98.918274 MB [49455:oo][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:25.644	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49456/1651224 (0.363321 sec, 275.238701 blocks/sec), 98.767830 MB [49555:o.][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:25.981	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49556/1651224 (0.335829 sec, 297.770592 blocks/sec), 98.502403 MB [49655:...][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:26.264	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49656/1651224 (0.281597 sec, 355.117420 blocks/sec), 98.384758 MB [49755:oo....][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:26.776	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49756/1651224 (0.404512 sec, 247.211455 blocks/sec), 98.054283 MB [49855:oo.....][0m
    log-2018-08-31-20-07-29:2018-08-31 20:07:27.066	[P2P2]	INFO protocol_handler.inl:1171	[1;33m[80.241.216.213:18080 OUT]  Synced 49856/1651224 (0.289512 sec, 345.408826 blocks/sec), 98.258347 MB [49955:oo.....][0m
    """

    # Set the rules for extracting the vectors
    record_delimiter: str = 'lo'
    tokens_dictionary: Dict[str, Tuple[str, str]] = {
        # varname: (left_token, right_token)
        'date_time_stamp': ('g-', ':'),
        'height': ('Synced ', '/'),
        'time_to_load': ('(', ' sec'),
    }

    # Extract the text into a timeseries
    timeseries: VectorMultiset = VectorMultiset(
        data=parse_string_with_manual_tokens(log,
                                             record_delimiter=record_delimiter,
                                             parallelize_processing=True,
                                             tokens_dictionary=tokens_dictionary))
    print(timeseries.data)


def mock_hill(kwargs) -> float:
    x = kwargs.get('x')
    y = kwargs.get('y')
    return 1 / x + 1 / (y ** 2)


def test_recursive_batch_evaluation():
    inital_point: Dict[str, Any] = {'x': 400, 'y': 300}
    current_best_input, history = recursive_batch_evaluation(
        func=mock_hill,
        initial_input=inital_point,
        selection_method=max,
        batch_generator_kwargs={'width_prct': 10, 'num_samples': 3},
        max_deep=5,
        print_progress=True,
        num_workers=64,
        return_history=True,
    )
    assert current_best_input['x'] < inital_point['x']
    assert current_best_input['y'] < inital_point['y']


def test_selector():
    inputs: List[Any] = ['abc', 'def', 'efg']
    eval_outputs: List[Dict[str, Any]] = [{'foobar': 'xyz', 'fitness': 4}, {'fitness': 99, 'baz': 12345},
                                          {'fitness': -5}]
    best = return_best_input(inputs, eval_outputs)
    assert best == 'def'


def y(x: float) -> float:
    return 10 - (x - 5) ** 2


def y_return_scalar(d) -> float:
    return y(d.get('x'))


def test_recursive_batch_evaluation_new_with_scalar(show_plot: bool = True):
    # Set up the run
    start_at: float = 15

    best_x, run_data = recursive_batch_evaluation(
        func=y_return_scalar,
        initial_input={'x': start_at},
        max_deep=5,
        return_history=True,
        evaluate_initial_inputs=True,
        infinite_memory=True,
        selection_method=max,
        catch_exceptions=False,
        batch_generator_kwargs={
            "width_prct": 50,
            "num_samples": 5,
            "fields": ['x'],
            "width_temperature_prct": 0,
        },
    )

    print(f"{best_x=}\n{run_data}")
    if show_plot:
        x_vals: List[float] = list(range(-10, 25))
        plt.figure(facecolor='w')
        plt.plot(x_vals, [y_return_scalar({'x': x}) for x in x_vals], color='k')
        x_vec = [d.get('in').get('x') for d in run_data]
        y_vec = [x.get('out') for x in run_data]
        plt.scatter(x_vec, y_vec, c=list(range(len(x_vec))), s=50)
        plt.show()

    assert best_x['x'] < start_at


def y_return_dict(d) -> dict:
    return {'y': y(d.get('x'))}


def test_recursive_batch_evaluation_new_with_dict(show_plot: bool = True):
    # Set up the run
    start_at: float = 15

    best_x, run_data = recursive_batch_evaluation(
        func=y_return_dict,
        initial_input={'x': start_at},
        max_deep=5,
        return_history=True,
        evaluate_initial_inputs=True,
        infinite_memory=True,
        selection_method='y',
        catch_exceptions=False,
        batch_generator_kwargs={
            "width_prct": 50,
            "num_samples": 5,
            "fields": ['x'],
            "width_temperature_prct": 0,
        },
    )

    print(f"{best_x=}\n{run_data}")
    if show_plot:
        x_vals: List[float] = list(range(-10, 25))
        plt.figure(facecolor='w')
        plt.plot(x_vals, [y_return_dict({'x': x})['y'] for x in x_vals], color='k')
        x_vec = [d.get('in').get('x') for d in run_data]
        y_vec = [x.get('out').get('y') for x in run_data]
        plt.scatter(x_vec, y_vec, c=list(range(len(x_vec))), s=50)
        plt.show()

    print(f"{run_data[-1]['in']}")
    assert run_data[-1]['in']['incremented_counter'] == 4
    assert run_data[-1]['in']['clock_time_at_start'] > 100000
    assert best_x['x'] < start_at
