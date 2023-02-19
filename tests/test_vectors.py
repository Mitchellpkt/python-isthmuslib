import pytest
from src.isthmuslib.vectors import Timeseries
import pandas as pd


def test_resample():
    v = Timeseries(data=pd.DataFrame({"timestamp": range(15), "b": [x * 3 for x in range(15)]}))
    v.resample(interval=3, inplace=True)
    assert v.data["timestamp"].tolist() == [0, 3, 6, 9, 12]
    assert v.data["b"].tolist() == [0, 9, 18, 27, 36]
