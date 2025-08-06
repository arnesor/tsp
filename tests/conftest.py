import pandas as pd
import pytest


@pytest.fixture
def df_xy1() -> pd.DataFrame:
    return pd.DataFrame({"name": ["A", "B", "C"], "x": [0, 3, 0], "y": [0, 0, 4]})


@pytest.fixture
def df_xy1_facit_euclidian() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
