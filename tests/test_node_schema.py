import pandas as pd
import pytest
from pandera.errors import SchemaError

from tsp.node_schema import CartesianNodeModel, GeographicNodeModel, NodeType


# 1. All permanent nodes (valid)
def test_geographic_schema_accepts_all_permanent() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "node_type": [NodeType.PERMANENT.value] * 3,
            "lat": [10.0, 20.0, 30.0],
            "lon": [100.0, 110.0, 120.0],
        }
    )
    GeographicNodeModel.validate(df)


# 2. One startend, rest permanent (valid)
def test_geographic_schema_accepts_one_startend_rest_permanent() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "node_type": [NodeType.STARTEND.value] + [NodeType.PERMANENT.value] * 2,
            "lat": [1.0, 2.0, 3.0],
            "lon": [100.0, 101.0, 102.0],
        }
    )
    GeographicNodeModel.validate(df)


# 3. Two starts (invalid)
def test_geographic_schema_rejects_two_starts() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "node_type": [
                NodeType.START.value,
                NodeType.START.value,
                NodeType.PERMANENT.value,
            ],
            "lat": [1.0, 2.0, 3.0],
            "lon": [10.0, 11.0, 12.0],
        }
    )
    with pytest.raises(SchemaError):
        GeographicNodeModel.validate(df)


# 4. One start, no end (invalid)
def test_geographic_schema_rejects_one_start_no_end() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "node_type": [
                NodeType.START.value,
                NodeType.PERMANENT.value,
                NodeType.PERMANENT.value,
            ],
            "lat": [1.0, 2.0, 3.0],
            "lon": [10.0, 11.0, 12.0],
        }
    )
    with pytest.raises(SchemaError):
        GeographicNodeModel.validate(df)


# 5. One start, one end, rest permanent (valid)
def test_geographic_schema_accepts_start_end_permanent() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "node_type": [NodeType.START.value, NodeType.END.value]
            + [NodeType.PERMANENT.value] * 2,
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lon": [10.0, 11.0, 12.0, 13.0],
        }
    )
    GeographicNodeModel.validate(df)


# 6. Unknown node type (invalid value)
def test_geographic_schema_rejects_unknown_node_type() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "node_type": ["permanent", "mystery", "start"],
            "lat": [1, 2, 3],
            "lon": [10, 11, 12],
        }
    )
    with pytest.raises(SchemaError):
        GeographicNodeModel.validate(df)


# 7. Duplicate names (unique constraint fail)
def test_geographic_schema_enforces_unique_names() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "A", "B"],
            "node_type": [NodeType.PERMANENT.value] * 3,
            "lat": [1.0, 2.0, 3.0],
            "lon": [10.0, 11.0, 12.0],
        }
    )
    with pytest.raises(SchemaError):
        GeographicNodeModel.validate(df)


# 8. Out of range lat/lon
@pytest.mark.parametrize(
    "lat,lon",
    [
        (100.0, 50.0),  # invalid lat
        (-91.0, 100.0),  # invalid lat
        (0.0, 181.0),  # invalid lon
        (45.0, -181.0),  # invalid lon
    ],
)
def test_geographic_schema_rejects_out_of_range_latlon(lat: float, lon: float) -> None:
    df = pd.DataFrame(
        {
            "name": ["A"],
            "node_type": [NodeType.PERMANENT.value],
            "lat": [lat],
            "lon": [lon],
        }
    )
    with pytest.raises(SchemaError):
        GeographicNodeModel.validate(df)


# 9. Minimal valid DataFrame (one node)
def test_geographic_schema_succeeds_with_one_node() -> None:
    df = pd.DataFrame(
        {
            "name": ["Node1"],
            "node_type": [NodeType.PERMANENT.value],
            "lat": [0.0],
            "lon": [0.0],
        }
    )
    GeographicNodeModel.validate(df)


# 10. Cartesian schema equivalents
@pytest.mark.parametrize(
    "names,node_types,xs,ys,should_pass",
    [
        (["A", "B", "C"], [NodeType.PERMANENT.value] * 3, [0, 1, 2], [0, 1, 2], True),
        (
            ["A", "B", "C"],
            [
                NodeType.STARTEND.value,
                NodeType.PERMANENT.value,
                NodeType.PERMANENT.value,
            ],
            [0, 1, 2],
            [1, 2, 3],
            True,
        ),
        (
            ["A", "B", "C"],
            [NodeType.START.value, NodeType.START.value, NodeType.PERMANENT.value],
            [0, 1, 2],
            [1, 2, 3],
            False,
        ),
    ],
)
def test_cartesian_schema_configurations(
    names: list[str],
    node_types: list[str],
    xs: list[int],
    ys: list[int],
    should_pass: bool,
) -> None:
    df = pd.DataFrame(
        {
            "name": names,
            "node_type": node_types,
            "x": [float(val) for val in xs],
            "y": [float(val) for val in ys],
        }
    )
    if should_pass:
        CartesianNodeModel.validate(df)
    else:
        with pytest.raises(SchemaError):
            CartesianNodeModel.validate(df)
