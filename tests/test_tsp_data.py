import pandas as pd
import pytest

from tsp.node_schema import NodeType
from tsp.tsp_data import (
    CartesianTspData,
    CoordinateSystem,
    GeographicTspData,
    TspData,
)


class TestTspDataFactories:
    def test_from_dataframe_geographic_happy_path(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "node_type": [
                    NodeType.STARTEND.value,
                    NodeType.PERMANENT.value,
                    NodeType.PERMANENT.value,
                ],
                "lat": [10.0, 20.0, 30.0],
                "lon": [100.0, 110.0, 120.0],
            }
        )

        # Act
        data = TspData.from_dataframe(df)

        # Assert
        assert isinstance(data, GeographicTspData)
        assert data.coordinate_system == CoordinateSystem.GEOGRAPHIC
        assert list(data.df.columns) == ["name", "node_type", "lat", "lon"]

    def test_from_dataframe_cartesian_happy_path(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "node_type": [
                    NodeType.START.value,
                    NodeType.END.value,
                    NodeType.PERMANENT.value,
                ],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
            }
        )

        # Act
        data = TspData.from_dataframe(df)

        # Assert
        assert isinstance(data, CartesianTspData)
        assert data.coordinate_system == CoordinateSystem.CARTESIAN
        assert list(data.df.columns) == ["name", "node_type", "x", "y"]

    def test_from_dataframe_prefers_geographic_when_both_present(self):
        # Arrange: Contains both lat/lon and x/y -> should prefer Geographic
        df = pd.DataFrame(
            {
                "name": ["A", "B"],
                "node_type": [NodeType.STARTEND.value, NodeType.PERMANENT.value],
                "lat": [10.0, 20.0],
                "lon": [100.0, 110.0],
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            }
        )

        # Act
        data = TspData.from_dataframe(df)

        # Assert
        assert isinstance(data, GeographicTspData)
        assert data.coordinate_system == CoordinateSystem.GEOGRAPHIC

    def test_from_dataframe_missing_coordinates_raises_value_error(self):
        # Arrange: Missing both lat/lon and x/y
        df = pd.DataFrame(
            {
                "name": ["A", "B"],
                "node_type": [NodeType.PERMANENT.value, NodeType.PERMANENT.value],
            }
        )

        # Act & Assert
        with pytest.raises(ValueError, match="either lat/lon or x/y coordinates"):
            TspData.from_dataframe(df)

    def test_from_csv_cartesian_happy_path(self, tmp_path: pytest.TempPathFactory):
        # Arrange: Write a simple cartesian CSV
        df = pd.DataFrame(
            {
                "name": ["N1", "N2", "N3"],
                "node_type": [
                    NodeType.START.value,
                    NodeType.END.value,
                    NodeType.PERMANENT.value,
                ],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
            }
        )
        path = tmp_path / "cartesian.csv"
        df.to_csv(path, index=False)

        # Act
        data = TspData.from_csv(path)

        # Assert
        assert isinstance(data, CartesianTspData)
        assert data.coordinate_system == CoordinateSystem.CARTESIAN


class TestTspDataBehaviors:
    def test_get_start_node_name_startend_present(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["Depot", "C1", "C2"],
                "node_type": [
                    NodeType.STARTEND.value,
                    NodeType.PERMANENT.value,
                    NodeType.PERMANENT.value,
                ],
                "lat": [0.0, 1.0, 2.0],
                "lon": [0.0, 1.0, 2.0],
            }
        )
        data = TspData.from_dataframe(df)

        # Act
        start_name = data.get_start_node_name()

        # Assert
        assert start_name == "Depot"

    def test_get_start_node_name_start_and_end_present(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["S", "E", "C1"],
                "node_type": [
                    NodeType.START.value,
                    NodeType.END.value,
                    NodeType.PERMANENT.value,
                ],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
            }
        )
        data = TspData.from_dataframe(df)

        # Act
        start_name = data.get_start_node_name()

        # Assert
        assert start_name == "S"

    def test_get_start_node_name_only_permanent_returns_first(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "node_type": [NodeType.PERMANENT.value] * 3,
                "lat": [10.0, 20.0, 30.0],
                "lon": [100.0, 110.0, 120.0],
            }
        )
        data = TspData.from_dataframe(df)

        # Act
        start_name = data.get_start_node_name()

        # Assert
        assert start_name == "A"

    def test_get_start_and_permanent_nodes_filtering(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["S", "E", "C1", "C2"],
                "node_type": [
                    NodeType.START.value,
                    NodeType.END.value,
                    NodeType.PERMANENT.value,
                    NodeType.PERMANENT.value,
                ],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            }
        )
        data = TspData.from_dataframe(df)

        # Act
        start_nodes = data.get_start_nodes()
        permanent_nodes = data.get_permanent_nodes()

        # Assert
        assert list(start_nodes["name"]) == ["S"]
        assert list(permanent_nodes["name"]) == ["C1", "C2"]

    def test_len_and_repr(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["A", "B"],
                "node_type": [NodeType.STARTEND.value, NodeType.PERMANENT.value],
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            }
        )
        data = TspData.from_dataframe(df)

        # Act
        n = len(data)
        r = repr(data)

        # Assert
        assert n == 2
        assert "GeographicTspData(" in r
        assert "coordinate_system=geographic" in r
