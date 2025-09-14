from pathlib import Path

import networkx as nx
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
    def test_from_dataframe_geographic_happy_path(self) -> None:
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

    def test_from_dataframe_cartesian_happy_path(self) -> None:
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

    def test_from_dataframe_prefers_geographic_when_both_present(self) -> None:
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

    def test_from_dataframe_missing_coordinates_raises_value_error(self) -> None:
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

    def test_from_csv_cartesian_happy_path(self, tmp_path: Path) -> None:
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
    def test_get_start_node_name_startend_present(self) -> None:
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

    def test_get_start_node_name_start_and_end_present(self) -> None:
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

    def test_get_start_node_name_only_permanent_returns_first(self) -> None:
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

    def test_get_start_and_permanent_nodes_filtering(self) -> None:
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

    def test_len_and_repr(self) -> None:
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


class TestTspDataToGraph:
    def test_to_graph_basic(self) -> None:
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "node_type": ["startend", "permanent", "permanent", "permanent"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            }
        )
        data = CartesianTspData(df)

        # Act
        graph = data.to_graph()

        # Assert
        assert isinstance(graph, nx.Graph), f"Expected nx.Graph, got {type(graph)}"
        assert len(graph.nodes) == 4, f"Expected 4 nodes, got {len(graph.nodes)}"
        assert len(graph.edges) == 0, f"Expected 0 edges, got {len(graph.edges)}"

    def test_to_graph_node_attributes(self) -> None:
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["NodeA", "NodeB"],
                "node_type": ["startend", "permanent"],
                "x": [10.5, 20.7],
                "y": [30.2, 40.8],
            }
        )
        data = CartesianTspData(df)

        # Act
        graph = data.to_graph()

        # Assert
        node_a = graph.nodes["NodeA"]
        assert node_a["pos"] == (
            10.5,
            30.2,
        ), f"Expected pos=(10.5, 30.2), got {node_a['pos']}"
        assert node_a["name"] == "NodeA", f"Expected name='NodeA', got {node_a['name']}"
        assert (
            node_a["node_type"] == "startend"
        ), f"Expected node_type='startend', got {node_a['node_type']}"

        node_b = graph.nodes["NodeB"]
        assert node_b["pos"] == (
            20.7,
            40.8,
        ), f"Expected pos=(20.7, 40.8), got {node_b['pos']}"
        assert node_b["name"] == "NodeB", f"Expected name='NodeB', got {node_b['name']}"
        assert (
            node_b["node_type"] == "permanent"
        ), f"Expected node_type='permanent', got {node_b['node_type']}"

    def test_to_graph_empty_data(self) -> None:
        # Arrange
        df = pd.DataFrame({"name": [], "node_type": [], "x": [], "y": []})
        df = df.astype(
            {"name": "object", "node_type": "object", "x": "float64", "y": "float64"}
        )
        data = CartesianTspData(df)

        # Act
        graph = data.to_graph()

        # Assert
        assert len(graph.nodes) == 0, f"Expected 0 nodes, got {len(graph.nodes)}"
        assert len(graph.edges) == 0, f"Expected 0 edges, got {len(graph.edges)}"

    def test_to_graph_different_node_types(self) -> None:
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["Start", "End", "Perm1", "Perm2"],
                "node_type": ["start", "end", "permanent", "permanent"],
                "x": [0.0, 1.0, 3.0, 4.0],
                "y": [0.0, 1.0, 3.0, 4.0],
            }
        )
        data = CartesianTspData(df)

        # Act
        graph = data.to_graph()

        # Assert
        expected_nodes = {"Start", "End", "Perm1", "Perm2"}
        actual_nodes = set(graph.nodes())
        assert (
            actual_nodes == expected_nodes
        ), f"Expected {expected_nodes}, got {actual_nodes}"

        assert graph.nodes["Start"]["node_type"] == "start"
        assert graph.nodes["End"]["node_type"] == "end"
        assert graph.nodes["Perm1"]["node_type"] == "permanent"
        assert graph.nodes["Perm2"]["node_type"] == "permanent"
