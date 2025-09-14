"""TSP Data management classes with validation and coordinate system handling.

This module provides classes for managing TSP data with proper validation
and support for different coordinate systems (geographic and Cartesian).
"""

# TODO: Should switch to python 3.12 and use @override decorator
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import cast

import networkx as nx
import pandas as pd

from .node_schema import CartesianNodeModel, GeographicNodeModel


class CoordinateSystem(StrEnum):
    """Represents coordinate system types."""

    GEOGRAPHIC = "geographic"
    CARTESIAN = "cartesian"


class TspData(ABC):
    """Abstract base class for TSP data management."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with validated DataFrame.

        Args:
            df: DataFrame to validate and store

        Raises:
            ValueError: If validation fails
        """
        self._df = self._validate(df)
        self._coordinate_system = self._get_coordinate_system()

    @abstractmethod
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the DataFrame against the appropriate schema.

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails
        """

    @abstractmethod
    def _get_coordinate_system(self) -> CoordinateSystem:
        """Get the coordinate system type for this data."""

    @classmethod
    def from_csv(cls, filepath: Path) -> TspData:
        """Factory method to create TspData from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Appropriate TspData subclass instance

        Raises:
            ValueError: If CSV doesn't contain valid coordinate data
            FileNotFoundError: If file doesn't exist
        """
        df = pd.read_csv(filepath, skipinitialspace=True)

        # Auto-detect coordinate system and return appropriate subclass
        if cls._has_geographic_coords(df):
            return GeographicTspData(df)
        elif cls._has_cartesian_coords(df):
            return CartesianTspData(df)
        else:
            raise ValueError(
                "DataFrame must have either lat/lon or x/y coordinates. "
                f"Found columns: {list(df.columns)}"
            )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> TspData:
        """Factory method to create TspData from DataFrame.

        Args:
            df: DataFrame with coordinate data

        Returns:
            Appropriate TspData subclass instance

        Raises:
            ValueError: If DataFrame doesn't contain valid coordinate data
        """
        # Auto-detect coordinate system and return appropriate subclass
        if cls._has_geographic_coords(df):
            return GeographicTspData(df)
        elif cls._has_cartesian_coords(df):
            return CartesianTspData(df)
        else:
            raise ValueError(
                "DataFrame must have either lat/lon or x/y coordinates. "
                f"Found columns: {list(df.columns)}"
            )

    @classmethod
    def from_tsplib(cls, filepath: Path) -> TspData:
        """Factory method to create TspData from a TSPLIB file.

        Args:
            filepath: Path to the TSPLIB file

        Returns:
            TspData instance with parsed node data

        Raises:
            ValueError: If file format is invalid or unsupported
            FileNotFoundError: If the specified file doesn't exist
        """
        if not filepath.exists():
            raise FileNotFoundError(f"TSPLIB file not found: {filepath}")

        # Parse the TSPLIB file
        nodes_data = []
        depot_nodes = []
        in_coord_section = False
        in_depot_section = False

        with open(filepath, encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check for section starts
                if line == "NODE_COORD_SECTION":
                    in_coord_section = True
                    in_depot_section = False
                    continue
                elif line == "DEPOT_SECTION":
                    in_coord_section = False
                    in_depot_section = True
                    continue

                # Check for end of file
                if line == "EOF":
                    break

                # Parse coordinate data
                if in_coord_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = parts[0]
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])

                            # Create node data - assume all nodes are permanent initially
                            nodes_data.append(
                                {
                                    "name": node_id,
                                    "node_type": "permanent",
                                    "x": x_coord,
                                    "y": y_coord,
                                }
                            )
                        except (ValueError, IndexError) as e:
                            raise ValueError(
                                f"Invalid coordinate data in line: {line}"
                            ) from e

                # Parse depot data
                elif in_depot_section:
                    # Depot section contains depot node IDs terminated by -1
                    if line == "-1":
                        break
                    try:
                        depot_id = line.strip()
                        depot_nodes.append(depot_id)
                    except ValueError as e:
                        raise ValueError(f"Invalid depot data in line: {line}") from e

        if not nodes_data:
            raise ValueError("No coordinate data found in TSPLIB file")

        # Update the first depot node to have startend type
        if depot_nodes:
            first_depot = depot_nodes[0]
            for node in nodes_data:
                if node["name"] == first_depot:
                    node["node_type"] = "startend"
                    break

        # Create DataFrame and return CartesianTspData
        df = pd.DataFrame(nodes_data)
        return CartesianTspData(df)

    @staticmethod
    def _has_geographic_coords(df: pd.DataFrame) -> bool:
        """Check if DataFrame has geographic coordinates."""
        return "lat" in df.columns and "lon" in df.columns

    @staticmethod
    def _has_cartesian_coords(df: pd.DataFrame) -> bool:
        """Check if DataFrame has Cartesian coordinates."""
        return "x" in df.columns and "y" in df.columns

    @property
    def df(self) -> pd.DataFrame:
        """Get a copy of the validated DataFrame."""
        return self._df.copy()

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Get the coordinate system type."""
        return self._coordinate_system

    def get_start_nodes(self) -> pd.DataFrame:
        """Get nodes marked as start or startend.

        Returns:
            DataFrame containing only start/startend nodes
        """
        return self._df[self._df["node_type"].isin(["start", "startend"])].copy()

    def get_permanent_nodes(self) -> pd.DataFrame:
        """Get nodes marked as permanent.

        Returns:
            DataFrame containing only permanent nodes
        """
        return self._df[self._df["node_type"] == "permanent"].copy()

    def get_node_names(self) -> list[str]:
        """Get list of all node names.

        Returns:
            List of node names
        """
        return self._df["name"].tolist()

    def get_start_node_name(self) -> str:
        """Get the name of the startend or start node.

        If only permanent nodes, returns the name of the first permanent node.

        Returns:
            The name og start or startend node.
        """
        start_nodes = self.get_start_nodes()
        return (
            start_nodes["name"].iloc[0]
            if len(start_nodes) >= 1
            else self.get_node_names()[0]
        )

    @abstractmethod
    def to_graph(self, reverse_positions: bool = False) -> nx.Graph:
        """Convert the TspData to a networkx Graph.

        Args:
            reverse_positions: If True, reverse the x/y or lat/lon coordinates in the graph
        """

    def __len__(self) -> int:
        """Get number of nodes."""
        return len(self._df)

    def __repr__(self) -> str:
        """String representation of TspData."""
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._df)}, "
            f"coordinate_system={self._coordinate_system})"
        )


class GeographicTspData(TspData):
    """TSP data with geographic coordinates (lat/lon)."""

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against geographic schema."""
        return cast(pd.DataFrame, GeographicNodeModel.validate(df))

    def _get_coordinate_system(self) -> CoordinateSystem:
        """Get coordinate system type."""
        return CoordinateSystem.GEOGRAPHIC

    def to_graph(self, reverse_positions: bool = False) -> nx.Graph:
        """Convert the GeographicTspData to a networkx Graph.

        Args:
            reverse_positions: If True, reverse the x/y or lat/lon coordinates in the graph

        Returns:
            nx.Graph: A networkx graph with nodes containing position and metadata
        """
        graph: nx.Graph = nx.Graph()
        for _, row in self._df.iterrows():
            pos = (
                (row["lon"], row["lat"])
                if reverse_positions
                else (row["lat"], row["lon"])
            )
            graph.add_node(
                row["name"], pos=pos, name=row["name"], node_type=row["node_type"]
            )
        return graph


class CartesianTspData(TspData):
    """TSP data with Cartesian coordinates (x/y)."""

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against Cartesian schema."""
        return cast(pd.DataFrame, CartesianNodeModel.validate(df))

    def _get_coordinate_system(self) -> CoordinateSystem:
        """Get coordinate system type."""
        return CoordinateSystem.CARTESIAN

    def to_graph(self, reverse_positions: bool = False) -> nx.Graph:
        """Convert the CartesianTspData to a networkx Graph.

        Args:
            reverse_positions: If True, reverse the x/y or lat/lon coordinates in the graph

        Returns:
            nx.Graph: A networkx graph with nodes containing position and metadata
        """
        graph: nx.Graph = nx.Graph()
        for _, row in self._df.iterrows():
            pos = (row["y"], row["x"]) if reverse_positions else (row["x"], row["y"])
            graph.add_node(
                row["name"], pos=pos, name=row["name"], node_type=row["node_type"]
            )
        return graph
