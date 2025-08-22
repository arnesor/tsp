"""TSP Data management classes with validation and coordinate system handling.

This module provides classes for managing TSP data with proper validation
and support for different coordinate systems (geographic and Cartesian).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import cast

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

    @staticmethod
    def _has_geographic_coords(df: pd.DataFrame) -> bool:
        """Check if DataFrame has geographic coordinates."""
        return "lat" in df.columns and "lon" in df.columns

    @staticmethod
    def _has_cartesian_coords(df: pd.DataFrame) -> bool:
        """Check if DataFrame has Cartesian coordinates."""
        return "x" in df.columns and "y" in df.columns

    @property
    def data(self) -> pd.DataFrame:
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


class CartesianTspData(TspData):
    """TSP data with Cartesian coordinates (x/y)."""

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against Cartesian schema."""
        return cast(pd.DataFrame, CartesianNodeModel.validate(df))

    def _get_coordinate_system(self) -> CoordinateSystem:
        """Get coordinate system type."""
        return CoordinateSystem.CARTESIAN
