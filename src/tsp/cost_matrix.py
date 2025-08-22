"""Cost matrix calculation module for TSP optimization.

This module provides classes and functions to calculate cost matrices for route optimization
problems. It supports different coordinate systems (lat/lon and x/y) and various distance
calculation methods including Euclidean, geodesic, and API-based routing.
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pyproj import Transformer
from scipy.spatial.distance import pdist, squareform

from .tsp_data import CoordinateSystem


class CostCalculator(ABC):
    """Abstract base class for cost matrix calculators."""

    @abstractmethod
    def calculate_cost_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the cost matrix for the given DataFrame.

        Args:
            df: DataFrame with coordinate columns and node identifiers

        Returns:
            Square DataFrame with costs between all pairs of nodes
        """
        pass

    @abstractmethod
    def supports_coordinates(self, df: pd.DataFrame) -> bool:
        """Check if this calculator supports the coordinate system in the DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            True if this calculator can handle the coordinate system
        """
        pass

    @staticmethod
    def has_latlon(df: pd.DataFrame) -> bool:
        """Check if DataFrame has lat/lon columns."""
        return "lat" in df.columns and "lon" in df.columns

    @staticmethod
    def has_xy(df: pd.DataFrame) -> bool:
        """Check if DataFrame has x/y columns."""
        return "x" in df.columns and "y" in df.columns


class EuclideanCalculator(CostCalculator):
    """Calculator for Euclidean distance using x/y coordinates."""

    def supports_coordinates(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has x and y columns."""
        return self.has_latlon(df) or self.has_xy(df)

    @staticmethod
    def _add_utm_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and add UTM coordinates to the DataFrame."""
        avg_lon = df["lon"].mean()
        utm_zone = int((avg_lon + 180) / 6) + 1

        avg_lat = df["lat"].mean()
        hemisphere = "north" if avg_lat >= 0 else "south"

        # Create transformer from WGS84 to UTM
        utm_crs = (
            f"EPSG:326{utm_zone:02d}"
            if hemisphere == "north"
            else f"EPSG:327{utm_zone:02d}"
        )
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Transform coordinates to UTM (vectorized operation)
        x_coords, y_coords = transformer.transform(df["lon"].values, df["lat"].values)

        df_with_utm = df.copy()
        df_with_utm["x"] = x_coords.astype(int)
        df_with_utm["y"] = y_coords.astype(int)

        return df_with_utm

    def calculate_cost_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Euclidean distance matrix for x/y coordinates."""
        if not self.supports_coordinates(df):
            raise ValueError(
                "DataFrame must have 'x' and 'y' columns or 'lat' and 'lon' columns"
            )

        if not self.has_xy(df):
            df = self._add_utm_coordinates(df)

        coords = df[["x", "y"]].values

        # Calculate pairwise Euclidean distances (vectorized)
        distances = squareform(pdist(coords, metric="euclidean"))

        # Create DataFrame with node names as index/columns
        node_names = df["name"].tolist() if "name" in df.columns else df.index.tolist()
        return pd.DataFrame(distances, index=node_names, columns=node_names)


class GeodesicCalculator(CostCalculator):
    """Calculator for geodesic distance using lat/lon coordinates."""

    def supports_coordinates(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has lat and lon columns."""
        return self.has_latlon(df)

    def calculate_cost_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate geodesic distance matrix for lat/lon coordinates."""
        if not self.supports_coordinates(df):
            raise ValueError("DataFrame must have 'lat' and 'lon' columns")

        # Extract coordinates
        coords = df[["lat", "lon"]].values

        n = len(coords)
        # Calculate distance matrix using geopy geodesic distance
        distances = np.zeros((n, n))
        for i, j in itertools.product(range(n), range(n)):
            if i != j:
                point1 = (coords[i][0], coords[i][1])  # (lat, lon)
                point2 = (coords[j][0], coords[j][1])  # (lat, lon)
                distances[i][j] = geodesic(point1, point2).meters

        # Create DataFrame with node names as index/columns
        node_names = df["name"].tolist() if "name" in df.columns else df.index.tolist()
        return pd.DataFrame(distances, index=node_names, columns=node_names)


class OpenRouteServiceCalculator(CostCalculator):
    """Calculator using OpenRouteService API for walk distance/duration."""

    def __init__(
        self, api_key: str, profile: str = "foot-walking", metric: str = "distance"
    ) -> None:
        """Initialize OpenRouteService calculator.

        Args:
            api_key: OpenRouteService API key
            profile: Routing profile (foot-walking, driving-car, etc.)
            metric: What to calculate - 'distance' or 'duration'
        """
        self.api_key = api_key
        self.profile = profile
        self.metric = metric
        self.base_url = "https://api.openrouteservice.org/v2/matrix"

    def supports_coordinates(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has lat and lon columns."""
        return "lat" in df.columns and "lon" in df.columns

    def calculate_cost_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cost matrix using OpenRouteService API."""
        if not self.supports_coordinates(df):
            raise ValueError("DataFrame must have 'lat' and 'lon' columns")

        # This would require the requests library
        # Implementation would make API calls to OpenRouteService
        # For now, return a placeholder
        raise NotImplementedError(
            "OpenRouteService calculator requires 'requests' library and API implementation"
        )


class CostMatrixFactory:
    """Factory class for creating appropriate cost calculators."""

    @staticmethod
    def create_calculator(
        df: pd.DataFrame, method: str | None = None
    ) -> CostCalculator:
        """Create an appropriate cost calculator based on data and method.

        Args:
            df: DataFrame with coordinate data
            method: Specific method to use ('euclidean', 'geodesic', 'openroute')
                   If None, uses default based on coordinate system

        Returns:
            Appropriate CostCalculator instance
        """
        has_latlon = CostCalculator.has_latlon(df)
        has_xy = CostCalculator.has_xy(df)

        if has_latlon:
            coordinate_system = CoordinateSystem.GEOGRAPHIC
        elif has_xy:
            coordinate_system = CoordinateSystem.CARTESIAN
        else:
            raise ValueError(
                "DataFrame must have either 'x'/'y' or 'lat'/'lon' columns"
            )

        # Select appropriate calculator based on method and coordinate system
        if method is None:
            # Use default based on coordinate system
            if coordinate_system == CoordinateSystem.GEOGRAPHIC:
                return GeodesicCalculator()
            else:
                return EuclideanCalculator()

        if method == "euclidean":
            return EuclideanCalculator()
        elif method == "geodesic":
            if coordinate_system != CoordinateSystem.GEOGRAPHIC:
                raise ValueError(
                    "geodesic method requires geographic coordinates (lat/lon)"
                )
            return GeodesicCalculator()
        elif method == "openroute":
            if coordinate_system != CoordinateSystem.GEOGRAPHIC:
                raise ValueError(
                    "openroute method requires geographic coordinates (lat/lon)"
                )
            raise NotImplementedError(
                "OpenRouteService calculator not fully implemented"
            )
        else:
            raise ValueError(f"Unknown method: {method}")


def calculate_cost_matrix(  # type: ignore[no-untyped-def]
    df: pd.DataFrame,
    method: str | None = None,
    **kwargs,  # noqa: ANN003
) -> pd.DataFrame:
    """Convenience function to calculate cost matrix.

    Args:
        df: DataFrame with coordinate data
        method: Cost calculation method
        **kwargs: Additional arguments for specific calculators

    Returns:
        Cost matrix as DataFrame
    """
    calculator = CostMatrixFactory.create_calculator(df, method)
    return calculator.calculate_cost_matrix(df)
