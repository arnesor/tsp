import itertools

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tsp.cost_matrix import (
    CostMatrixFactory,
    EuclideanCalculator,
    GeodesicCalculator,
    OpenRouteServiceCalculator,
    calculate_cost_matrix,
)


class TestEuclideanCalculator:
    def test_euclidean_calculator_xy_happy_path(
        self, df_xy1: pd.DataFrame, df_xy1_facit_euclidian: pd.DataFrame
    ) -> None:
        # Act
        calculator = EuclideanCalculator()
        cost_matrix = calculator.calculate_cost_matrix(df_xy1)

        # Assert: Matrix should be symmetric and distances correct
        pdt.assert_frame_equal(cost_matrix, df_xy1_facit_euclidian, check_exact=False)
        pdt.assert_frame_equal(cost_matrix, df_xy1_facit_euclidian.T, check_exact=False)

    def test_euclidean_calculator_missing_coordinates_edge_case(self) -> None:
        # Arrange: DataFrame missing both x/y and lat/lon
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        calculator = EuclideanCalculator()

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="DataFrame must have 'x' and 'y' columns or 'lat' and 'lon' columns",
        ):
            calculator.calculate_cost_matrix(df)

    def test_euclidean_calculator_no_name_column(self) -> None:
        # Arrange: DataFrame with x/y coordinates but no name column
        df = pd.DataFrame({"x": [0, 3, 0], "y": [0, 0, 4]})
        calculator = EuclideanCalculator()

        # Act
        cost_matrix = calculator.calculate_cost_matrix(df)

        # Assert: Matrix should use DataFrame index as node names
        expected = pd.DataFrame(
            [[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]],
            index=[0, 1, 2],
            columns=[0, 1, 2],
        )
        pdt.assert_frame_equal(cost_matrix, expected, check_exact=False)
        pdt.assert_frame_equal(cost_matrix, cost_matrix.T, check_exact=False)


class TestGeodesicCalculator:
    def test_geodesic_calculator_latlon_happy_path(self) -> None:
        # Arrange: DataFrame with lat/lon coordinates and names (Oslo, Stockholm, Copenhagen)
        df = pd.DataFrame(
            {
                "name": ["Oslo", "Stockholm", "Copenhagen"],
                "lat": [59.9133, 59.3294, 55.6761],
                "lon": [10.7389, 18.0686, 12.5683],
            }
        )
        calculator = GeodesicCalculator()

        # Act
        cost_matrix = calculator.calculate_cost_matrix(df)

        # Assert: Matrix should be symmetric and distances reasonable
        # Oslo-Stockholm ~416km, Oslo-Copenhagen ~484km, Stockholm-Copenhagen ~522km
        expected = pd.DataFrame(
            [
                [
                    0.0,
                    pytest.approx(416_000, rel=0.05),
                    pytest.approx(484_000, rel=0.05),
                ],
                [
                    pytest.approx(416_000, rel=0.05),
                    0.0,
                    pytest.approx(522_000, rel=0.05),
                ],
                [
                    pytest.approx(484_000, rel=0.05),
                    pytest.approx(522_000, rel=0.05),
                    0.0,
                ],
            ],
            index=["Oslo", "Stockholm", "Copenhagen"],
            columns=["Oslo", "Stockholm", "Copenhagen"],
        )
        # Check symmetry and diagonal
        assert np.allclose(cost_matrix.values, cost_matrix.values.T)
        assert np.allclose(np.diag(cost_matrix.values), 0)
        # Check approximate values
        for i, j in itertools.product(range(3), range(3)):
            assert cost_matrix.iloc[i, j] == expected.iloc[i, j]


class TestCostMatrixFactory:
    def test_cost_matrix_factory_selection_happy_path(self) -> None:
        # Arrange: DataFrames for different coordinate systems
        df_xy = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        df_latlon = pd.DataFrame({"lat": [0, 1], "lon": [0, 1]})

        # Act & Assert: Default (should pick EuclideanCalculator for x/y)
        calc_xy = CostMatrixFactory.create_calculator(df_xy)
        assert isinstance(calc_xy, EuclideanCalculator)

        # Default (should pick GeodesicCalculator for lat/lon)
        calc_latlon = CostMatrixFactory.create_calculator(df_latlon)
        assert isinstance(calc_latlon, GeodesicCalculator)

        # Explicit method: euclidean
        calc_euclid = CostMatrixFactory.create_calculator(df_xy, method="euclidean")
        assert isinstance(calc_euclid, EuclideanCalculator)

        # Explicit method: geodesic
        calc_geo = CostMatrixFactory.create_calculator(df_latlon, method="geodesic")
        assert isinstance(calc_geo, GeodesicCalculator)

        # Explicit method: openroute should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            CostMatrixFactory.create_calculator(df_latlon, method="openroute")

        # Unknown method should raise ValueError
        with pytest.raises(ValueError):
            CostMatrixFactory.create_calculator(df_xy, method="unknown")


class TestGeodesicCalculatorEdge:
    def test_geodesic_calculator_missing_latlon_edge_case(self) -> None:
        # Arrange: DataFrame missing lat or lon
        df_missing_lat = pd.DataFrame({"lon": [10, 20]})
        df_missing_lon = pd.DataFrame({"lat": [50, 60]})
        calculator = GeodesicCalculator()

        # Act & Assert
        with pytest.raises(
            ValueError, match="DataFrame must have 'lat' and 'lon' columns"
        ):
            calculator.calculate_cost_matrix(df_missing_lat)
        with pytest.raises(
            ValueError, match="DataFrame must have 'lat' and 'lon' columns"
        ):
            calculator.calculate_cost_matrix(df_missing_lon)


class TestOpenRouteServiceCalculator:
    def test_openrouteservice_calculator_not_implemented_edge_case(self) -> None:
        # Arrange: DataFrame with lat/lon
        df = pd.DataFrame({"lat": [0, 1], "lon": [0, 1]})
        calculator = OpenRouteServiceCalculator(api_key="dummy")

        # Act & Assert
        with pytest.raises(
            NotImplementedError,
            match="OpenRouteService calculator requires 'requests' library and API implementation",
        ):
            calculator.calculate_cost_matrix(df)


class TestEuclideanCalculatorLatLonToUTM:
    def test_euclidean_calculator_latlon_to_utm_happy_path(self) -> None:
        # Arrange: DataFrame with only lat/lon columns (three cities forming a triangle)
        df = pd.DataFrame(
            {
                "name": ["P1", "P2", "P3"],
                "lat": [59.9139, 59.3293, 55.6761],
                "lon": [10.7522, 18.0686, 12.5683],
            }
        )
        calculator = EuclideanCalculator()

        # Act
        cost_matrix = calculator.calculate_cost_matrix(df)

        # Assert: Matrix should be symmetric and distances positive, diagonal zero
        assert cost_matrix.shape == (3, 3)
        assert (cost_matrix.index == df["name"]).all()
        assert (cost_matrix.columns == df["name"]).all()
        np.testing.assert_allclose(np.diag(cost_matrix.values), 0)
        assert np.allclose(cost_matrix.values, cost_matrix.values.T)
        # Distances should be positive and nonzero off-diagonal
        cm_float = cost_matrix.to_numpy(dtype=float)
        for i, j in itertools.product(range(3), range(3)):
            if i != j:
                assert cm_float[i, j] > 0


class TestCalculateCostMatrixFunction:
    def test_calculate_cost_matrix_function_happy_path(
        self, df_xy1: pd.DataFrame, df_xy1_facit_euclidian: pd.DataFrame
    ) -> None:
        # Act
        cost_matrix = calculate_cost_matrix(df_xy1, method="euclidean")

        # Assert: Matrix should be symmetric and distances correct
        pdt.assert_frame_equal(cost_matrix, df_xy1_facit_euclidian, check_exact=False)
        pdt.assert_frame_equal(cost_matrix, df_xy1_facit_euclidian.T, check_exact=False)
