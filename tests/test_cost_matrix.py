import pandas as pd

from tsp.cost_matrix import (
    EuclideanCalculator,
    GeodesicCalculator,
    CostMatrixFactory,
    calculate_cost_matrix,
)


def test_euclidean_calculator():
    """Test EuclideanCalculator with x/y coordinates."""
    print("Testing EuclideanCalculator...")

    # Create test data with x/y coordinates
    data = {"id": [1, 2, 3], "name": ["A", "B", "C"], "x": [0, 3, 0], "y": [0, 0, 4]}
    df = pd.DataFrame(data)

    calculator = EuclideanCalculator()

    # Test supports_coordinates
    assert calculator.supports_coordinates(df), "Should support x/y coordinates"

    # Test calculate_cost_matrix
    cost_matrix = calculator.calculate_cost_matrix(df)

    # Verify matrix shape
    assert cost_matrix.shape == (3, 3), f"Expected (3, 3), got {cost_matrix.shape}"

    # Verify diagonal is zero
    for i in range(3):
        assert cost_matrix.iloc[i, i] == 0, f"Diagonal element [{i}, {i}] should be 0"

    # Verify specific distances
    # Distance from A(0,0) to B(3,0) should be 0.003
    assert (
        abs(cost_matrix.loc["A", "B"] - 0.003) < 1e-10
    ), f"Expected 3.0, got {cost_matrix.loc['A', 'B']}"

    # Distance from A(0,0) to C(0,4) should be 4
    assert (
        abs(cost_matrix.loc["A", "C"] - 0.004) < 1e-10
    ), f"Expected 4.0, got {cost_matrix.loc['A', 'C']}"

    # Distance from B(3,0) to C(0,4) should be 5 (3-4-5 triangle)
    assert (
        abs(cost_matrix.loc["B", "C"] - 0.005) < 1e-10
    ), f"Expected 5.0, got {cost_matrix.loc['B', 'C']}"

    print("✓ EuclideanXYCalculator tests passed")


def test_geodesic_calculator():
    """Test GeodesicCalculator with lat/lon coordinates."""
    print("Testing GeodesicCalculator...")

    # Create test data with lat/lon coordinates (Paris area)
    data = {
        "id": [1, 2],
        "name": ["Louvre", "Eiffel Tower"],
        "lat": [48.8607, 48.8584],
        "lon": [2.3376, 2.2945],
    }
    df = pd.DataFrame(data)

    calculator = GeodesicCalculator()

    # Test supports_coordinates
    assert calculator.supports_coordinates(df), "Should support lat/lon coordinates"

    # Test calculate_cost_matrix
    cost_matrix = calculator.calculate_cost_matrix(df)

    # Verify matrix shape
    assert cost_matrix.shape == (2, 2), f"Expected (2, 2), got {cost_matrix.shape}"

    # Verify diagonal is zero
    assert cost_matrix.iloc[0, 0] == 0, "Diagonal should be 0"
    assert cost_matrix.iloc[1, 1] == 0, "Diagonal should be 0"

    # Verify distance is reasonable (should be around 3-4 km between Louvre and Eiffel Tower)
    distance = cost_matrix.loc["Louvre", "Eiffel Tower"]
    assert 2 < distance < 6, f"Expected distance between 2-6 km, got {distance}"

    # Verify symmetry
    assert (
        abs(
            cost_matrix.loc["Louvre", "Eiffel Tower"]
            - cost_matrix.loc["Eiffel Tower", "Louvre"]
        )
        < 1e-10
    )

    print("✓ GeodesicCalculator tests passed")


def test_cost_matrix_factory():
    """Test CostMatrixFactory for automatic calculator selection."""
    print("Testing CostMatrixFactory...")

    # Test with x/y coordinates
    xy_data = {"id": [1, 2], "name": ["A", "B"], "x": [0, 1], "y": [0, 1]}
    xy_df = pd.DataFrame(xy_data)

    calculator = CostMatrixFactory.create_calculator(xy_df)
    assert isinstance(
        calculator, EuclideanCalculator
    ), "Should create EuclideanCalculator for x/y data"

    # Test with lat/lon coordinates
    latlon_data = {
        "id": [1, 2],
        "name": ["A", "B"],
        "lat": [48.8607, 48.8584],
        "lon": [2.3376, 2.2945],
    }
    latlon_df = pd.DataFrame(latlon_data)

    calculator = CostMatrixFactory.create_calculator(latlon_df)
    assert isinstance(
        calculator, EuclideanCalculator
    ), "Should create EuclideanCalculator for lat/lon data"

    calculator = CostMatrixFactory.create_calculator(latlon_df, "geodesic")
    assert isinstance(
        calculator, GeodesicCalculator
    ), "Should create GeodesicCalculator when geodesic is given."

    # Test error cases
    try:
        invalid_data = {"id": [1, 2], "name": ["A", "B"]}
        invalid_df = pd.DataFrame(invalid_data)
        CostMatrixFactory.create_calculator(invalid_df)
        assert False, "Should raise ValueError for invalid coordinates"
    except ValueError as e:
        assert "must have either" in str(e), f"Unexpected error message: {e}"

    print("✓ CostMatrixFactory tests passed")


def test_calculate_cost_matrix_convenience_function():
    """Test the convenience function calculate_cost_matrix."""
    print("Testing calculate_cost_matrix convenience function...")

    # Test with x/y coordinates
    xy_data = {"id": [1, 2, 3], "name": ["A", "B", "C"], "x": [0, 1, 0], "y": [0, 0, 1]}
    xy_df = pd.DataFrame(xy_data)

    cost_matrix = calculate_cost_matrix(xy_df)

    # Verify it returns a DataFrame
    assert isinstance(cost_matrix, pd.DataFrame), "Should return a DataFrame"

    # Verify shape
    assert cost_matrix.shape == (3, 3), f"Expected (3, 3), got {cost_matrix.shape}"

    # Verify diagonal is zero
    for i in range(3):
        assert cost_matrix.iloc[i, i] == 0, f"Diagonal element [{i}, {i}] should be 0"

    # Test with specific method
    cost_matrix_geodesic = calculate_cost_matrix(
        pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["A", "B"],
                "lat": [48.8607, 48.8584],
                "lon": [2.3376, 2.2945],
            }
        ),
        method="geodesic",
    )

    assert isinstance(cost_matrix_geodesic, pd.DataFrame), "Should return a DataFrame"
    assert cost_matrix_geodesic.shape == (2, 2), "Should have correct shape"

    print("✓ calculate_cost_matrix convenience function tests passed")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")

    # Test with single point
    single_point_data = {"id": [1], "name": ["A"], "x": [0], "y": [0]}
    single_df = pd.DataFrame(single_point_data)

    calculator = EuclideanCalculator()
    cost_matrix = calculator.calculate_cost_matrix(single_df)
    assert cost_matrix.shape == (1, 1), "Should handle single point"
    assert cost_matrix.iloc[0, 0] == 0, "Single point distance to itself should be 0"

    # Test with DataFrame without name column (should use index)
    no_name_data = {"id": [1, 2], "x": [0, 1], "y": [0, 1]}
    no_name_df = pd.DataFrame(no_name_data)

    cost_matrix = calculator.calculate_cost_matrix(no_name_df)
    assert cost_matrix.shape == (2, 2), "Should handle DataFrame without name column"

    print("✓ Edge cases tests passed")


if __name__ == "__main__":
    try:
        test_euclidean_calculator()
        test_geodesic_calculator()
        test_cost_matrix_factory()
        test_calculate_cost_matrix_convenience_function()
        test_edge_cases()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
