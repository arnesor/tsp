"""Demonstration of the cost matrix calculation system integrated with TSP solving.

This script shows how to use the new cost matrix calculation classes with
different coordinate systems and distance calculation methods.
"""

import pandas as pd
import pyomo.environ as pyo

from .cost_matrix import CostMatrixFactory, calculate_cost_matrix
from .tsp_data import TspData


def create_sample_xy_data() -> pd.DataFrame:
    """Create sample data with x/y coordinates."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["hotel", "museum", "park", "restaurant"],
            "node_type": ["startend", "permanent", "permanent", "permanent"],
            "x": [0, 10, 5, 8],
            "y": [0, 0, 8, 12],
        }
    )


def create_sample_latlon_data() -> pd.DataFrame:
    """Create sample data with lat/lon coordinates (Paris area)."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["hotel", "Louvre", "Eiffel Tower", "Notre Dame"],
            "node_type": ["startend", "permanent", "permanent", "permanent"],
            "lat": [48.8527, 48.8607, 48.8584, 48.8530],
            "lon": [2.3542, 2.3376, 2.2945, 2.3499],
        }
    )


def solve_tsp_with_cost_matrix(
    df: pd.DataFrame, cost_matrix: pd.DataFrame[float]
) -> pyo.ConcreteModel:
    """Solve TSP using the provided cost matrix.

    Args:
        df: DataFrame with node data
        cost_matrix: Cost matrix as DataFrame

    Returns:
        The solved Pyomo model
    """
    # Create TSP model
    tsp_model = pyo.ConcreteModel("TSP_with_cost_matrix")

    # Sets
    sites = cost_matrix.index.tolist()
    tsp_model.sites = pyo.Set(initialize=sites, doc="Set of all sites")

    # Valid arcs (all pairs except self-loops)
    def valid_arc_filter(model: pyo.ConcreteModel, i: int, j: int) -> bool:
        return i != j

    tsp_model.valid_arcs = pyo.Set(
        initialize=tsp_model.sites * tsp_model.sites,
        filter=valid_arc_filter,
        doc="Valid arcs between sites",
    )

    # Parameters - distance from cost matrix
    def distance_param(model: pyo.ConcreteModel, i: int, j: int) -> float:
        return cost_matrix.loc[i, j]

    tsp_model.distance = pyo.Param(
        tsp_model.valid_arcs, initialize=distance_param, doc="Distance between sites"
    )

    # Variables
    tsp_model.x = pyo.Var(
        tsp_model.valid_arcs,
        within=pyo.Binary,
        doc="Whether to travel from site i to site j",
    )

    # Objective - minimize total distance
    def total_distance(model: pyo.ConcreteModel) -> float:
        return pyo.summation(model.distance, model.x)

    tsp_model.obj = pyo.Objective(rule=total_distance, sense=pyo.minimize)

    # Constraints
    # Each site must be entered exactly once
    def enter_once(model: pyo.ConcreteModel, j: int) -> bool:
        return sum(model.x[i, j] for i in model.sites if i != j) == 1

    tsp_model.enter_once = pyo.Constraint(tsp_model.sites, rule=enter_once)

    # Each site must be exited exactly once
    def exit_once(model: pyo.ConcreteModel, i: int) -> bool:
        return sum(model.x[i, j] for j in model.sites if i != j) == 1

    tsp_model.exit_once = pyo.Constraint(tsp_model.sites, rule=exit_once)

    # Subtour elimination (simplified - for demonstration)
    # In practice, you'd use more sophisticated subtour elimination

    return tsp_model


def demonstrate_xy_coordinates() -> None:
    """Demonstrate cost matrix calculation with x/y coordinates."""
    print("=== Demonstration with x/y coordinates ===")

    # Create sample data
    df = create_sample_xy_data()
    print("Sample data:")
    print(df)
    print()

    # Create TspData instance (validation happens automatically)
    try:
        tsp_data = TspData.from_dataframe(df)
        print("✓ Data validation passed")
        print(f"✓ Detected coordinate system: {tsp_data.coordinate_system}")
        print(f"✓ Data type: {type(tsp_data).__name__}")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return

    # Calculate cost matrix using TspData method
    cost_matrix = tsp_data.calculate_cost_matrix()
    print("\nCost matrix (Euclidean distance):")
    print(cost_matrix.round(2))
    print()

    # Show factory selection
    calculator = CostMatrixFactory.create_calculator(tsp_data)
    print(f"Automatically selected calculator: {type(calculator).__name__}")
    print()


def demonstrate_latlon_coordinates() -> None:
    """Demonstrate cost matrix calculation with lat/lon coordinates."""
    print("=== Demonstration with lat/lon coordinates ===")

    # Create sample data
    df = create_sample_latlon_data()
    print("Sample data:")
    print(df)
    print()

    # Create TspData instance (validation happens automatically)
    try:
        tsp_data = TspData.from_dataframe(df)
        print("✓ Data validation passed")
        print(f"✓ Detected coordinate system: {tsp_data.coordinate_system}")
        print(f"✓ Data type: {type(tsp_data).__name__}")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return

    # Calculate cost matrix using different methods
    methods = ["geodesic", "euclidean"]

    for method in methods:
        print(f"\nCost matrix using {method} method:")
        cost_matrix = tsp_data.calculate_cost_matrix(method=method)
        print(cost_matrix.round(3))

        # Show calculator type
        calculator = CostMatrixFactory.create_calculator(tsp_data, method=method)
        print(f"Calculator used: {type(calculator).__name__}")

    print()


def demonstrate_tsp_integration() -> None:
    """Demonstrate integration with TSP solver."""
    print("=== TSP Integration Demonstration ===")

    # Use x/y data for simplicity
    df = create_sample_xy_data()
    tsp_data = TspData.from_dataframe(df)
    cost_matrix = tsp_data.calculate_cost_matrix()

    print("Solving TSP with calculated cost matrix...")
    print("Cost matrix:")
    print(cost_matrix.round(2))

    # Create and solve TSP model
    model = solve_tsp_with_cost_matrix(df, cost_matrix)

    # Check if solver is available
    solver = pyo.SolverFactory("highs")
    if not solver.available(exception_flag=False):
        print("❌ HiGHS solver not available - cannot solve TSP")
        return

    # Solve the model
    try:
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("✓ Optimal solution found!")
            print(f"Total distance: {pyo.value(model.obj):.2f}")

            # Extract solution
            print("\nOptimal route:")
            for i, j in model.valid_arcs:
                if pyo.value(model.x[i, j]) > 0.5:  # Binary variable is 1
                    distance = pyo.value(model.distance[i, j])
                    print(f"  {i} → {j} (distance: {distance:.2f})")
        else:
            print(
                f"❌ Solver terminated with condition: {results.solver.termination_condition}"
            )

    except Exception as e:
        print(f"❌ Error solving TSP: {e}")

    print()


def demonstrate_method_comparison() -> None:
    """Compare different distance calculation methods."""
    print("=== Method Comparison ===")

    # Use lat/lon data
    #    df = create_sample_latlon_data()
    df = pd.DataFrame(
        {
            "name": ["Oslo", "Stockholm", "Copenhagen"],
            "lat": [59.9133, 59.3294, 55.6761],
            "lon": [10.7389, 18.0686, 12.5683],
        }
    )

    # Calculate distances between first two points using different methods
    point1 = df.iloc[0]["name"]
    point2 = df.iloc[2]["name"]

    methods = ["geodesic", "euclidean"]

    print(f"Distance between {point1} and {point2}:")

    for method in methods:
        cost_matrix = calculate_cost_matrix(df, method=method)
        distance = cost_matrix.loc[point1, point2]
        print(f"  {method}: {distance:.3f} km")

    print()


if __name__ == "__main__":
    print("Cost Matrix Calculation System Demonstration")
    print("=" * 50)
    print()

    try:
        demonstrate_xy_coordinates()
        demonstrate_latlon_coordinates()
        demonstrate_method_comparison()
        demonstrate_tsp_integration()

        print("🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        raise
