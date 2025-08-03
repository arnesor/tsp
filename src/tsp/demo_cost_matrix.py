"""
Demonstration of the cost matrix calculation system integrated with TSP solving.

This script shows how to use the new cost matrix calculation classes with
different coordinate systems and distance calculation methods.
"""

import pandas as pd
import pyomo.environ as pyo
from cost_matrix import calculate_cost_matrix, CostMatrixFactory
from node_schema import NodeInputModel


def create_sample_xy_data():
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


def create_sample_latlon_data():
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


def solve_tsp_with_cost_matrix(df, cost_matrix):
    """
    Solve TSP using the provided cost matrix.

    Args:
        df: DataFrame with node data
        cost_matrix: Cost matrix as DataFrame

    Returns:
        Solved Pyomo model
    """
    # Create TSP model
    model = pyo.ConcreteModel("TSP_with_cost_matrix")

    # Sets
    sites = cost_matrix.index.tolist()
    model.sites = pyo.Set(initialize=sites, doc="Set of all sites")

    # Valid arcs (all pairs except self-loops)
    def valid_arc_filter(model, i, j):
        return i != j

    model.valid_arcs = pyo.Set(
        initialize=model.sites * model.sites,
        filter=valid_arc_filter,
        doc="Valid arcs between sites",
    )

    # Parameters - distance from cost matrix
    def distance_param(model, i, j):
        return cost_matrix.loc[i, j]

    model.distance = pyo.Param(
        model.valid_arcs, initialize=distance_param, doc="Distance between sites"
    )

    # Variables
    model.x = pyo.Var(
        model.valid_arcs,
        within=pyo.Binary,
        doc="Whether to travel from site i to site j",
    )

    # Objective - minimize total distance
    def total_distance(model):
        return pyo.summation(model.distance, model.x)

    model.obj = pyo.Objective(rule=total_distance, sense=pyo.minimize)

    # Constraints
    # Each site must be entered exactly once
    def enter_once(model, j):
        return sum(model.x[i, j] for i in model.sites if i != j) == 1

    model.enter_once = pyo.Constraint(model.sites, rule=enter_once)

    # Each site must be exited exactly once
    def exit_once(model, i):
        return sum(model.x[i, j] for j in model.sites if i != j) == 1

    model.exit_once = pyo.Constraint(model.sites, rule=exit_once)

    # Subtour elimination (simplified - for demonstration)
    # In practice, you'd use more sophisticated subtour elimination

    return model


def demonstrate_xy_coordinates():
    """Demonstrate cost matrix calculation with x/y coordinates."""
    print("=== Demonstration with x/y coordinates ===")

    # Create sample data
    df = create_sample_xy_data()
    print("Sample data:")
    print(df)
    print()

    # Validate data
    try:
        NodeInputModel.validate(df)
        print("✓ Data validation passed")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return

    # Calculate cost matrix using default method (Euclidean for x/y)
    cost_matrix = calculate_cost_matrix(df)
    print("\nCost matrix (Euclidean distance):")
    print(cost_matrix.round(2))
    print()

    # Show factory selection
    calculator = CostMatrixFactory.create_calculator(df)
    print(f"Automatically selected calculator: {type(calculator).__name__}")
    print()


def demonstrate_latlon_coordinates():
    """Demonstrate cost matrix calculation with lat/lon coordinates."""
    print("=== Demonstration with lat/lon coordinates ===")

    # Create sample data
    df = create_sample_latlon_data()
    print("Sample data:")
    print(df)
    print()

    # Validate data
    try:
        NodeInputModel.validate(df)
        print("✓ Data validation passed")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return

    # Calculate cost matrix using different methods
    methods = ["geodesic", "euclidean"]

    for method in methods:
        print(f"\nCost matrix using {method} method:")
        cost_matrix = calculate_cost_matrix(df, method=method)
        print(cost_matrix.round(3))

        # Show calculator type
        calculator = CostMatrixFactory.create_calculator(df, method=method)
        print(f"Calculator used: {type(calculator).__name__}")

    print()


def demonstrate_tsp_integration():
    """Demonstrate integration with TSP solver."""
    print("=== TSP Integration Demonstration ===")

    # Use x/y data for simplicity
    df = create_sample_xy_data()
    cost_matrix = calculate_cost_matrix(df)

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


def demonstrate_method_comparison():
    """Compare different distance calculation methods."""
    print("=== Method Comparison ===")

    # Use lat/lon data
    df = create_sample_latlon_data()

    # Calculate distances between first two points using different methods
    point1 = df.iloc[0]["name"]
    point2 = df.iloc[1]["name"]

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
