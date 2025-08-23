from pathlib import Path

import pyomo.environ as pyo
import pyomo.version

from tsp.cost_matrix import calculate_cost_matrix
from tsp.tsp_data import TspData
from tsp.tsp_model import create_tsp_model


def print_model_info(model: pyo.ConcreteModel) -> None:
    """Print model information."""
    print(
        f"Name: {model.name}",
        f"Num variables: {model.nvariables()}",
        f"Num constraints: {model.nconstraints()}",
        sep="\n- ",
    )


def main() -> None:
    """Main function."""
    filename = Path(__file__).parent.parent / "data" / "Paris.csv"
    tsp_data = TspData.from_csv(filename)

    cost_matrix = calculate_cost_matrix(tsp_data.df, method="geodesic")
    model = create_tsp_model(cost_matrix, tsp_data.get_start_node_name())

    # model.pprint()
    # latex_printer(model, ostream="model.tex", use_equation_environment=True)

    # Solving

    solver = pyo.SolverFactory("highs")
    solver_available = solver.available(exception_flag=False)
    print(f"Solver '{solver.name}' available: {solver_available}")

    if solver_available:
        print(f"Solver version: {solver.version()}")
    print("pyomo version:", pyomo.version.__version__)

    print_model_info(model)

    result = solver.solve(model)  # optimize the model
    print(f"Optimal solution found: {pyo.check_optimal_termination(result)}")
    print(f"Objective value: {pyo.value(model.total_cost):.0f}")
    # model.x_ij.pprint()


if __name__ == "__main__":
    main()
