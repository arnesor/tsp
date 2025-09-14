import time
from pathlib import Path

import pyomo.environ as pyo
import pyomo.version
from pyomo.contrib.solver.common.factory import SolverFactory

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
    # filename = Path(__file__).parent.parent / "data" / "Paris.csv"
    # tsp_data = TspData.from_csv(filename)

    filename = Path(__file__).parent.parent / "data" / "kroA150-15.tsp"
    tsp_data = TspData.from_tsplib(filename)

    cost_matrix = calculate_cost_matrix(tsp_data.df)
    model = create_tsp_model(cost_matrix, tsp_data.get_start_node_name())

    # model.pprint()
    # latex_printer(model, ostream="model.tex", use_equation_environment=True)

    # Solving

    solver_name = "highs"  # highs or scip

    if solver_name == "highs":
        solver = SolverFactory("highs")
    else:
        solver = pyo.SolverFactory("scip")
    solver_available = solver.available()
    print(f"Solver '{solver.name}' available: {bool(solver_available)}")

    if solver_available:
        print(f"Solver version: {solver.version()}")
    print("pyomo version:", pyomo.version.__version__)

    print_model_info(model)

    start_time = time.perf_counter()
    result = solver.solve(model)  # optimize the model
    wall_time = time.perf_counter() - start_time

    print(f"Optimal solution found: {pyo.check_optimal_termination(result)}")
    print(f"Objective value: {pyo.value(model.total_cost):.0f}")
    print(f"Solution time (measured): {wall_time:.1f} s")

    if solver_name == "highs":
        result.timing_info.display()


if __name__ == "__main__":
    main()
