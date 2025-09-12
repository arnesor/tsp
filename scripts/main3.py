from pathlib import Path

from tsp.cost_matrix import calculate_cost_matrix
from tsp.plot_graph import plot
from tsp.solver import Solver
from tsp.tsp_data import TspData
from tsp.tsp_model import create_tsp_model


def main() -> None:
    """Main function."""
    filename = Path(__file__).parent.parent / "data" / "kroA150-15.tsp"
    tsp_data = TspData.from_tsplib(filename)

    cost_matrix = calculate_cost_matrix(tsp_data.df)
    model = create_tsp_model(cost_matrix, tsp_data.get_start_node_name())

    solver = Solver(model, "highs")
    print(solver.get_solver_info())
    solver.print_model_info()

    result = solver.solve()
    objective_value = int(result["objective_value"])
    print(result)
    arcs = solver.extract_solution_as_arcs()

    graph = tsp_data.to_graph()
    graph.add_edges_from(arcs)

    plot(graph, title=f"Objective value: {objective_value}")


if __name__ == "__main__":
    main()
