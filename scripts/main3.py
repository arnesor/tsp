from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from tsp.cost_matrix import calculate_cost_matrix
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
    print(result)
    arcs = solver.extract_solution_as_arcs()

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes with coordinates from tsp_data.df
    for _, row in tsp_data.df.iterrows():
        node_name = row["name"]
        # Determine coordinate columns based on data structure
        if "x" in tsp_data.df.columns and "y" in tsp_data.df.columns:
            # Cartesian coordinates
            pos = (row["x"], row["y"])
        elif "lat" in tsp_data.df.columns and "lon" in tsp_data.df.columns:
            # Geographic coordinates
            pos = (row["lon"], row["lat"])  # Use lon as x, lat as y
        else:
            pos = (0, 0)  # Fallback

        G.add_node(node_name, pos=pos)

    # Add solution arcs as edges
    G.add_edges_from(arcs)

    # Get node positions for plotting
    pos = nx.get_node_attributes(G, "pos")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color="red", node_size=50)
    nx.draw_networkx_edges(G, pos, edge_color="blue", width=1)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Add title with objective value (truncated to integer)
    objective_int = int(result["objective_value"])
    plt.title(f"TSP Solution - Objective Value: {objective_int}", fontsize=14)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
