from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

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

    graph = tsp_data.to_graph()
    graph.add_edges_from(arcs)

    # Get node positions and attributes for plotting
    pos = nx.get_node_attributes(graph, "pos")
    node_names = nx.get_node_attributes(graph, "name")
    node_types = nx.get_node_attributes(graph, "node_type")

    # Create edge traces for plotly
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None creates breaks between line segments
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#FF6B35"),  # Orange-red that works in both themes
        hoverinfo="none",
        mode="lines",
        name="TSP Route",
    )

    # Create color map for different node types with theme-neutral colors
    color_map = {
        "permanent": "#1f77b4",  # Plotly default blue - good contrast
        "start": "#2ca02c",  # Plotly default green - good contrast
        "end": "#d62728",  # Plotly default red - good contrast
        "startend": "#9467bd",  # Plotly default purple - good contrast
    }

    # Create node traces for plotly
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    # Create hover text with node metadata and colors
    hover_text = []
    node_text = []  # For displaying node names on the plot
    node_colors = []  # For node colors based on type

    for node in graph.nodes():
        name = node_names.get(node, str(node))
        node_type = node_types.get(node, "unknown")
        x, y = pos[node]

        # Text to display on nodes
        node_text.append(name)

        # Color based on node type
        node_colors.append(color_map.get(node_type, "#1f77b4"))

        # Hover information
        hover_info = (
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"Position: ({x:.2f}, {y:.2f})"
        )
        hover_text.append(hover_info)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#2F4F4F"),  # Dark gray that works in both themes
        marker=dict(
            size=8,
            color=node_colors,
            line=dict(width=1, color="#2F4F4F"),  # Dark gray border for visibility
        ),
        name="Nodes",
    )

    # Create the plotly figure
    objective_int = int(result["objective_value"])
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"TSP Solution - Objective Value: {objective_int}",
                font=dict(color="#2F4F4F"),  # Dark gray title that works in both themes
            ),
            font=dict(color="#2F4F4F"),  # Set default font color for all text elements
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            autosize=True,
        ),
    )

    # Save the plot as HTML file with scroll zoom enabled and removed toolbar buttons
    html_filename = Path(__file__).parent / "tsp_nodes_visualization2.html"
    config = {
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "displaylogo": False,
    }
    fig.write_html(html_filename, config=config)
    print(f"Plot saved as: {html_filename}")

    # Show the plot with scroll zoom enabled and removed toolbar buttons
    fig.show(config=config)


if __name__ == "__main__":
    main()
