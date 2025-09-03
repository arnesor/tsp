from pathlib import Path

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def get_node_colors(graph: nx.Graph) -> list[str]:
    """Returns a list of colors for each node based on its node_color_index."""
    # Create a color map for different node types, maps to color index in theme
    color_map = {
        "permanent": 0,
        "start": 5,
        "end": 8,
        "startend": 2,
    }
    default_idx = 0

    node_types = nx.get_node_attributes(graph, "node_type")
    node_color_indices = [
        color_map.get(node_types.get(n), default_idx) for n in graph.nodes()
    ]

    tpl_name = pio.templates.default
    colorway = pio.templates[tpl_name].layout.colorway or px.colors.qualitative.Plotly
    return [colorway[i % len(colorway)] for i in node_color_indices]


def get_hover_text(graph: nx.Graph) -> list[str]:
    """Returns a list of hover text strings for each node.

    Args:
        graph: NetworkX graph with node attributes

    Returns:
        List of formatted hover text strings for each node
    """
    # Get attribute dictionaries using NetworkX's proper API
    names = nx.get_node_attributes(graph, "name")
    node_types = nx.get_node_attributes(graph, "node_type")
    positions = nx.get_node_attributes(graph, "pos")

    hover_texts = []
    for node in graph.nodes():
        # Safely get attributes with defaults
        name = names.get(node, "noname")
        node_type = node_types.get(node, "unknown")
        pos = positions.get(node, (0, 0))

        hover_info = (
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"Position: ({pos[0]:.0f}, {pos[1]:.0f})"
        )
        hover_texts.append(hover_info)
    return hover_texts


def plot(graph: nx.Graph, file: Path | None = None) -> None:
    """Plots a graph using Plotly.

    Args:
        graph: NetworkX graph with required node attributes: pos, name, and node_type
        file: Optional path to save the plot

    Raises:
        ValueError: If any required node attributes are missing
    """
    # Set color template, plotly, plotly_dark, ggplot2
    pio.templates.default = "plotly"

    required_attributes = ["pos", "name", "node_type"]

    # Check that all nodes have required attributes
    for attr in required_attributes:
        attr_dict = nx.get_node_attributes(graph, attr)
        missing_nodes = [node for node in graph.nodes() if node not in attr_dict]
        if missing_nodes:
            raise ValueError(
                f"Nodes {missing_nodes} are missing required attribute '{attr}'"
            )

    pos = nx.get_node_attributes(graph, "pos")
    names = nx.get_node_attributes(graph, "name")

    # Create node traces for plotly
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]
    node_colors = get_node_colors(graph)
    hover_texts = get_hover_text(graph)

    # Use node names for display text when available, fallback to node IDs
    display_texts = [names.get(node, str(node)) for node in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=display_texts,
        marker=dict(color=node_colors),
        textposition="top center",
        hoverinfo="text",
        hovertext=hover_texts,
    )

    # Create edge traces for plotly
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None creates breaks between line segments
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="skip", name="TSP Route"
    )

    config = {
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "displaylogo": False,
    }

    fig = go.Figure(
        data=[node_trace, edge_trace],
        layout=go.Layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            autosize=True,
        ),
    )
    fig.show(config=config)
