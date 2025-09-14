import webbrowser
from pathlib import Path

import plotly.graph_objects as go

from tsp.tsp_data import CartesianTspData, TspData

filename = Path(__file__).parent.parent / "data" / "kroA150-15.tsp"
nodes = TspData.from_tsplib(filename)
assert isinstance(nodes, CartesianTspData)

G = nodes.to_graph()
print(f"Graph has {len(G.nodes)} nodes")

# Extract node coordinates and attributes
x_coords = []
y_coords = []
node_names = []
node_types = []

for _, attrs in G.nodes(data=True):
    x_coords.append(attrs["x"])
    y_coords.append(attrs["y"])
    node_names.append(attrs["name"])
    node_types.append(attrs["node_type"])

# Calculate bounding box with border
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# Add 10% border around the data
x_range = x_max - x_min
y_range = y_max - y_min
border_x = x_range * 0.1 if x_range > 0 else 1
border_y = y_range * 0.1 if y_range > 0 else 1

x_min_plot = x_min - border_x
x_max_plot = x_max + border_x
y_min_plot = y_min - border_y
y_max_plot = y_max + border_y

# Create color map for different node types
color_map = {"permanent": "blue", "start": "green", "end": "red", "startend": "purple"}

colors = [color_map.get(node_type, "blue") for node_type in node_types]

# Create the scatter plot
fig = go.Figure()

# Add scatter trace for nodes
fig.add_trace(
    go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        marker=dict(color=colors, size=8, line=dict(width=1, color="black")),
        text=node_names,
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>Type: %{customdata}<extra></extra>",
        customdata=node_types,
        name="Nodes",
    )
)

# Update layout for better visualization
fig.update_layout(
    title=f"TSP Nodes Visualization ({len(G.nodes)} nodes)",
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    xaxis=dict(
        range=[x_min_plot, x_max_plot],
        scaleanchor="y",
        scaleratio=1,
    ),
    yaxis=dict(
        range=[y_min_plot, y_max_plot],
    ),
    showlegend=False,
    width=800,
    height=600,
    hovermode="closest",
)

# Save the plot as HTML file and open in browser
html_filename = Path(__file__).parent / "tsp_nodes_visualization.html"
fig.write_html(html_filename, config={"scrollZoom": True})
print(f"Plot saved as: {html_filename}")

# Open the HTML file in the default browser
webbrowser.open(f"file://{html_filename.absolute()}")
