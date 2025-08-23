import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pyomo.environ as pyo
import pyomo.version

solver = pyo.SolverFactory("highs")
solver_available = solver.available(exception_flag=False)
print(f"Solver '{solver.name}' available: {solver_available}")

if solver_available:
    print(f"Solver version: {solver.version()}")

print("pyomo version:", pyomo.version.__version__)
print("networkx version:", nx.__version__)

path_data = (
    "https://raw.githubusercontent.com/carlosjuribe/"
    "traveling-tourist-problem/main/"
    "Paris_sites_spherical_distance_matrix.csv"
)
df_distances = pd.read_csv(path_data, index_col="site")
print(df_distances)


m_tsp = pyo.ConcreteModel("TSP")

# Sets

list_of_sites = df_distances.index.tolist()
m_tsp.sites = pyo.Set(
    initialize=list_of_sites, domain=pyo.Any, doc="set of all sites to be visited (S)"
)
m_tsp.sites_except_hotel = pyo.Set(
    initialize=m_tsp.sites - {"hotel"},
    domain=m_tsp.sites,
    doc="Sites of interest, i.e., all sites except the hotel (S*)",
)


def valid_arc_filter(model: pyo.ConcreteModel, value: tuple[str, str]) -> bool:
    """All possible arcs connecting the sites (A)."""
    # only create pair (i, j) if site i and site j are different
    i, j = value  # Unpack the value tuple
    return i != j


m_tsp.valid_arcs = pyo.Set(
    initialize=m_tsp.sites * m_tsp.sites,  # S * S
    filter=valid_arc_filter,
    doc=valid_arc_filter.__doc__,
)
# m_tsp.valid_arcs.pprint()

# Parameters


def distance_between_sites(model: pyo.ConcreteModel, i: str, j: str) -> float:
    """Distance between site i and site j (Dij)."""
    return df_distances.at[i, j]  # fetch the distance from dataframe


m_tsp.distance_ij = pyo.Param(
    m_tsp.valid_arcs,
    initialize=distance_between_sites,
    doc=distance_between_sites.__doc__,
)
# model.distance_ij.pprint()

m_tsp.M = pyo.Param(
    initialize=1 - len(m_tsp.sites_except_hotel),
    doc="big M to make some constraints redundant",
)

# Variables

m_tsp.x_ij = pyo.Var(
    m_tsp.valid_arcs,
    within=pyo.Binary,
    doc="Whether to go from site i to site j (x_ij)",
)

m_tsp.rank_i = pyo.Var(
    m_tsp.sites_except_hotel,  # i ∈ S* (index)
    within=pyo.NonNegativeReals,  # rᵢ ∈ R₊ (domain)
    bounds=(1, len(m_tsp.sites_except_hotel)),  # 1 ≤ rᵢ ≤ |S*|
    doc="Rank of each site to track visit order",
)

# Objective function


def total_distance_traveled(model: pyo.ConcreteModel) -> float:
    """Total distance traveled."""
    return pyo.summation(model.distance_ij, model.x_ij)


m_tsp.obj_total_distance = pyo.Objective(
    rule=total_distance_traveled,
    sense=pyo.minimize,
    doc=total_distance_traveled.__doc__,
)

# Constraints


def site_is_entered_once(model: pyo.ConcreteModel, j: str) -> bool:
    """Each site j must be visited from exactly one other site."""
    return sum(model.x_ij[i, j] for i in model.sites if i != j) == 1


def site_is_exited_once(model: pyo.ConcreteModel, i: str) -> bool:
    """Each site i must departure to exactly one other site."""
    return sum(model.x_ij[i, j] for j in model.sites if j != i) == 1


m_tsp.constr_each_site_is_entered_once = pyo.Constraint(
    m_tsp.sites, rule=site_is_entered_once, doc=site_is_entered_once.__doc__
)
m_tsp.constr_each_site_is_exited_once = pyo.Constraint(
    m_tsp.sites, rule=site_is_exited_once, doc=site_is_exited_once.__doc__
)


# Subtour elimination
def path_is_single_tour(model: pyo.ConcreteModel, i: str, j: str) -> bool:
    """For each pair of non-startend sites (i, j), if site j is visited from site i, the rank of j must be strictly greater than the rank of i."""
    if i == j:  # if sites coincide, skip creating a constraint
        return pyo.Constraint.Skip

    r_i = model.rank_i[i]
    r_j = model.rank_i[j]
    x_ij = model.x_ij[i, j]
    return r_j >= r_i + x_ij + (1 - x_ij) * model.M


# cross product of non-hotel sites, to index the constraint
non_hotel_site_pairs = m_tsp.sites_except_hotel * m_tsp.sites_except_hotel

m_tsp.constr_path_is_single_tour = pyo.Constraint(
    non_hotel_site_pairs, rule=path_is_single_tour, doc=path_is_single_tour.__doc__
)
# m_tsp.constr_path_is_single_tour.pprint()
m_tsp.pprint()


def print_model_info(model: pyo.ConcreteModel) -> None:
    """Print model information."""
    print(
        f"Name: {model.name}",
        f"Num variables: {model.nvariables()}",
        f"Num constraints: {model.nconstraints()}",
        sep="\n- ",
    )


print_model_info(m_tsp)

res = solver.solve(m_tsp)  # optimize the model
print(f"Optimal solution found: {pyo.check_optimal_termination(res)}")
print(f"Objective value: {pyo.value(m_tsp.obj_total_distance):.0f}")
m_tsp.x_ij.pprint()


def extract_solution_as_arcs(model: pyo.ConcreteModel) -> list[tuple[str, str]]:
    """Extract a list of active (selected) arcs from the solved model."""
    return [
        (i, j) for i, j in model.valid_arcs if math.isclose(model.x_ij[i, j].value, 1.0)
    ]


def plot_arcs_as_graph(tour_as_arcs: list[tuple[str, str]]) -> None:
    """Take in a list of arcs, convert it to a networkx graph and draw it."""
    G: nx.DiGraph = nx.DiGraph()
    G.add_edges_from(tour_as_arcs)  # store solution as graph

    node_colors = ["red" if node == "hotel" else "skyblue" for node in G.nodes()]
    nx.draw(
        G,
        node_color=node_colors,
        with_labels=True,
        font_size=6,
        node_shape="o",
        arrowsize=5,
        style="solid",
    )
    plt.show()


def plot_solution_as_graph(model: pyo.ConcreteModel) -> None:
    """Plot the solution of the given model as a graph."""
    print(f"Total distance: {model.obj_total_distance()}")

    active_arcs = extract_solution_as_arcs(model)
    plot_arcs_as_graph(active_arcs)


plot_solution_as_graph(m_tsp)


def main() -> None:
    """Main function."""
    print("Hello from tsp!")


if __name__ == "__main__":
    main()
