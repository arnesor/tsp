import pandas as pd
import pyomo.environ as pyo


def valid_arc_filter(model: pyo.ConcreteModel, value: tuple[str, str]) -> bool:
    """All possible arcs connecting the sites (A)."""
    # only create pair (i, j) if site i and site j are different
    i, j = value  # Unpack the value tuple
    return i != j


def total_cost(model: pyo.ConcreteModel) -> float:
    """Total cost for the tour."""
    return pyo.summation(model.cost_ij, model.x_ij)


def node_is_entered_once(model: pyo.ConcreteModel, j: str) -> bool:
    """Each node j must be visited from exactly one other node."""
    return sum(model.x_ij[i, j] for i in model.nodes if i != j) == 1


def node_is_exited_once(model: pyo.ConcreteModel, i: str) -> bool:
    """Each node i must departure to exactly one other node."""
    return sum(model.x_ij[i, j] for j in model.nodes if j != i) == 1


def path_is_single_tour(model: pyo.ConcreteModel, i: str, j: str) -> bool:
    """For each pair of permanent nodes (i, j), if site j is visited from node i, the rank of j must be strictly greater than the rank of i."""
    if i == j:  # if node coincide, skip creating a constraint
        return pyo.Constraint.Skip

    r_i = model.rank_i[i]
    r_j = model.rank_i[j]
    x_ij = model.x_ij[i, j]
    # Standard MTZ form: r_j >= r_i + 1 - M * (1 - x_ij), with M = |S*|
    return r_j >= r_i + 1 - model.M * (1 - x_ij)


def create_tsp_model(
    cost_matrix: pd.DataFrame, startend_name: str | None = None
) -> pyo.ConcreteModel:
    if startend_name is not None:
        assert (
            startend_name in cost_matrix.index
        ), f"Startend node {startend_name} not found in cost matrix"

    model = pyo.ConcreteModel("TSP")
    list_of_nodes = cost_matrix.index.tolist()

    # Sets

    model.nodes = pyo.Set(
        initialize=list_of_nodes,
        within=pyo.Any,
        doc="Set of all nodes to be visited (S)",
    )

    model.nodes_except_startend = pyo.Set(
        initialize=model.nodes - {startend_name},
        within=model.nodes,
        doc="Nodes of interest, i.e., all nodes except the startend node (S*)",
    )

    model.valid_arcs = pyo.Set(
        initialize=model.nodes * model.nodes,  # S * S
        filter=valid_arc_filter,
        doc=valid_arc_filter.__doc__,
    )
    # (Avoid pprint in library code)

    # Parameters

    # Precompute initializer dict to avoid storing cost_matrix on the model
    costs_init = {
        (i, j): float(cost_matrix.at[i, j])
        for i in list_of_nodes
        for j in list_of_nodes
        if i != j
    }
    model.cost_ij = pyo.Param(
        model.valid_arcs,
        initialize=costs_init,
        doc="Cost between node i and node j (Cij).",
    )
    # model.cost_ij.pprint()

    # Use standard MTZ big-M: M = |S*|
    model.M = pyo.Param(
        initialize=len(model.nodes_except_startend),
        doc="Big-M constant for MTZ subtour elimination (M = |S*|)",
    )

    # Variables

    model.x_ij = pyo.Var(
        model.valid_arcs,
        domain=pyo.Binary,
        doc="Whether to go from node i to node j (x_ij)",
    )

    model.rank_i = pyo.Var(
        model.nodes_except_startend,  # i ∈ S* (index)
        domain=pyo.NonNegativeReals,  # rᵢ ∈ R₊ (domain)
        bounds=(1, len(model.nodes_except_startend)),  # 1 ≤ rᵢ ≤ |S*|
        doc="Rank of each node to track visit order",
    )

    # Objective function

    model.obj_total_cost = pyo.Objective(
        rule=total_cost,
        sense=pyo.minimize,
        doc=total_cost.__doc__,
    )

    # Constraints

    model.constr_each_site_is_entered_once = pyo.Constraint(
        model.nodes, rule=node_is_entered_once, doc=node_is_entered_once.__doc__
    )
    model.constr_each_site_is_exited_once = pyo.Constraint(
        model.nodes, rule=node_is_exited_once, doc=node_is_exited_once.__doc__
    )

    # Subtour elimination

    # cross product of permanent nodes, to index the constraint
    permanent_node_pairs = model.nodes_except_startend * model.nodes_except_startend

    model.constr_path_is_single_tour = pyo.Constraint(
        permanent_node_pairs, rule=path_is_single_tour, doc=path_is_single_tour.__doc__
    )
    # (Avoid pprint in library code)
    # model.pprint()

    return model
