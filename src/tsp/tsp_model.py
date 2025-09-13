from typing import Any

import pandas as pd
import pyomo.environ as pyo


def create_tsp_model(
    cost_matrix: pd.DataFrame, startend_name: str | None = None
) -> pyo.ConcreteModel:
    """Creates a Travelling Salesman Problem (TSP) model using Pyomo.

    This function generates a ConcreteModel instance for the TSP based
    on the provided cost matrix.

    Args:
        cost_matrix: A square DataFrame where both the index
            and columns represent the nodes, and each entry represents the
            travel cost between two nodes. Diagonal entries should be zero
            (or ignored).
        startend_name: Optional. The name of the start and end node (start-end depot).
            If not provided, the TSP may use any node as the start or end node.

    Returns:
        A Pyomo ConcreteModel instance encapsulating the TSP.
    """
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

    def valid_arc_filter(model: pyo.ConcreteModel, value: tuple[str, str]) -> bool:
        """All possible arcs connecting the sites (A)."""
        # only create pair (i, j) if site i and site j are different
        i, j = value  # Unpack the value tuple
        return i != j

    model.valid_arcs = pyo.Set(
        initialize=model.nodes * model.nodes,  # S * S
        filter=valid_arc_filter,
        doc=valid_arc_filter.__doc__,
    )

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

    @model.Objective(sense=pyo.minimize, doc="Total cost for the tour.")  # type: ignore[misc]
    def total_cost(m: pyo.ConcreteModel) -> pyo.Expression:
        return pyo.summation(m.cost_ij, m.x_ij)

    # Constraints

    @model.Constraint(model.nodes)  # type: ignore[misc]
    def node_is_entered_once(m: pyo.ConcreteModel, j: str) -> Any:
        """Each node j must be visited from exactly one other node."""
        return sum(m.x_ij[i, j] for i in m.nodes if i != j) == 1

    @model.Constraint(model.nodes)  # type: ignore[misc]
    def node_is_exited_once(m: pyo.ConcreteModel, i: str) -> Any:
        """Each node i must depart to exactly one other node."""
        return sum(m.x_ij[i, j] for j in m.nodes if j != i) == 1

    # Subtour elimination

    @model.Constraint(model.nodes_except_startend, model.nodes_except_startend)  # type: ignore[misc]
    def path_is_single_tour(
        m: pyo.ConcreteModel, i: str, j: str
    ) -> pyo.Constraint.Skip | Any:
        """For each pair of permanent nodes (i, j), if site j is visited from node i, the rank of j must be strictly greater than the rank of i."""
        if i == j:  # if node coincide, skip creating a constraint
            return pyo.Constraint.Skip

        r_i = m.rank_i[i]
        r_j = m.rank_i[j]
        x_ij = m.x_ij[i, j]
        # Standard MTZ form: r_j >= r_i + 1 - M * (1 - x_ij), with M = |S*|
        return r_j >= r_i + 1 - m.M * (1 - x_ij)

    return model
