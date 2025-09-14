import pandas as pd
import pyomo.environ as pyo
import pytest

from tsp.tsp_model import create_tsp_model


def _simple_cost_matrix(names: list[str]) -> pd.DataFrame:
    # Create a simple symmetric cost matrix with zero diagonal
    n = len(names)
    data: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        row.extend(0.0 if i == j else float(i + j + 1) for j in range(n))
        data.append(row)
    return pd.DataFrame(data, index=names, columns=names)


class TestTspModel:
    def test_model_structure_sets_and_params(self) -> None:
        # Arrange
        names = ["A", "B", "C"]
        cm = _simple_cost_matrix(names)

        # Act
        model = create_tsp_model(cm, startend_name="A")

        # Assert basic sets
        assert set(model.nodes.data()) == set(names)
        assert len(model.valid_arcs) == len(names) * (len(names) - 1)

        # cost_ij should be defined for all valid arcs
        assert len(model.cost_ij) == len(model.valid_arcs)

        # No self loops in valid_arcs
        for i, j in model.valid_arcs:
            assert i != j

    def test_nodes_except_startend_and_rank_bounds(self) -> None:
        # Arrange
        names = ["A", "B", "C", "D"]
        cm = _simple_cost_matrix(names)

        # Act
        model = create_tsp_model(cm, startend_name="A")

        # Assert: nodes_except_startend excludes the depot and includes others
        assert set(model.nodes_except_startend.data()) == {"B", "C", "D"}
        assert pyo.value(model.M) == 3

        # rank_i is only defined for nodes in S* and has bounds [1, |S*|]
        for name in model.nodes_except_startend:
            assert name in model.rank_i
            var = model.rank_i[name]
            assert var.lb == 1
            assert var.ub == 3

    def test_objective_evaluates_assignment(self) -> None:
        # Arrange
        names = ["A", "B", "C"]
        cm = _simple_cost_matrix(names)
        model = create_tsp_model(cm, startend_name="A")

        # Pick two arcs to set to 1, all others 0
        chosen = [("A", "B"), ("B", "C")]
        expected = sum(float(cm.at[i, j]) for (i, j) in chosen)

        for ij in model.valid_arcs:
            model.x_ij[ij].set_value(1 if ij in chosen else 0)

        # Assert
        assert pytest.approx(pyo.value(model.total_cost.expr), rel=1e-9) == expected

    def test_degree_constraints_satisfied_by_cycle(self) -> None:
        # Arrange: 3-node cycle A->B->C->A
        names = ["A", "B", "C"]
        cm = _simple_cost_matrix(names)
        model = create_tsp_model(cm, startend_name="A")

        # Assignment for a simple tour
        tour = {("A", "B"), ("B", "C"), ("C", "A")}
        for ij in model.valid_arcs:
            model.x_ij[ij].set_value(1 if ij in tour else 0)

        # Assert: each node has in-degree 1 and out-degree 1
        for j in model.nodes:
            body_enter = pyo.value(sum(model.x_ij[i, j] for i in model.nodes if i != j))
            assert body_enter == 1
        for i in model.nodes:
            body_exit = pyo.value(sum(model.x_ij[i, j] for j in model.nodes if j != i))
            assert body_exit == 1

    def test_mtz_constraints_respected_with_ranks(self) -> None:
        # Arrange
        names = ["A", "B", "C"]
        cm = _simple_cost_matrix(names)
        model = create_tsp_model(cm, startend_name="A")

        # Tour: A->B->C->A (only pairs in S* x S* are (B,C) and (C,B))
        tour = {("A", "B"), ("B", "C"), ("C", "A")}
        for ij in model.valid_arcs:
            model.x_ij[ij].set_value(1 if ij in tour else 0)

        # Set ranks over S* = {B, C}
        model.rank_i["B"].set_value(1)
        model.rank_i["C"].set_value(2)

        # For x_{B,C} = 1, constraint should enforce r_C >= r_B + 1
        c_bc = model.path_is_single_tour["B", "C"]
        assert pyo.value(c_bc.body) >= 0  # body is lhs - rhs >= 0

        # For x_{C,B} = 0, constraint should be slack (non-binding)
        c_cb = model.path_is_single_tour["C", "B"]
        assert pyo.value(c_cb.body) >= 0

    def test_startend_name_assertion(self) -> None:
        # Arrange
        cm = _simple_cost_matrix(["A", "B"])  # startend not present

        # Act & Assert
        with pytest.raises(AssertionError, match="Startend node Z not found"):
            create_tsp_model(cm, startend_name="Z")

    def test_none_startend_includes_all_nodes_in_s_star(self) -> None:
        # Arrange
        names = ["A", "B", "C"]
        cm = _simple_cost_matrix(names)

        # Act
        model = create_tsp_model(cm, startend_name=None)

        # Assert: S* should equal S when no depot specified
        assert set(model.nodes_except_startend.data()) == set(names)
        assert pyo.value(model.M) == len(names)
        # rank is defined for all nodes with bounds [1, n]
        for name in names:
            var = model.rank_i[name]
            assert var.lb == 1
            assert var.ub == len(names)

    def test_valid_arcs_exclude_self_loops(self) -> None:
        # Arrange
        names = ["A", "B", "C", "D"]
        cm = _simple_cost_matrix(names)

        # Act
        model = create_tsp_model(cm, startend_name="A")

        # Assert
        for i, j in model.valid_arcs:
            assert i != j
