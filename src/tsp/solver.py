import math
import time
from typing import NotRequired, TypedDict

import pyomo.environ as pyo
import pyomo.version
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.results import Results


class SolverInfo(TypedDict):
    """Typed dictionary for solver information."""

    name: str
    available: bool
    pyomo_version: str
    version: NotRequired[str]


class SolveResult(TypedDict):
    """Represents the result of a solving process.

    Attributes:
        optimal (bool): Indicates whether the solution is optimal.
        objective_value (float | None): The objective value of the solution,
            if available. None if no objective value is computed.
        wall_time (float): The time elapsed during the solving process, in seconds.
        solver_info (SolverInfo): Additional information about the solver used.
    """

    optimal: bool
    objective_value: float | None
    wall_time: float
    solver_info: SolverInfo


class Solver:
    """TSP Solver class for solving Pyomo ConcreteModel using different solvers."""

    def __init__(self, model: pyo.ConcreteModel, solver_name: str) -> None:
        """Initialize the TSP Solver.

        Args:
            model: The Pyomo ConcreteModel representing the TSP problem.
            solver_name: Name of the solver to use. Can be "highs" or "scip".

        Raises:
            ValueError: If solver_name is not "highs" or "scip".
        """
        if solver_name not in ("highs", "scip"):
            raise ValueError("solver_name must be 'highs' or 'scip'")

        self.model = model
        self.solver_name = solver_name
        self.solver = self._create_solver()
        self.result: Results | None = None

    def _create_solver(self) -> SolverBase:
        """Create and return the appropriate solver instance."""
        if self.solver_name == "highs":
            return SolverFactory("highs")
        else:
            return pyo.SolverFactory("scip")

    def is_solver_available(self) -> bool:
        """Check if the solver is available.

        Returns:
            True if the solver is available, False otherwise.
        """
        return bool(self.solver.available())

    def get_solver_info(self) -> SolverInfo:
        """Get solver information including name, availability, and version.

        Returns:
            Dictionary containing solver information.
        """
        solver_available = self.is_solver_available()
        info: SolverInfo = {
            "name": self.solver.name,
            "available": solver_available,
            "pyomo_version": pyomo.version.__version__,
        }

        if solver_available:
            info["version"] = self.solver.version()

        return info

    def print_model_info(self) -> None:
        """Print model information including name, variables, and constraints."""
        print(
            f"Name: {self.model.name}",
            f"Num variables: {self.model.nvariables()}",
            f"Num constraints: {self.model.nconstraints()}",
            sep="\n- ",
        )

    def solve(self) -> SolveResult:
        """Solve the TSP model and return results.

        Returns:
            Dictionary containing solution results including:
            - optimal: Whether optimal solution was found
            - objective_value: The objective function value
            - wall_time: Time taken to solve (in seconds)
            - solver_info: Information about the solver used

        Raises:
            RuntimeError: If the solver is not available.
        """
        if not self.is_solver_available():
            raise RuntimeError(f"Solver '{self.solver_name}' is not available")

        # Solve the model with timing
        start_time = time.perf_counter()
        self.result = self.solver.solve(self.model)
        wall_time = time.perf_counter() - start_time

        # Extract results
        optimal = pyo.check_optimal_termination(self.result)
        objective_value = pyo.value(self.model.total_cost) if optimal else None

        return {
            "optimal": optimal,
            "objective_value": objective_value,
            "wall_time": wall_time,
            "solver_info": self.get_solver_info(),
        }

    def extract_solution_as_arcs(self) -> list[tuple[str, str]]:
        """Extract a list of active (selected) arcs from the solved model."""
        return [
            (i, j)
            for i, j in self.model.valid_arcs
            if math.isclose(self.model.x_ij[i, j].value, 1.0)
        ]

    def print_results(self, results: SolveResult) -> None:
        """Print the solution results in a formatted manner.

        Args:
            results: Dictionary containing solution results from solve() method.
        """
        print(f"Optimal solution found: {results['optimal']}")
        if results["objective_value"] is not None:
            print(f"Objective value: {results['objective_value']:.0f}")
        print(f"Solution time (measured): {results['wall_time']:.1f} s")

        # Display timing info for highs solver
        if (
            self.solver_name == "highs"
            and self.result is not None
            and hasattr(self.result, "timing_info")
        ):
            self.result.timing_info.display()
