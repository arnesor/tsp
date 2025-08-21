# TSP Project Development Guidelines

This document provides essential development information for the Traveling Salesman Problem (TSP) project. This is intended for advanced developers working on this specific project.

## Build/Configuration Instructions

### Package Management
The project uses **uv** as the package manager (not pip/conda). This is evident from:
- `uv.lock` file in the root directory
- `uv_build` backend in pyproject.toml build-system

### Project Setup
```bash
# Install dependencies (including dev dependencies)
uv sync --group dev

# Install the project in editable mode
uv pip install -e .
```

### Python Version Requirements
- **Minimum**: Python 3.11
- The project uses modern Python features including StrEnum and advanced type hints

### Key Dependencies
- **Optimization**: `pyomo`, `pyscipopt`, `highspy` (multiple solver backends)
- **Data Processing**: `pandas`, `numpy`, `scipy`
- **Geospatial**: `geopy`, `pyproj` (coordinate transformations)
- **Visualization**: `folium`, `matplotlib`, `networkx`
- **Validation**: `pandera` (DataFrame schema validation)

## Testing Information

### Testing Framework
- **Framework**: pytest (configured in pyproject.toml)
- **Location**: `tests/` directory
- **Configuration**: Uses `conftest.py` for shared fixtures

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file with verbose output
uv run pytest tests/test_cost_matrix.py -v

# Run tests with coverage (if coverage is configured)
uv run pytest --cov=src
```

### Test Structure Patterns
- Tests are organized in classes (e.g., `TestEuclideanCalculator`)
- Follow Arrange-Act-Assert pattern
- Use descriptive test names ending with context (e.g., `_happy_path`, `_edge_case`)
- Fixtures in `conftest.py` provide reusable test data

### Adding New Tests
1. Create test files with `test_*.py` naming convention
2. Organize related tests in classes with `Test*` naming
3. Use fixtures for shared test data
4. Follow existing naming patterns for consistency

### Example Test Pattern
```python
class TestMyFeature:
    def test_my_function_happy_path(self):
        # Arrange: Set up test data
        input_data = create_test_data()

        # Act: Execute the function
        result = my_function(input_data)

        # Assert: Verify results
        assert result.expected_property == expected_value
```

## Code Quality & Development Guidelines

### Code Style Enforcement
The project uses a comprehensive code quality stack:

#### Linting & Formatting
- **Ruff**: Primary linter with extensive rule set including:
  - `ANN`: Type annotation checks
  - `D`: Documentation requirements (Google style)
  - `B`: Security/bugbear warnings
  - `F`, `E`: Standard Python checks
- **Black**: Code formatting
- **MyPy**: Strict type checking enabled

#### Pre-commit Hooks
Pre-commit hooks are configured (`.pre-commit-config.yaml`) and run:
- File validation (large files, merge conflicts)
- Text formatting (trailing whitespace, line endings)
- Format validation (YAML, JSON, TOML)
- Python syntax checking
- Ruff linting with auto-fix
- Black formatting

### Setting Up Development Environment
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Run individual tools
uv run ruff check --fix
uv run black .
uv run mypy src/
```

### Code Patterns & Conventions

#### Type Hints
- **Strict typing** is enforced via MyPy
- Use modern type hints (Python 3.11+ features)
- Pandas integration with pandera for DataFrame validation
- Example: `Series[str]`, `DataFrame["NodeInputModel"]`

#### Documentation
- **Google-style docstrings** are required
- Comprehensive class and method documentation
- Include parameter descriptions and return types
- Example from NodeType enum:
```python
class NodeType(StrEnum):
    """Represents types of nodes or sites/controls.

    Attributes:
        PERMANENT (str): Represents a permanent node type that remains constant.
        START (str): Represents a starting node type, indicating the beginning of a tour.
    """
```

#### Data Validation
- **Pandera schemas** for DataFrame validation
- Custom validation methods using `@pa.dataframe_check`
- Support for multiple coordinate systems (lat/lon and x/y)
- Strict field validation with constraints

#### Architecture Patterns
- **Abstract Base Classes** for extensibility (e.g., `CostCalculator`)
- **Factory Pattern** for algorithm selection (`CostMatrixFactory`)
- **Strategy Pattern** for different calculation methods
- **Enum classes** for constants (e.g., `NodeType`)

### Project Structure
```
src/tsp/
├── __init__.py
├── cost_matrix.py      # Distance calculation algorithms
├── tsp_model.py        # TSP solving logic
├── node_schema.py      # Data validation schemas
├── map.py             # Visualization utilities
├── main.py            # Primary entry point
└── demo_*.py          # Demonstration scripts
```

### Configuration Files
- `pyproject.toml`: Main project configuration with tool settings
- `.pre-commit-config.yaml`: Code quality automation
- `uv.lock`: Dependency lock file (do not edit manually)

### Key Development Notes
1. **Multiple Solvers**: The project supports multiple optimization backends (SCIP, HiGHS, Pyomo)
2. **Geospatial Support**: Handles both Cartesian (x,y) and geographic (lat,lon) coordinates
3. **Data Validation**: Uses pandera for runtime DataFrame schema validation
4. **Modern Python**: Takes advantage of Python 3.11+ features throughout

### Debugging Tips
- Use type checking (`uv run mypy src/`) to catch issues early
- Leverage pandera validation errors for data issues
- The project has comprehensive test coverage for edge cases
- Use the demo scripts to understand algorithm behavior

---
*Generated: 2025-08-17*
