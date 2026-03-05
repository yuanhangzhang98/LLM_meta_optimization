"""
Utilities for writing and validating solver files.
"""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from problem_config import get_problem_config
import workspace_config


def validate_solver_code(code: str) -> Tuple[bool, List[str]]:
    """
    Validate solver code contains ONLY the required components.

    Args:
        code: Python source code string (should contain only required components + minimal imports)

    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = get_problem_config()
    errors = []

    # Syntax check
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors

    # Required components check (from config)
    required_components = {
        comp.name: comp.description
        for comp in config.solver_components
        if comp.required
    }

    for component, description in required_components.items():
        if component not in code:
            errors.append(f"Missing required component: {component} ({description})")

    # Check for required imports (from config)
    import_patterns = config.get_import_patterns()
    import_found = any(pattern in code for pattern in import_patterns)
    if not import_found:
        import_list = ", ".join(config.required_imports)
        errors.append(f"Solver code should import one of: {import_list}")

    # AST validation for components
    try:
        tree = ast.parse(code)

        # Get defined variable names
        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)

        # Get defined function names
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        # Validate each component based on its type
        for comp in config.solver_components:
            if not comp.required:
                continue

            if comp.type == "dict":
                if comp.name not in defined_names:
                    errors.append(f"{comp.name} must be defined as a variable")
            elif comp.type == "function":
                if comp.name not in function_names:
                    errors.append(f"{comp.name} must be defined as a function")

    except Exception as e:
        errors.append(f"AST parsing error: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors


def write_solver_file(
    solver_code: str,
    solver_id: int,
    solvers_dir: Path = None,
    validate: bool = True,
    overwrite: bool = False
) -> Path:
    """
    Write solver code to a file with validation.

    Args:
        solver_code: Complete Python source code for the solver
        solver_id: Experiment ID for the solver
        solvers_dir: Directory to write solver files (default: ./solvers)
        validate: Whether to validate code before writing (default: True)
        overwrite: Whether to overwrite existing file (default: False)

    Returns:
        Path to the written solver file

    Raises:
        ValueError: If validation fails or file exists and overwrite=False
    """
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    solvers_dir = Path(solvers_dir)
    solvers_dir.mkdir(parents=True, exist_ok=True)

    solver_path = solvers_dir / f"solver_{solver_id}.py"

    # Check if file already exists
    if solver_path.exists() and not overwrite:
        raise ValueError(
            f"Solver file already exists: {solver_path}. "
            f"Use overwrite=True to replace it."
        )

    # Validate code if requested
    if validate:
        is_valid, errors = validate_solver_code(solver_code)
        if not is_valid:
            error_msg = "Solver code validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    # Write to file
    solver_path.write_text(solver_code, encoding='utf-8')

    return solver_path


def load_solver_module(solver_id: int, solvers_dir: Path = None):
    """
    Dynamically load a solver module.

    Args:
        solver_id: Experiment ID for the solver
        solvers_dir: Directory containing solver files (default: ./solvers)

    Returns:
        Imported solver module

    Raises:
        ImportError: If solver file doesn't exist or import fails
    """
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    solvers_dir = Path(solvers_dir)
    solver_path = solvers_dir / f"solver_{solver_id}.py"

    if not solver_path.exists():
        raise ImportError(f"Solver file not found: {solver_path}")

    # Load module directly from file path (avoids module name conflicts with workspace)
    full_module_name = f"_loaded_solvers.solver_{solver_id}"

    # Always reload to support dynamic updates
    spec = importlib.util.spec_from_file_location(full_module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load solver module from {solver_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)

    return module


def read_solver_code(solver_id: int, solvers_dir: Path = None) -> str:
    """
    Read solver code from file.

    Args:
        solver_id: Experiment ID for the solver
        solvers_dir: Directory containing solver files (default: ./solvers)

    Returns:
        Solver source code as string
    """
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    solvers_dir = Path(solvers_dir)
    solver_path = solvers_dir / f"solver_{solver_id}.py"

    if not solver_path.exists():
        raise FileNotFoundError(f"Solver file not found: {solver_path}")

    return solver_path.read_text(encoding='utf-8')


def extract_solver_id_from_path(solver_path: str) -> Optional[int]:
    """
    Extract solver ID from a solver file path.

    Args:
        solver_path: Path to solver file (e.g., "solvers/solver_42.py")

    Returns:
        Solver ID as integer, or None if not a valid solver path
    """
    path = Path(solver_path)

    # Check if filename matches pattern solver_N.py
    if not path.stem.startswith("solver_"):
        return None

    try:
        solver_id = int(path.stem.split("_")[1])
        return solver_id
    except (IndexError, ValueError):
        return None
