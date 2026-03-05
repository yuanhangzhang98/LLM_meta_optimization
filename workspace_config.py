"""Workspace configuration for multi-run support.

This module centralizes all workspace-dependent path management.
Call set_workspace() once at startup to enable workspace prefixing.

When workspace is set (e.g., "run_22"):
- database/ becomes database_run_22/
- solvers/ becomes solvers_run_22/
- etc.

When workspace is None (default):
- All paths remain unchanged for backwards compatibility.
"""

from pathlib import Path
from typing import Optional

# Project root (computed once)
ROOT = Path(__file__).resolve().parent

# Global workspace name (None = default paths, no suffix)
_workspace: Optional[str] = None


def set_workspace(name: Optional[str]) -> None:
    """Set the workspace name for all path resolution.

    Should be called once at application startup, before any paths are used.

    Args:
        name: Workspace name (e.g., "run_22") or None for default paths.
              When set, output directories get this as a suffix.
    """
    global _workspace
    _workspace = name


def get_workspace() -> Optional[str]:
    """Get the current workspace name.

    Returns:
        Current workspace name or None if using default paths.
    """
    return _workspace


def _suffixed_dir(base_name: str) -> Path:
    """Get a directory path with optional workspace suffix.

    Args:
        base_name: Base directory name (e.g., "database", "solvers")

    Returns:
        Path with workspace suffix if workspace is set, otherwise original path.
    """
    if _workspace:
        return ROOT / f"{base_name}_{_workspace}"
    return ROOT / base_name


# ============================================================================
# Workspace-DEPENDENT paths (get suffix when workspace is set)
# ============================================================================

def get_database_dir() -> Path:
    """Get the database directory path."""
    return _suffixed_dir("database")


def get_database_path() -> Path:
    """Get the database.json file path."""
    return get_database_dir() / "database.json"


def get_solvers_dir() -> Path:
    """Get the solvers directory path."""
    return _suffixed_dir("solvers")


def get_llm_log_dir() -> Path:
    """Get the LLM log directory path."""
    return _suffixed_dir("llm_log")


def get_objectives_dir() -> Path:
    """Get the objectives directory path."""
    return _suffixed_dir("objectives")


def get_schedules_dir() -> Path:
    """Get the schedules directory path."""
    return _suffixed_dir("schedules")


def get_results_dir() -> Path:
    """Get the results directory path."""
    return _suffixed_dir("results")


def get_meta_agent_dir() -> Path:
    """Get the meta-agent state directory path."""
    return _suffixed_dir("meta_agent")


# ============================================================================
# Workspace-INDEPENDENT paths (shared code/templates, no suffix)
# ============================================================================

def get_prompts_dir() -> Path:
    """Get the prompts directory (workspace-independent)."""
    return ROOT / "prompts"


def get_knowledge_dir() -> Path:
    """Get the domain knowledge directory (workspace-independent)."""
    return ROOT / "domain_knowledge"


def get_root() -> Path:
    """Get the project root directory."""
    return ROOT


__all__ = [
    "set_workspace",
    "get_workspace",
    "get_database_dir",
    "get_database_path",
    "get_solvers_dir",
    "get_llm_log_dir",
    "get_objectives_dir",
    "get_schedules_dir",
    "get_results_dir",
    "get_meta_agent_dir",
    "get_prompts_dir",
    "get_knowledge_dir",
    "get_root",
]
