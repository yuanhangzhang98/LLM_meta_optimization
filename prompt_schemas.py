"""Structured output models for Planner and Designer prompts.

Renamed from Supervisor/Researcher to Planner/Designer for the new architecture.
experiment_id renamed to design_id throughout.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from problem_config import get_problem_config
import objective_loader
import schedule_loader


class ReferenceWeight(BaseModel):
    """Reference to a parent design with weight (replaces parent_experiment_id)."""
    design_id: int = Field(..., ge=0)
    weight: float = Field(..., ge=0.0, le=1.0)


class SuccessLevel(str, Enum):
    poor = "poor"
    moderate = "moderate"
    good = "good"
    excellent = "excellent"


class DesignerAnalysis(BaseModel):
    """Response schema for designer_analysis_2 (post-execution analysis).

    Renamed from ResearcherAnalysis. This provides analysis of execution results.
    """

    short_name: str = Field(..., max_length=40, min_length=1)
    key_insight: str = Field(..., min_length=1)
    success_level: SuccessLevel
    detailed_analysis: str = Field(..., min_length=1)
    comparison_to_references: str = Field(..., min_length=1)
    recommended_next_steps: str = Field(..., min_length=1)
    reference_weights: List[ReferenceWeight] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_weights(cls, model: "DesignerAnalysis") -> "DesignerAnalysis":
        weights = model.reference_weights
        if not weights:
            return model

        total = sum(item.weight for item in weights)
        if total <= 0:
            raise ValueError("Sum of reference_weights must be positive.")

        normalized = [ReferenceWeight(design_id=w.design_id, weight=w.weight / total) for w in weights]
        object.__setattr__(model, "reference_weights", normalized)
        return model


class FidelityPromotionDecision(BaseModel):
    """Response schema for designer_analysis_3 (fidelity promotion decision).

    Decides whether to promote to next fidelity level or conclude experimentation.
    """

    reasoning: str = Field(
        ...,
        description="Brief explanation of the promotion decision"
    )
    promote_fidelity: bool = Field(
        ...,
        description="True to promote to next fidelity, False to conclude"
    )


class DesignerPlan(BaseModel):
    """Response schema for designer_analysis_1 (new design).

    Renamed from SolverPlan. Designer agent creates new solver designs.
    """

    explanation: str = Field(..., min_length=1)
    solver_code: str = Field(..., min_length=50)

    @field_validator("solver_code")
    @classmethod
    def validate_solver_structure(cls, code: str) -> str:
        """Validate that solver code contains ONLY the required components."""
        config = get_problem_config()
        required = config.get_required_component_names()
        missing = [item for item in required if item not in code]
        if missing:
            raise ValueError(f"Solver code missing required components: {', '.join(missing)}")

        # Basic syntax check
        try:
            compile(code, '<solver_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Solver code has syntax error at line {e.lineno}: {e.msg}")

        return code


class DesignerErrorFix(BaseModel):
    """Response schema for fixing solver code errors.

    Renamed from SolverErrorFix.
    """

    error_summary: str = Field(..., min_length=1)
    solver_code: str = Field(..., min_length=50)

    @field_validator("solver_code")
    @classmethod
    def validate_solver_structure(cls, code: str) -> str:
        """Validate that fixed solver code contains ONLY the required components."""
        config = get_problem_config()
        required = config.get_required_component_names()
        missing = [item for item in required if item not in code]
        if missing:
            raise ValueError(f"Fixed solver code missing required components: {', '.join(missing)}")

        # Basic syntax check
        try:
            compile(code, '<solver_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Fixed solver code has syntax error at line {e.lineno}: {e.msg}")

        return code


class PlannerLookup(BaseModel):
    """Response schema for planner_analysis_1 (design lookup request).

    Renamed from SupervisorLookup. experiment_id → design_id.
    """

    lookup_design_ids: List[int] = Field(default_factory=list)
    context_rationale: str = Field(..., min_length=1)

    @field_validator("lookup_design_ids")
    @classmethod
    def ensure_unique_ids(cls, ids: Sequence[int]) -> Sequence[int]:
        if len(set(ids)) != len(ids):
            raise ValueError("lookup_design_ids must be unique.")
        return ids

    @field_validator("lookup_design_ids")
    @classmethod
    def ensure_non_negative(cls, ids: Sequence[int]) -> Sequence[int]:
        for design_id in ids:
            if design_id < 0:
                raise ValueError("Design IDs must be non-negative integers.")
        return ids


class ResearchPhase(str, Enum):
    early_exploration = "early_exploration"
    systematic_search = "systematic_search"
    exploitation = "exploitation"
    stagnation = "stagnation"
    breakthrough_needed = "breakthrough_needed"


class FidelityLevel(str, Enum):
    """Fidelity level for experiment schedules.

    Maps to predefined schedules:
    - low → schedule_0 (baseline, quick evaluation)
    - medium → schedule_1 (moderate depth)
    - high → schedule_2 (thorough evaluation)
    """
    low = "low"
    medium = "medium"
    high = "high"


# Mapping from fidelity level to default schedule ID
FIDELITY_TO_SCHEDULE = {
    FidelityLevel.low: 0,
    FidelityLevel.medium: 1,
    FidelityLevel.high: 2,
}


class PlannerRoundSummary(BaseModel):
    """Response schema for planner_analysis_2 (next-round directives).

    Renamed from SupervisorRoundSummary. reference_experiment_ids → reference_design_ids.
    Extended with fidelity guidance for multi-fidelity optimization.
    """

    current_phase: ResearchPhase
    key_insights: List[str] = Field(default_factory=list)
    success_patterns: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    research_directions: List[str] = Field(default_factory=list)
    strategy_rationales: List[str] = Field(default_factory=list)
    focus_areas: List[str] = Field(default_factory=list)
    avoid_areas: List[str] = Field(default_factory=list)
    reference_design_ids: List[List[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lengths(cls, model: "PlannerRoundSummary") -> "PlannerRoundSummary":
        n_dirs = len(model.research_directions)
        related_fields = {
            "strategy_rationales": model.strategy_rationales,
            "focus_areas": model.focus_areas,
            "avoid_areas": model.avoid_areas,
            "reference_design_ids": model.reference_design_ids,
        }

        for field_name, values in related_fields.items():
            if len(values) != n_dirs:
                raise ValueError(
                    f"Field '{field_name}' must have the same length as 'research_directions' "
                    f"(expected {n_dirs}, got {len(values)})."
                )
        return model


class ObjectiveSchedulePlan(BaseModel):
    """Unified response schema for generating both objective and schedule code together.

    This replaces the separate ObjectivePlan and SchedulerPlan workflows. The agent
    generates a proxy objective function AND multi-fidelity experiment schedules that
    work together to estimate the true research goal.
    """

    objective_description: str = Field(
        ...,
        min_length=1,
        description="Clear description of what this objective measures"
    )
    objective_code: str = Field(
        ...,
        min_length=50,
        description="Complete Python code with imports and objective function definition"
    )
    schedule_description: str = Field(
        ...,
        min_length=1,
        description="Description of experiment schedules at different fidelities"
    )
    schedule_code: str = Field(
        ...,
        min_length=50,
        description="Complete Python code with imports and 3 schedule functions (low, medium, high)"
    )

    @field_validator("objective_code")
    @classmethod
    def validate_objective_structure(cls, code: str) -> str:
        """Validate that objective code compiles and contains an 'objective' function."""
        # Basic syntax check
        try:
            compile(code, '<objective_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Objective code has syntax error at line {e.lineno}: {e.msg}")

        # Check for 'objective' function definition
        if 'def objective(' not in code and 'def objective (' not in code:
            raise ValueError("Objective code must contain an 'objective' function definition")

        # Validate function signature by executing and checking
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Objective code failed to execute: {e}")

        if 'objective' not in namespace:
            raise ValueError("Objective code must define an 'objective' function")

        objective_fn = namespace['objective']
        if not objective_loader.validate_objective(objective_fn):
            raise ValueError(
                "Objective function must be callable and accept 'experiment_results' as first parameter"
            )

        return code

    @field_validator("schedule_code")
    @classmethod
    def validate_schedule_structure(cls, code: str) -> str:
        """Validate that schedule code compiles and contains 3 required schedule functions."""
        # Basic syntax check
        try:
            compile(code, '<schedule_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Schedule code has syntax error at line {e.lineno}: {e.msg}")

        # Check for all three required schedule functions
        required_functions = ['schedule_low', 'schedule_medium', 'schedule_high']
        for func_name in required_functions:
            if f'def {func_name}(' not in code and f'def {func_name} (' not in code:
                raise ValueError(f"Schedule code must contain a '{func_name}' function definition")

        # Execute code and validate the schedule functions
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Schedule code failed to execute: {e}")

        for func_name in required_functions:
            if func_name not in namespace:
                raise ValueError(f"Schedule code must define a '{func_name}' function")

            schedule_fn = namespace[func_name]
            if not schedule_loader.validate_schedule(schedule_fn):
                raise ValueError(
                    f"{func_name} function must be callable and accept 'design_id' as its first parameter"
                )

        return code


class ObjectiveErrorFix(BaseModel):
    """Response schema for fixing objective/schedule code errors.

    Used when generated objective or schedule code has runtime errors and needs correction.
    Both code fields are optional - include only the code that needs fixing.
    """

    error_summary: str = Field(
        ...,
        min_length=1,
        description="Summary of what went wrong and how it was fixed"
    )
    objective_code: str = Field(
        default="",
        description="Corrected Python code defining the objective function (blank if not needed)"
    )
    schedule_code: str = Field(
        default="",
        description="Corrected Python code with schedule_low/medium/high functions (blank if not needed)"
    )

    @field_validator("objective_code")
    @classmethod
    def validate_objective_structure(cls, code: str) -> str:
        """Validate objective code only if non-empty."""
        if not code or code.strip() == "":
            return code

        # Basic syntax check
        try:
            compile(code, '<objective_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Fixed objective code has syntax error at line {e.lineno}: {e.msg}")

        # Check for 'objective' function definition
        if 'def objective(' not in code and 'def objective (' not in code:
            raise ValueError("Fixed objective code must contain an 'objective' function definition")

        # Validate function signature by executing and checking
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Fixed objective code failed to execute: {e}")

        if 'objective' not in namespace:
            raise ValueError("Fixed objective code must define an 'objective' function")

        objective_fn = namespace['objective']
        if not objective_loader.validate_objective(objective_fn):
            raise ValueError(
                "Fixed objective function must be callable and accept 'experiment_results' as first parameter"
            )

        return code

    @field_validator("schedule_code")
    @classmethod
    def validate_schedule_structure(cls, code: str) -> str:
        """Validate schedule code only if non-empty."""
        if not code or code.strip() == "":
            return code

        # Basic syntax check
        try:
            compile(code, '<schedule_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Fixed schedule code has syntax error at line {e.lineno}: {e.msg}")

        # Check for all three schedule functions
        required_funcs = ['schedule_low', 'schedule_medium', 'schedule_high']
        for func_name in required_funcs:
            if f'def {func_name}(' not in code and f'def {func_name} (' not in code:
                raise ValueError(f"Fixed schedule code must contain '{func_name}' function definition")

        # Validate functions by executing and checking
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Fixed schedule code failed to execute: {e}")

        # Verify all functions exist and are callable
        for func_name in required_funcs:
            if func_name not in namespace:
                raise ValueError(f"Fixed schedule code must define '{func_name}' function")

            schedule_fn = namespace[func_name]
            if not schedule_loader.validate_schedule(schedule_fn):
                raise ValueError(
                    f"Fixed schedule function '{func_name}' must be callable and accept 'design_id' as first parameter"
                )

        return code


class ObjectiveOnlyPlan(BaseModel):
    """Schema for objective-only generation (no schedule).

    Used when schedule generation is disabled and only objectives should be created.
    The baseline schedule will be used for experiments.
    """

    objective_description: str = Field(
        ...,
        min_length=1,
        description="Clear description of what this objective measures"
    )
    objective_code: str = Field(
        ...,
        min_length=50,
        description="Complete Python code with imports and objective function definition"
    )

    @field_validator("objective_code")
    @classmethod
    def validate_objective_structure(cls, code: str) -> str:
        """Validate that objective code compiles and contains an 'objective' function."""
        # Basic syntax check
        try:
            compile(code, '<objective_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Objective code has syntax error at line {e.lineno}: {e.msg}")

        # Check for 'objective' function definition
        if 'def objective(' not in code and 'def objective (' not in code:
            raise ValueError("Objective code must contain an 'objective' function definition")

        # Validate function signature by executing and checking
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Objective code failed to execute: {e}")

        if 'objective' not in namespace:
            raise ValueError("Objective code must define an 'objective' function")

        objective_fn = namespace['objective']
        if not objective_loader.validate_objective(objective_fn):
            raise ValueError(
                "Objective function must be callable and accept 'experiment_results' as first parameter"
            )

        return code


class ObjectiveOnlyErrorFix(BaseModel):
    """Schema for fixing objective-only code errors.

    Used when only objective code needs fixing (schedule generation disabled).
    """

    error_summary: str = Field(
        ...,
        min_length=1,
        description="Summary of what went wrong and how it was fixed"
    )
    objective_code: str = Field(
        ...,
        min_length=50,
        description="Corrected Python code defining the objective function"
    )

    @field_validator("objective_code")
    @classmethod
    def validate_objective_structure(cls, code: str) -> str:
        """Validate objective code."""
        # Basic syntax check
        try:
            compile(code, '<objective_validation>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Fixed objective code has syntax error at line {e.lineno}: {e.msg}")

        # Check for 'objective' function definition
        if 'def objective(' not in code and 'def objective (' not in code:
            raise ValueError("Fixed objective code must contain an 'objective' function definition")

        # Validate function signature by executing and checking
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Fixed objective code failed to execute: {e}")

        if 'objective' not in namespace:
            raise ValueError("Fixed objective code must define an 'objective' function")

        objective_fn = namespace['objective']
        if not objective_loader.validate_objective(objective_fn):
            raise ValueError(
                "Fixed objective function must be callable and accept 'experiment_results' as first parameter"
            )

        return code


# --------------------------------------------------------------------------- #
# Meta-Agent Schemas                                                           #
# --------------------------------------------------------------------------- #


class MetaResearchPhase(str, Enum):
    """Research phase as assessed by the meta-agent.

    More strategic than the planner's ResearchPhase - focuses on overall
    research progress rather than iteration-level tactics.
    """
    exploring = "exploring"           # Early phase, trying diverse approaches
    converging = "converging"         # Found promising direction, narrowing focus
    stuck = "stuck"                   # Progress stalled, need new approach
    breakthrough_needed = "breakthrough_needed"  # Major innovation required
    refining = "refining"             # Good solution found, polishing


class MetaObjectiveAssessment(BaseModel):
    """Assessment of a single objective function by the meta-agent.

    The meta-agent evaluates each objective's usefulness and assigns
    a weight multiplier to adjust its influence on the consensus.
    """

    objective_id: int = Field(..., ge=0)
    assessment: str = Field(
        ...,
        min_length=1,
        description="Analysis of this objective's effectiveness"
    )
    weight_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Multiplier for this objective's consensus weight (0=disable, >1=amplify)"
    )


class MetaAgentAnalysis(BaseModel):
    """Response schema for meta-agent analysis.

    The meta-agent oversees research progress, evaluates objectives,
    and provides strategic guidance for objective generation.
    """

    research_assessment: str = Field(
        ...,
        min_length=10,
        description="Overall assessment of research progress"
    )
    research_phase: MetaResearchPhase = Field(
        ...,
        description="Current strategic research phase"
    )
    objective_analysis: List[MetaObjectiveAssessment] = Field(
        default_factory=list,
        description="Assessment and weight adjustment for each objective"
    )
    objective_directions: str = Field(
        ...,
        min_length=10,
        description="Strategic guidance for next objective generation"
    )


# Export both old and new names for backward compatibility during transition
__all__ = [
    "ReferenceWeight",
    "SuccessLevel",
    "ResearchPhase",
    "FidelityLevel",
    "FIDELITY_TO_SCHEDULE",
    # Planner/Designer schemas
    "DesignerAnalysis",
    "DesignerPlan",
    "DesignerErrorFix",
    "FidelityPromotionDecision",
    "PlannerLookup",
    "PlannerRoundSummary",
    # Objective schemas
    "ObjectiveSchedulePlan",
    "ObjectiveErrorFix",
    "ObjectiveOnlyPlan",
    "ObjectiveOnlyErrorFix",
    # Meta-agent schemas
    "MetaResearchPhase",
    "MetaObjectiveAssessment",
    "MetaAgentAnalysis",
]
