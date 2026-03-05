# LLM-in-the-loop Digital MemComputing Machine (DMM) Optimization

## Research Objective
Design digital MemComputing machine (DMM) dynamics that solve planted 3-SAT problems faster than the current baseline, achieving polynomial-time complexity.

## Problem Definition

**3-SAT Instance**: 
- N Boolean variables V_i ∈ {0, 1}
- M clauses C_m = L_i ∨ L_j ∨ L_k, where L_i is either V_i or its negation
- A DMM relaxes each V_i to continuous v_i ∈ [-1, 1] and adds auxiliary "memory" variables

**Goal**: Find dynamical equations where only fixed points correspond to satisfying assignments.

## Baseline DMM Equations (Reference)

The current baseline uses these dynamics:

**State Evolution**:
$$\dot{v}_n = \sum_{m=1}^M x_{l,m} x_{s,m} G_{n,m}(v_n,v_j,v_k) + (1+\zeta x_{l,m}) (1-x_{s,m}) R_{n,m}(v_n,v_j,v_k)$$

**Memory Variables**:
$$\dot{x}_{s,m} = \beta (x_{s,m} + \epsilon) (c_m(v_i,v_j,v_k) - \gamma)$$
$$\dot{x}_{l,m} = \alpha (c_m(v_i,v_j,v_k) - \delta)$$

**Supporting Functions**:
$$c_m = \frac{1}{2}\min[(1-q_{i,m}v_i),(1-q_{j,m}v_j),(1-q_{k,m}v_k)]$$
$$G_{n,m} = \frac{1}{2}q_{n,m}\min[(1-q_{j,m}v_j), (1-q_{k,m}v_k)]$$
$$R_{n,m} = \begin{cases}
    \frac{1}{2}q_{n,m}(1-q_{n,m}v_n), & \text{if } c_m = \frac{1}{2}(1-q_{n,m}v_n) \\
    0, & \text{otherwise}
\end{cases}$$

**Default Hyperparameters**: α=5, β=20, γ=0.25, δ=0.05, ε=10^-3, ζ=10^-3

## Dynamics Interpretation

- **State variables**: v_n ∈ [-1,1] (continuous relaxation of Boolean variables)
- **Clause monitor**: c_m ∈ [0,1] tracks clause satisfaction (satisfied when c_m < 0.5)
- **Memory variables**:
  - **Long-term weight** x_l,m ∈ [1,10^6]: Grows for persistently violated clauses
  - **Short-term switch** x_s,m ∈ [0,1]: Toggles between "push" and "hold" modes
- **Forces on v_n**:
  - **Gradient term** G_n,m: Nudges literals toward satisfaction
  - **Rigidity term** R_n,m: Prevents satisfied literals from flipping

## Code Structure

The Designer Agent would modify these core solve components in the PyTorch implementation:

1. **VARIABLES_SPEC**: Variable definitions (name → {init, shape, bounds})
2. **HYPER_SPACE**: Hyperparameter ranges (name → {type, default, low, high})  
3. **GRAD_SINGLE**: Core dynamics function _grad_single(vars, idx, sgn, hp)

**Key Constraints**:
- Fixed points must correspond to valid SAT solutions
- No data-dependent control flow (vmap limitation)
- Variables must respect their bounds

## Research Strategy

**Incremental Approach**: Make small, principled modifications to track causal effects
**Evidence-Based**: Build upon successful patterns, avoid failed approaches
**Multi-Scale**: Consider both local dynamics and global scaling behavior

The system uses a multi-agent approach:
- **Planner**: Analyzes all experiment history, identifies patterns, provides strategic guidance
- **Designer**: Designs focused experiments based on planner's recommendations
- **Objective**: Designs objective functions and experiment schedules to guide exploration
