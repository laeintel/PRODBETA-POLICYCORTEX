"""
Multi-Objective Optimization Platform
Patent 2: Unified AI-Driven Platform with Multi-Objective Optimization
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from backend.core.config import settings
from backend.core.exceptions import APIError
from backend.core.redis_client import redis_client
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.decomposition.aasf import AASF
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ObjectiveType(str, Enum):
    """Types of objectives to optimize"""

    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_SECURITY = "maximize_security"
    MINIMIZE_COMPLIANCE_RISK = "minimize_compliance_risk"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_COMPLEXITY = "minimize_complexity"
    MAXIMIZE_AVAILABILITY = "maximize_availability"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_SCALABILITY = "maximize_scalability"


class ConstraintType(str, Enum):
    """Types of constraints"""

    BUDGET = "budget"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    RESOURCE = "resource"


@dataclass
class Objective:
    """Represents an optimization objective"""

    name: str
    type: ObjectiveType
    weight: float = 1.0
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """Represents an optimization constraint"""

    name: str
    type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    current_value: Optional[float] = None
    is_hard: bool = True
    penalty_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Represents optimization results"""

    solution_id: str
    objectives: List[Objective]
    constraints: List[Constraint]
    pareto_front: np.ndarray
    selected_solution: Dict[str, Any]
    all_solutions: List[Dict[str, Any]]
    convergence_history: List[float]
    optimization_time: float
    metadata: Dict[str, Any]


class GovernanceProblem(Problem):
    """Multi-objective governance optimization problem"""

    def __init__(
        self,
        objectives: List[Objective],
        constraints: List[Constraint],
        decision_variables: Dict[str, Any],
        evaluator,
    ):
        n_var = len(decision_variables)
        n_obj = len(objectives)
        n_constr = len([c for c in constraints if c.is_hard])

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)

        self.objectives = objectives
        self.constraints = constraints
        self.decision_variables = decision_variables
        self.evaluator = evaluator

        # Set bounds for decision variables
        self.xl = np.array([var["min"] for var in decision_variables.values()])
        self.xu = np.array([var["max"] for var in decision_variables.values()])

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives and constraints"""
        # Evaluate objectives
        obj_values = []
        for i, obj in enumerate(self.objectives):
            value = self.evaluator.evaluate_objective(x, obj)
            if obj.type.value.startswith("minimize"):
                obj_values.append(value)
            else:  # maximize -> minimize negative
                obj_values.append(-value)

        out["F"] = np.array(obj_values)

        # Evaluate constraints
        if self.n_constr > 0:
            constr_values = []
            for constr in self.constraints:
                if constr.is_hard:
                    value = self.evaluator.evaluate_constraint(x, constr)
                    constr_values.append(value)
            out["G"] = np.array(constr_values)


class ObjectiveEvaluator:
    """Evaluates objectives and constraints"""

    def __init__(self, resource_analyzer, compliance_checker, performance_monitor):
        self.resource_analyzer = resource_analyzer
        self.compliance_checker = compliance_checker
        self.performance_monitor = performance_monitor
        self.cache = {}

    async def evaluate_objective(self, x: np.ndarray, objective: Objective) -> float:
        """Evaluate a single objective"""
        cache_key = f"{hash(x.tobytes())}_{objective.name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        value = 0.0

        if objective.type == ObjectiveType.MINIMIZE_COST:
            value = await self._evaluate_cost(x)
        elif objective.type == ObjectiveType.MAXIMIZE_SECURITY:
            value = await self._evaluate_security(x)
        elif objective.type == ObjectiveType.MINIMIZE_COMPLIANCE_RISK:
            value = await self._evaluate_compliance_risk(x)
        elif objective.type == ObjectiveType.MAXIMIZE_PERFORMANCE:
            value = await self._evaluate_performance(x)
        elif objective.type == ObjectiveType.MINIMIZE_COMPLEXITY:
            value = await self._evaluate_complexity(x)
        elif objective.type == ObjectiveType.MAXIMIZE_AVAILABILITY:
            value = await self._evaluate_availability(x)
        elif objective.type == ObjectiveType.MINIMIZE_LATENCY:
            value = await self._evaluate_latency(x)
        elif objective.type == ObjectiveType.MAXIMIZE_SCALABILITY:
            value = await self._evaluate_scalability(x)

        self.cache[cache_key] = value
        return value

    async def evaluate_constraint(self, x: np.ndarray, constraint: Constraint) -> float:
        """Evaluate a single constraint (returns violation amount)"""
        current_value = await self._get_constraint_value(x, constraint)

        violation = 0.0
        if constraint.min_value is not None and current_value < constraint.min_value:
            violation = constraint.min_value - current_value
        elif constraint.max_value is not None and current_value > constraint.max_value:
            violation = current_value - constraint.max_value

        return violation * constraint.penalty_weight

    async def _evaluate_cost(self, x: np.ndarray) -> float:
        """Evaluate cost objective"""
        # Implement cost calculation based on decision variables
        resource_costs = await self.resource_analyzer.calculate_costs(x)
        operational_costs = await self.resource_analyzer.calculate_operational_costs(x)
        return resource_costs + operational_costs

    async def _evaluate_security(self, x: np.ndarray) -> float:
        """Evaluate security objective"""
        # Implement security score calculation
        security_score = await self.compliance_checker.calculate_security_score(x)
        return security_score

    async def _evaluate_compliance_risk(self, x: np.ndarray) -> float:
        """Evaluate compliance risk objective"""
        risk_score = await self.compliance_checker.calculate_risk_score(x)
        return risk_score

    async def _evaluate_performance(self, x: np.ndarray) -> float:
        """Evaluate performance objective"""
        perf_metrics = await self.performance_monitor.get_performance_metrics(x)
        return perf_metrics["overall_score"]

    async def _evaluate_complexity(self, x: np.ndarray) -> float:
        """Evaluate complexity objective"""
        # Implement complexity calculation
        complexity_score = np.sum(x * np.log(x + 1))  # Example complexity metric
        return complexity_score

    async def _evaluate_availability(self, x: np.ndarray) -> float:
        """Evaluate availability objective"""
        availability = await self.performance_monitor.calculate_availability(x)
        return availability

    async def _evaluate_latency(self, x: np.ndarray) -> float:
        """Evaluate latency objective"""
        latency = await self.performance_monitor.calculate_latency(x)
        return latency

    async def _evaluate_scalability(self, x: np.ndarray) -> float:
        """Evaluate scalability objective"""
        scalability_score = await self.resource_analyzer.calculate_scalability(x)
        return scalability_score

    async def _get_constraint_value(self, x: np.ndarray, constraint: Constraint) -> float:
        """Get current value for a constraint"""
        if constraint.type == ConstraintType.BUDGET:
            return await self._evaluate_cost(x)
        elif constraint.type == ConstraintType.COMPLIANCE:
            return await self._evaluate_compliance_risk(x)
        elif constraint.type == ConstraintType.PERFORMANCE:
            return await self._evaluate_performance(x)
        elif constraint.type == ConstraintType.SECURITY:
            return await self._evaluate_security(x)
        elif constraint.type == ConstraintType.AVAILABILITY:
            return await self._evaluate_availability(x)
        elif constraint.type == ConstraintType.RESOURCE:
            return await self.resource_analyzer.get_resource_usage(x)

        return 0.0


class ParetoOptimalSelector:
    """Selects optimal solution from Pareto front"""

    def __init__(self):
        self.selection_methods = {
            "weighted_sum": self._weighted_sum_selection,
            "reference_point": self._reference_point_selection,
            "fuzzy_logic": self._fuzzy_logic_selection,
            "topsis": self._topsis_selection,
            "compromise": self._compromise_programming,
        }

    async def select_solution(
        self, pareto_front: np.ndarray, objectives: List[Objective], method: str = "weighted_sum"
    ) -> Tuple[int, Dict[str, float]]:
        """Select optimal solution from Pareto front"""

        if method not in self.selection_methods:
            method = "weighted_sum"

        selection_func = self.selection_methods[method]
        selected_idx, scores = await selection_func(pareto_front, objectives)

        return selected_idx, scores

    async def _weighted_sum_selection(
        self, pareto_front: np.ndarray, objectives: List[Objective]
    ) -> Tuple[int, Dict[str, float]]:
        """Select using weighted sum method"""
        weights = np.array([obj.weight * obj.importance for obj in objectives])
        weights = weights / np.sum(weights)

        # Normalize objectives
        normalized = self._normalize_objectives(pareto_front)

        # Calculate weighted scores
        scores = np.dot(normalized, weights)

        # Select best (minimum for minimization objectives)
        selected_idx = np.argmin(scores)

        return selected_idx, {"weighted_score": float(scores[selected_idx])}

    async def _reference_point_selection(
        self, pareto_front: np.ndarray, objectives: List[Objective]
    ) -> Tuple[int, Dict[str, float]]:
        """Select using reference point method"""
        # Define reference point (ideal values)
        ref_point = []
        for i, obj in enumerate(objectives):
            if obj.target_value is not None:
                ref_point.append(obj.target_value)
            else:
                # Use best value in Pareto front
                ref_point.append(np.min(pareto_front[:, i]))

        ref_point = np.array(ref_point)

        # Calculate distances to reference point
        distances = np.linalg.norm(pareto_front - ref_point, axis=1)

        # Select closest to reference point
        selected_idx = np.argmin(distances)

        return selected_idx, {"distance_to_reference": float(distances[selected_idx])}

    async def _fuzzy_logic_selection(
        self, pareto_front: np.ndarray, objectives: List[Objective]
    ) -> Tuple[int, Dict[str, float]]:
        """Select using fuzzy logic"""
        # Implement fuzzy membership functions
        membership_scores = []

        for solution in pareto_front:
            score = 0.0
            for i, obj in enumerate(objectives):
                # Triangular membership function
                if obj.target_value is not None:
                    diff = abs(solution[i] - obj.target_value)
                    membership = max(0, 1 - diff / (obj.target_value * 0.2))
                else:
                    # Use min-max normalization
                    min_val = np.min(pareto_front[:, i])
                    max_val = np.max(pareto_front[:, i])
                    if max_val > min_val:
                        membership = 1 - (solution[i] - min_val) / (max_val - min_val)
                    else:
                        membership = 1.0

                score += membership * obj.weight * obj.importance

            membership_scores.append(score)

        membership_scores = np.array(membership_scores)
        selected_idx = np.argmax(membership_scores)

        return selected_idx, {"fuzzy_score": float(membership_scores[selected_idx])}

    async def _topsis_selection(
        self, pareto_front: np.ndarray, objectives: List[Objective]
    ) -> Tuple[int, Dict[str, float]]:
        """TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)"""
        # Normalize decision matrix
        normalized = self._normalize_objectives(pareto_front)

        # Apply weights
        weights = np.array([obj.weight * obj.importance for obj in objectives])
        weights = weights / np.sum(weights)
        weighted = normalized * weights

        # Determine ideal and anti-ideal solutions
        ideal = np.min(weighted, axis=0)  # For minimization
        anti_ideal = np.max(weighted, axis=0)

        # Calculate distances
        dist_to_ideal = np.linalg.norm(weighted - ideal, axis=1)
        dist_to_anti_ideal = np.linalg.norm(weighted - anti_ideal, axis=1)

        # Calculate relative closeness
        closeness = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal + 1e-10)

        # Select best
        selected_idx = np.argmax(closeness)

        return selected_idx, {"topsis_score": float(closeness[selected_idx])}

    async def _compromise_programming(
        self, pareto_front: np.ndarray, objectives: List[Objective]
    ) -> Tuple[int, Dict[str, float]]:
        """Compromise programming method"""
        # Normalize objectives
        normalized = self._normalize_objectives(pareto_front)

        # Apply weights
        weights = np.array([obj.weight * obj.importance for obj in objectives])
        weights = weights / np.sum(weights)

        # Calculate Lp metric (using L2 norm)
        ideal = np.zeros(len(objectives))  # Ideal is 0 for normalized minimization
        distances = np.sum((weights * (normalized - ideal)) ** 2, axis=1) ** 0.5

        # Select minimum distance
        selected_idx = np.argmin(distances)

        return selected_idx, {"compromise_distance": float(distances[selected_idx])}

    def _normalize_objectives(self, pareto_front: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1] range"""
        normalized = np.zeros_like(pareto_front)

        for i in range(pareto_front.shape[1]):
            min_val = np.min(pareto_front[:, i])
            max_val = np.max(pareto_front[:, i])

            if max_val > min_val:
                normalized[:, i] = (pareto_front[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.0

        return normalized


class MultiObjectiveOptimizer:
    """Main multi-objective optimization service"""

    def __init__(self):
        self.evaluator = None
        self.selector = ParetoOptimalSelector()
        self.optimization_history = defaultdict(list)
        self._initialized = False

    async def initialize(self, resource_analyzer, compliance_checker, performance_monitor):
        """Initialize the optimizer"""
        self.evaluator = ObjectiveEvaluator(
            resource_analyzer, compliance_checker, performance_monitor
        )
        self._initialized = True
        logger.info("Multi-objective optimizer initialized")

    async def optimize(
        self,
        objectives: List[Objective],
        constraints: List[Constraint],
        decision_variables: Dict[str, Any],
        algorithm: str = "nsga2",
        selection_method: str = "weighted_sum",
        max_generations: int = 100,
        population_size: int = 100,
    ) -> OptimizationResult:
        """Perform multi-objective optimization"""

        if not self._initialized:
            raise APIError("Optimizer not initialized", status_code=500)

        start_time = datetime.now()
        solution_id = f"opt_{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create optimization problem
            problem = GovernanceProblem(
                objectives=objectives,
                constraints=constraints,
                decision_variables=decision_variables,
                evaluator=self.evaluator,
            )

            # Select algorithm
            if algorithm == "nsga3":
                # Create reference directions for NSGA-III
                from pymoo.factory import get_reference_directions

                ref_dirs = get_reference_directions("das-dennis", len(objectives), n_partitions=12)
                optimizer = NSGA3(pop_size=population_size, ref_dirs=ref_dirs)
            else:  # Default to NSGA-II
                optimizer = NSGA2(pop_size=population_size)

            # Set termination criteria
            termination = get_termination("n_gen", max_generations)

            # Run optimization
            res = minimize(problem, optimizer, termination, seed=1, save_history=True, verbose=True)

            # Extract Pareto front
            pareto_front = res.F
            pareto_set = res.X

            # Select optimal solution
            selected_idx, selection_scores = await self.selector.select_solution(
                pareto_front, objectives, selection_method
            )

            # Prepare results
            selected_solution = {
                "decision_variables": pareto_set[selected_idx].tolist(),
                "objective_values": pareto_front[selected_idx].tolist(),
                "selection_scores": selection_scores,
            }

            # Prepare all solutions
            all_solutions = []
            for i in range(len(pareto_front)):
                sol = {
                    "index": i,
                    "decision_variables": pareto_set[i].tolist(),
                    "objective_values": pareto_front[i].tolist(),
                    "is_selected": i == selected_idx,
                }
                all_solutions.append(sol)

            # Extract convergence history
            convergence_history = [gen.opt.get("F").mean() for gen in res.history]

            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = OptimizationResult(
                solution_id=solution_id,
                objectives=objectives,
                constraints=constraints,
                pareto_front=pareto_front,
                selected_solution=selected_solution,
                all_solutions=all_solutions,
                convergence_history=convergence_history,
                optimization_time=optimization_time,
                metadata={
                    "algorithm": algorithm,
                    "selection_method": selection_method,
                    "generations": max_generations,
                    "population_size": population_size,
                    "n_solutions": len(pareto_front),
                },
            )

            # Store in history
            self.optimization_history[solution_id] = result

            # Cache result
            await self._cache_result(result)

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise APIError(f"Optimization failed: {str(e)}", status_code=500)

    async def get_optimization_history(self, limit: int = 10) -> List[OptimizationResult]:
        """Get optimization history"""
        history = list(self.optimization_history.values())
        history.sort(key=lambda x: x.solution_id, reverse=True)
        return history[:limit]

    async def apply_solution(self, solution_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Apply an optimization solution"""
        if solution_id not in self.optimization_history:
            raise APIError(f"Solution {solution_id} not found", status_code=404)

        result = self.optimization_history[solution_id]
        solution = result.selected_solution

        # Prepare configuration changes
        changes = await self._prepare_configuration_changes(
            solution["decision_variables"], result.objectives, result.constraints
        )

        if dry_run:
            return {
                "solution_id": solution_id,
                "dry_run": True,
                "proposed_changes": changes,
                "expected_improvements": await self._calculate_improvements(solution, result),
            }
        else:
            # Apply changes
            applied_changes = await self._apply_configuration_changes(changes)

            return {
                "solution_id": solution_id,
                "dry_run": False,
                "applied_changes": applied_changes,
                "timestamp": datetime.now().isoformat(),
            }

    async def _prepare_configuration_changes(
        self,
        decision_variables: List[float],
        objectives: List[Objective],
        constraints: List[Constraint],
    ) -> Dict[str, Any]:
        """Prepare configuration changes based on solution"""
        changes = {
            "resource_allocations": {},
            "policy_updates": {},
            "scaling_decisions": {},
            "security_configurations": {},
        }

        # Map decision variables to actual changes
        # This is highly domain-specific and would need customization

        return changes

    async def _apply_configuration_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration changes to the system"""
        applied = {}

        # Apply each type of change
        for change_type, change_details in changes.items():
            if change_details:
                # Implement actual application logic
                applied[change_type] = change_details

        return applied

    async def _calculate_improvements(
        self, solution: Dict[str, Any], result: OptimizationResult
    ) -> Dict[str, float]:
        """Calculate expected improvements from solution"""
        improvements = {}

        for i, obj in enumerate(result.objectives):
            current = obj.current_value or 0
            optimized = solution["objective_values"][i]

            if obj.type.value.startswith("minimize"):
                improvement = (current - optimized) / current * 100 if current > 0 else 0
            else:  # maximize
                improvement = (optimized - current) / current * 100 if current > 0 else 0

            improvements[obj.name] = improvement

        return improvements

    async def _cache_result(self, result: OptimizationResult):
        """Cache optimization result"""
        cache_key = f"optimization:{result.solution_id}"
        cache_data = {
            "solution_id": result.solution_id,
            "selected_solution": result.selected_solution,
            "metadata": result.metadata,
            "timestamp": datetime.now().isoformat(),
        }

        await redis_client.setex(cache_key, timedelta(hours=24), json.dumps(cache_data))


# Global instance
multi_objective_optimizer = MultiObjectiveOptimizer()
