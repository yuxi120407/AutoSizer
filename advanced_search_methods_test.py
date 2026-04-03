#!/usr/bin/env python3
"""
Advanced Search Methods for OTA Optimization
============================================
Adds intelligent optimization algorithms:
1. Bayesian Optimization (Gaussian Process)
2. Genetic Algorithm
3. Simulated Annealing
4. Latin Hypercube Sampling
5. Differential Evolution
6. Adaptive Search (exploitation + exploration balance)
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import skopt
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from itertools import product
import math 




class AdvancedSearchMethods:
    """Advanced optimization algorithms for discrete design space"""
    
    def __init__(self, config, W_values: List[float], optimization_config: dict = None):
        """
        Parameters:
        -----------
        config: dict or str
            Configuration with variable definitions
        W_values: List[float]
            Full list of available discrete width values
        optimization_config: dict, optional
            Output from LLM search space reduction with:
            - 'variables_to_optimize': {var_name: [list of allowed values]}
            - 'variables_fixed': {var_name: fixed_value}
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.all_var_names = list(self.config['variable'].keys())  # All variables
        self.W_values = W_values
        self.search_history = []
        
        # Setup variable-specific search spaces
        if optimization_config:
            self.variables_to_optimize = optimization_config['variables_to_optimize']
            self.variables_fixed = optimization_config['variables_fixed']
        else:
            # Default: all variables use full W_values range
            self.variables_to_optimize = {var: W_values for var in self.all_var_names}
            self.variables_fixed = {}
        
        # Create ordered lists for efficient indexing
        self.optimizable_vars = list(self.variables_to_optimize.keys())
        self.fixed_vars = list(self.variables_fixed.keys())
        
        # n_dims is the number of variables being optimized
        self.n_dims = len(self.optimizable_vars)
        self.total_vars = len(self.all_var_names)
        
        print(f"Initialized search space:")
        print(f"  Total variables: {self.total_vars}")
        print(f"  Variables to optimize (n_dims): {self.n_dims}")
        print(f"  Fixed variables: {len(self.fixed_vars)}")
        for var in self.optimizable_vars:
            print(f"    {var}: {len(self.variables_to_optimize[var])} choices")
        for var in self.fixed_vars:
            print(f"    {var}: fixed at {self.variables_fixed[var]}")


    def _safe_sort_results(self, results, key_fn, reverse=True, filter_none=True):
        """
        Universal helper to safely sort results, filtering out None/invalid values
        Works for ANY algorithm that needs to sort by fitness/objective
        
        Parameters:
        -----------
        results: List
            List of result objects to sort
        key_fn: callable
            Function to extract sort key from each result
            Example: lambda x: x.fom
                     lambda x: self.calculate_fitness(x, targets, weights)
        reverse: bool
            If True, sort descending (maximize). If False, ascending (minimize)
        filter_none: bool
            If True, filter out results where key_fn returns None
        
        Returns:
        --------
        List: Sorted valid results (or empty list if all invalid)
        
        Example:
        --------
        # Sort by FOM:
        sorted_results = self._safe_sort_results(
            previous_results, 
            key_fn=lambda x: getattr(x, 'fom', None),
            reverse=True
        )
        
        # Sort by composite metric:
        sorted_results = self._safe_sort_results(
            previous_results,
            key_fn=lambda x: self.optimizer.llm_agent._compute_composite_metric(x.to_dict(), metric),
            reverse=True
        )
        """
        if not results:
            return []
        
        valid_results = []
        invalid_count = 0
        
        for item in results:
            try:
                key_value = key_fn(item)
                
                if filter_none and key_value is None:
                    invalid_count += 1
                    continue
                
                # Check for NaN or inf
                if isinstance(key_value, (int, float)):
                    import math
                    if math.isnan(key_value) or math.isinf(key_value):
                        invalid_count += 1
                        continue
                
                valid_results.append(item)
                
            except Exception as e:
                # Skip items that cause errors in key_fn
                invalid_count += 1
                continue
        
        # Report filtering
        if invalid_count > 0:
            print(f"  ⚠️  Filtered {invalid_count}/{len(results)} results with invalid sort keys")
        
        # Check if we have any valid results
        if not valid_results:
            print(f"  ❌ All {len(results)} results have invalid sort keys")
            return []
        
        # Sort
        try:
            sorted_results = sorted(valid_results, key=key_fn, reverse=reverse)
            return sorted_results
        except Exception as e:
            print(f"  ❌ Error during sorting: {e}")
            return valid_results  # Return unsorted if sorting fails
            

    def _create_full_point(self, optimizable_values: Tuple) -> Tuple:
        """
        Create a full parameter point by combining optimizable and fixed values.
        
        Parameters:
        -----------
        optimizable_values: Tuple
            Values for variables being optimized (in order of self.optimizable_vars)
            
        Returns:
        --------
        Tuple: Full point with all variables in order of self.all_var_names
        """
        # Create dict for easy lookup
        value_dict = {}
        
        # Add optimizable values
        for var, val in zip(self.optimizable_vars, optimizable_values):
            value_dict[var] = val
        
        # Add fixed values
        for var, val in self.variables_fixed.items():
            value_dict[var] = val

        # For any variable not in optimize or fixed, use middle value from its range
        for var in self.all_var_names:
            if var not in value_dict:
                var_values = self.variables_to_optimize.get(var, self.W_values)
                value_dict[var] = var_values[len(var_values) // 2]

        # Return in correct order
        return tuple(value_dict[var] for var in self.all_var_names)
        
    
    def update_history(self, point: Tuple, fom: float):
        """Update search history with new evaluation"""
        self.search_history.append({'point': point, 'fom': fom})
    
    def latin_hypercube_sampling(self, n_samples: int, targets=None, weights=None, seed: int = None) -> List[Tuple]:
        """
        Latin Hypercube Sampling - better space coverage than random
        Ensures samples are well-distributed across the design space
        
        Parameters:
        -----------
        n_samples: int
            Number of samples to generate
        targets: List[str], optional
            Target metrics (not used by LHS but included for interface consistency)
        weights: Dict[str, float], optional
            Weights for multiple targets (not used by LHS)
        """
        print(f"  Using Latin Hypercube Sampling for {n_samples} samples")
        print(f"  Sampling {self.n_dims} variables (out of {self.total_vars} total)")
        
        if targets:
            print(f"  Note: Targets {targets} not used by LHS algorithm")
        
        # Only sample the optimizable variables
        if self.n_dims == 0:
            print("  Warning: No variables to optimize! Returning fixed point.")
            fixed_point = tuple(self.variables_fixed[var] for var in self.all_var_names)
            return [fixed_point] * n_samples
        
        # Create LHS sampler for optimizable dimensions only (n_dims)
        actual_seed = seed if seed is not None else random.randint(0, 10000)
        print(f"  Using seed: {actual_seed}")
        sampler = qmc.LatinHypercube(d=self.n_dims, seed=actual_seed)
        
        # Generate samples in [0, 1]^n_dims
        samples = sampler.random(n=n_samples)
        
        # Map to discrete values for each optimizable variable
        points = []
        for sample in samples:
            optimizable_values = []
            for i, var in enumerate(self.optimizable_vars):
                var_range = self.variables_to_optimize[var]
                idx = int(sample[i] * len(var_range)) % len(var_range)
                optimizable_values.append(var_range[idx])
            
            # Create full point with fixed values
            full_point = self._create_full_point(tuple(optimizable_values))
            points.append(full_point)
        
        return points

    def random_sampling(self, n_samples: int, seed: int = None) -> List[Tuple]:
        """
        Pure random sampling with variable-specific ranges.
        
        Parameters:
        -----------
        n_samples: int
            Number of samples to generate
        seed: int, optional
            Random seed for reproducibility
        """
        print(f"  Using Random Sampling for {n_samples} samples")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        points = []
        for _ in range(n_samples):
            optimizable_values = []
            for var in self.optimizable_vars:
                var_range = self.variables_to_optimize[var]
                optimizable_values.append(random.choice(var_range))
            
            full_point = self._create_full_point(tuple(optimizable_values))
            points.append(full_point)
        
        return points
        

    def calculate_fitness(self, result, targets, weights=None):
        """
        Calculate fitness score for a result based on targets and weights
        
        Parameters:
        -----------
        result: OTAResult
            Design result to evaluate
        targets: List[str]
            List of optimization targets ('gain_db', 'power_uw', 'ugbw_mhz', 'fom', 'composite')
        weights: Dict[str, float]
            Optional weights for each target. If not provided, all targets have equal weight.
        
        Returns:
        --------
        float: Fitness score (higher is better)
        """
        # Default to FOM if no targets
        if not targets:
            targets = ['fom']
            
        # Special case for composite metric (like ratio)

        if 'composite' in targets:
            # Check if we have complete target_metric information
            if (hasattr(self, 'target_metric') and 
                    self.target_metric and 
                    'numerator_keys' in self.target_metric and 
                    'denominator_keys' in self.target_metric):
                
                formulation_type = self.target_metric.get('formulation_type', '')
                
                if formulation_type == 'ratio':
                    numerator_keys = self.target_metric['numerator_keys']
                    denominator_keys = self.target_metric['denominator_keys']
                    
                    # Calculate numerator (product of all numerator values)
                    numerator = 1.0
                    for key in numerator_keys:
                        numerator *= getattr(result, key, 1.0)
                    
                    # Calculate denominator (product of all denominator values)
                    denominator = 1.0
                    for key in denominator_keys:
                        denominator *= getattr(result, key, 1.0)
                    
                    # Return the ratio
                    return numerator / (denominator + 1e-10)
                
        # If single target (that's not 'composite'), return that value directly
        if len(targets) == 1:
            target = targets[0]
            val = getattr(result, target)
            # For power, invert the value (since lower is better)
            return 1.0/(val + 1e-10) if target == 'power_uw' else val
        
        # Multi-objective with weights
        # If no weights provided, use equal weights
        if not weights:
            weights = {target: 1.0 for target in targets}
        
        # Normalize weights
        total_weight = sum(weights.values())
        norm_weights = {k: v/total_weight for k, v in weights.items()} if total_weight > 0 else weights
        
        # Calculate weighted score
        score = 0.0
        for target in targets:
            if target in weights:
                val = getattr(result, target)
                # Handle special case for power (lower is better)
                if target == 'power_uw':
                    # Invert power so that lower values are better
                    normalized_value = 1.0 / (val + 1e-10)  # Add small value to avoid division by zero
                else:
                    normalized_value = val
                
                score += norm_weights.get(target, 1.0) * normalized_value
        
        return score
    
    def _calculate_ratio(self, result, numerator_keys, denominator_keys):
        """Helper method to calculate ratio for any result object"""
        # Calculate numerator (product of all numerator metrics)
        numerator = 1.0
        for key in numerator_keys:
            numerator *= getattr(result, key, 1.0)
        
        # Calculate denominator (product of all denominator metrics)
        denominator = 1.0
        for key in denominator_keys:
            denominator *= getattr(result, key, 1.0)
        
        return numerator / (denominator + 1e-10)
    
    def genetic_algorithm(self, n_samples: int, previous_results: List = None,
                     targets: List[str] = None, weights: Dict[str, float] = None,
                     population_size: int = 20, generations: int = None,
                     mutation_rate: float = 0.1, crossover_rate: float = 0.8, 
                     tournament_size: int = 3) -> List[Tuple]:
    
        """Genetic Algorithm with tunable parameters and variable-specific ranges"""
        if generations is None:
            generations = max(3, n_samples // population_size)
        
        print(f"  Using Genetic Algorithm: {population_size} population × {generations} generations")
        print(f"  Parameters: mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, tournament_size={tournament_size}")
        print(f"  Targets: {targets}")
        
        # Initialize population
        if previous_results and len(previous_results) >= population_size // 2:
            # Sort by fitness based on targets/weights
            if targets:
                # ✅ Use centralized metric computation for composite targets
                if 'composite' in targets:
                    # Check if we have access to LLM agent
                    if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'llm_agent'):
                        # Use centralized computation - handles ALL formulation types
                        sorted_results = self._safe_sort_results(
                            previous_results,
                            key_fn =lambda x: self.optimizer.llm_agent._compute_composite_metric(
                                x.to_dict(), self.optimizer.llm_agent.target_metric
                            ),
                            reverse=(self.optimizer.target_metric['direction'] == 'maximize')
                        )
                    else:
                        # Fallback if no LLM agent (shouldn't happen in normal use)
                        print("  Warning: No LLM agent found, using FOM fallback")
                        sorted_results = sorted(previous_results, key=lambda x: x.fom, reverse=True)
                else:
                    # Non-composite: use calculate_fitness for single/multi-objective
                    sorted_results = self._safe_sort_results(
                        previous_results, 
                        key_fn =lambda x: getattr(x, 'fom', None),  # ← Changed to use helper
                        reverse=True
                    )
            else:
                # Default to FOM if no targets specified
                sorted_results = self._safe_sort_results(previous_results, key_fn =lambda x: x.fom, reverse=True)
            
            # Create initial population from top performers
            population = [
                tuple(result.results[var] for var in self.all_var_names)  # Use all_var_names
                for result in sorted_results[:population_size // 2]
            ]
            
            # Fill rest with random individuals
            while len(population) < population_size:
                population.append(self._random_individual())
        else:
            # Not enough previous results - start with random population
            population = [self._random_individual() for _ in range(population_size)]
        
        # Store all evaluated points
        all_points = list(population)
        
        # Main evolution loop
        for gen in range(generations - 1):  # -1 because we already have initial population
            # Selection: tournament selection with configurable tournament_size
            parents = self._tournament_selection(population, k=population_size, tournament_size=tournament_size)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    # Apply crossover based on crossover_rate
                    if random.random() < crossover_rate:
                        child1, child2 = self._crossover(parents[i], parents[i + 1])
                    else:
                        # Skip crossover, keep parents
                        child1, child2 = parents[i], parents[i+1]
                    
                    # Apply mutation with configurable rate
                    offspring.append(self._mutate(child1, mutation_rate=mutation_rate))
                    offspring.append(self._mutate(child2, mutation_rate=mutation_rate))
            
            # New population = offspring
            population = offspring[:population_size]
            all_points.extend(population)
        
        # Return unique points up to n_samples
        seen = set()
        unique_points = []
        for point in all_points:
            if point not in seen and len(unique_points) < n_samples:
                unique_points.append(point)
                seen.add(point)
        
        return unique_points[:n_samples]
    
    def _random_individual(self) -> Tuple:
        """Create random individual with variable-specific ranges"""
        optimizable_values = []
        for var in self.optimizable_vars:
            var_range = self.variables_to_optimize[var]
            optimizable_values.append(random.choice(var_range))
        
        # Create full point with fixed values
        return self._create_full_point(tuple(optimizable_values))
    
    def _tournament_selection(self, population: List[Tuple], k: int, 
                             tournament_size: int = 3) -> List[Tuple]:
        """Tournament selection with configurable size"""
        selected = []
        for _ in range(k):
            # Use the provided tournament size
            tournament = random.sample(population, min(tournament_size, len(population)))
            selected.append(random.choice(tournament))
        return selected
    
    def _mutate(self, individual: Tuple, mutation_rate: float = 0.3) -> Tuple:
        """Mutation with configurable rate and variable-specific ranges"""
        # Extract optimizable values from full individual
        value_dict = {var: val for var, val in zip(self.all_var_names, individual)}
        
        # Mutate only the optimizable variables
        for var in self.optimizable_vars:
            if random.random() < mutation_rate:
                var_range = self.variables_to_optimize[var]
                value_dict[var] = random.choice(var_range)
        
        # Return full individual in correct order
        return tuple(value_dict[var] for var in self.all_var_names)
    
    def _crossover(self, parent1: Tuple, parent2: Tuple) -> Tuple[Tuple, Tuple]:
        """Single-point crossover for GA - only crosses optimizable variables"""
        if self.n_dims <= 1:
            # Not enough dimensions to crossover
            return parent1, parent2
        
        # Extract optimizable values from both parents
        p1_dict = {var: val for var, val in zip(self.all_var_names, parent1)}
        p2_dict = {var: val for var, val in zip(self.all_var_names, parent2)}
        
        # Perform crossover on optimizable variables only
        crossover_point = random.randint(1, self.n_dims - 1)
        
        child1_dict = dict(p1_dict)  # Copy all values
        child2_dict = dict(p2_dict)
        
        # Swap optimizable variables after crossover point
        for i, var in enumerate(self.optimizable_vars):
            if i >= crossover_point:
                child1_dict[var] = p2_dict[var]
                child2_dict[var] = p1_dict[var]
        
        # Return full individuals in correct order
        child1 = tuple(child1_dict[var] for var in self.all_var_names)
        child2 = tuple(child2_dict[var] for var in self.all_var_names)
        
        return child1, child2
    
    def _simulated_annealing(self, n_samples: int, previous_best: Optional[Tuple] = None,
                           targets: List[str] = None, weights: Dict[str, float] = None,
                           initial_temperature: float = 1.0, cooling_rate: float = 0.95) -> List[Tuple]:
        
        """Simulated Annealing with tunable temperature parameters and variable-specific ranges"""
        print(f"  Using Simulated Annealing for {n_samples} evaluations")
        print(f"  Parameters: initial_temperature={initial_temperature}, cooling_rate={cooling_rate}")
        print(f"  Targets: {targets}")
        
        # Start from previous best or random point
        if previous_best:
            current = previous_best
        else:
            current = self._random_individual()
        
        points = [current]
        T = initial_temperature  # Use the provided initial temperature
        T_min = 0.01  # Minimum temperature
        
        while len(points) < n_samples and T > T_min:
            # Generate neighbor
            neighbor = self._get_neighbor(current)
            
            # In real use, this would use proper Metropolis criterion
            # For now, make acceptance probability depend on temperature
            acceptance_prob = min(1.0, 0.3 + 0.7 * (T / initial_temperature))
            if random.random() < acceptance_prob:
                current = neighbor
            
            points.append(current)
            
            # Cool down using the provided cooling rate
            T *= cooling_rate
        
        # Fill remaining with exploration
        while len(points) < n_samples:
            points.append(self._random_individual())
        
        return points[:n_samples]
    
    def _get_neighbor(self, point: Tuple) -> Tuple:
        """Get neighbor for simulated annealing - change 1-2 optimizable dimensions"""
        if self.n_dims == 0:
            return point
        
        # Extract values into dict
        value_dict = {var: val for var, val in zip(self.all_var_names, point)}
        
        # Change 1-2 optimizable dimensions
        n_changes = min(random.randint(1, 2), self.n_dims)
        vars_to_change = random.sample(self.optimizable_vars, n_changes)
        
        for var in vars_to_change:
            var_range = self.variables_to_optimize[var]
            current_val = value_dict[var]
            
            # Try to find current index
            try:
                current_idx = var_range.index(current_val)
                
                if random.random() < 0.7:  # Move to adjacent
                    if random.random() < 0.5 and current_idx > 0:
                        value_dict[var] = var_range[current_idx - 1]
                    elif current_idx < len(var_range) - 1:
                        value_dict[var] = var_range[current_idx + 1]
                    else:
                        value_dict[var] = random.choice(var_range)
                else:  # Jump randomly
                    value_dict[var] = random.choice(var_range)
            except ValueError:
                # Current value not in range (shouldn't happen, but handle gracefully)
                value_dict[var] = random.choice(var_range)
        
        # Return full individual in correct order
        return tuple(value_dict[var] for var in self.all_var_names)

    def simulated_annealing(self, n_samples: int, previous_best: Optional[Tuple] = None,
                       targets: List[str] = None, weights: Dict[str, float] = None,
                       initial_temperature: float = 1.0, cooling_rate: float = 0.95) -> List[Tuple]:
    
        """
        Simulated Annealing with proper Metropolis criterion and variable-specific ranges
        Uses previous results to estimate fitness and guide exploration
        """
        print(f"  Using Simulated Annealing for {n_samples} evaluations")
        print(f"  Parameters: initial_temperature={initial_temperature}, cooling_rate={cooling_rate}")
        print(f"  Targets: {targets}")
        
        # Start from previous best or random point
        if previous_best:
            current = previous_best
        else:
            current = self._random_individual()
        
        points = [current]
        
        # Temperature schedule
        T = initial_temperature
        T_min = 0.01  # Minimum temperature
        
        # Build fitness estimator from previous results (if available)
        use_surrogate = False
        previous_points = []
        previous_fitness = []
        
        if hasattr(self, 'optimizer') and self.optimizer.all_searched_designs:
            use_surrogate = True
            
            # Extract all previous evaluations
            for result in self.optimizer.all_searched_designs:
                point = tuple(
                    result.results.get(var) if hasattr(result, 'results') else getattr(result, var, None)
                    for var in self.all_var_names
                )
                previous_points.append(point)
                
                # Calculate fitness based on targets
                if targets and 'composite' in targets:
                    if hasattr(self.optimizer, 'llm_agent'):
                        fitness = self.optimizer.llm_agent._compute_composite_metric(
                            result.to_dict(), self.optimizer.llm_agent.target_metric
                        )
                    else:
                        fitness = result.fom
                elif targets:
                    fitness = self.calculate_fitness(result, targets, weights)
                else:
                    fitness = result.fom
                
                previous_fitness.append(fitness)
            
            print(f"  Using surrogate model with {len(previous_points)} previous evaluations")
            
            # Determine if we're maximizing or minimizing
            if hasattr(self.optimizer, 'target_metric'):
                is_maximizing = (self.optimizer.target_metric['direction'] == 'maximize')
            else:
                is_maximizing = True  # Default
        else:
            # No previous data - use random exploration with temperature-based acceptance
            print(f"  No previous data - using temperature-based exploration")
            is_maximizing = True
        
        # Estimate initial fitness
        if use_surrogate:
            current_fitness = self._estimate_fitness(
                current, previous_points, previous_fitness
            )
        else:
            current_fitness = 0.0
        
        # Simulated Annealing loop
        iteration = 0
        while len(points) < n_samples and T > T_min:
            iteration += 1
            
            # Generate neighbor
            neighbor = self._get_neighbor(current)
            
            # Estimate neighbor fitness
            if use_surrogate:
                neighbor_fitness = self._estimate_fitness(
                    neighbor, previous_points, previous_fitness
                )
                
                # Calculate fitness difference (accounting for maximize vs minimize)
                if is_maximizing:
                    delta = neighbor_fitness - current_fitness  # Positive if neighbor is better
                else:
                    delta = current_fitness - neighbor_fitness  # Positive if neighbor is better
                
                # Metropolis acceptance criterion
                if delta > 0:
                    # Neighbor is better - always accept
                    acceptance_prob = 1.0
                    accept = True
                else:
                    # Neighbor is worse - accept with probability exp(delta / T)
                    acceptance_prob = np.exp(delta / T)
                    accept = (random.random() < acceptance_prob)
                
                if accept:
                    current = neighbor
                    current_fitness = neighbor_fitness
                
                # Debug info every 10 iterations
                if iteration % 10 == 0:
                    print(f"    Iter {iteration}: T={T:.3f}, current_fitness={current_fitness:.4f}, "
                          f"last_accept_prob={acceptance_prob:.3f}")
            else:
                # No surrogate - use temperature-based exploration
                # Higher temperature = more likely to accept any move
                acceptance_prob = 0.3 + 0.7 * (T / initial_temperature)
                
                if random.random() < acceptance_prob:
                    current = neighbor
            
            # Add current point to search list
            points.append(current)
            
            # Cool down
            T *= cooling_rate
        
        # Fill remaining budget with exploration around best found points
        if len(points) < n_samples:
            print(f"  Temperature reached minimum, filling remaining with local exploration")
            
            # Use the last accepted point as center for local search
            remaining = n_samples - len(points)
            local_points = self._local_search_around_best(current, remaining, radius=1)
            points.extend(local_points)
        
        return points[:n_samples]
    
    def _estimate_fitness(self, point: Tuple, known_points: List[Tuple], 
                         known_fitness: List[float]) -> float:
        
        """
        Estimate fitness using inverse distance weighting from known points
        Works with variable-specific ranges
        """
        if not known_points:
            return 0.0
        
        # Find k nearest neighbors (k=5)
        k = min(5, len(known_points))
        distances = []
        
        for i, known_point in enumerate(known_points):
            # **FIX: Skip points with None fitness**
            if known_fitness[i] is None:
                continue
                
            dist = self._discrete_distance(point, known_point)
            distances.append((dist, i))
        
        # **FIX: Check if we have any valid points**
        if not distances:
            return 0.0
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        nearest = distances[:k]
        
        # If exact match found, return that fitness
        if nearest[0][0] == 0:
            return known_fitness[nearest[0][1]]
        
        # Inverse distance weighting
        total_weight = 0.0
        weighted_sum = 0.0
        
        for dist, idx in nearest:
            # **FIX: Double-check fitness is not None (defensive programming)**
            if known_fitness[idx] is None:
                continue
                
            weight = 1.0 / (dist + 0.1)  # Add small constant to avoid division by zero
            total_weight += weight
            weighted_sum += weight * known_fitness[idx]
        
        estimated_fitness = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Add uncertainty bonus for exploration (further from known points = higher uncertainty)
        min_dist = nearest[0][0]
        uncertainty_bonus = min_dist * 0.1  # Small bonus for unexplored regions
        
        return estimated_fitness + uncertainty_bonus
    
    def true_bayesian_optimization(self, n_samples: int, W_values: List[float],
                     previous_results: List = None, previous_best = None,
                     targets: List[str] = None, weights: Dict[str, float] = None,
                     acquisition_function: str = "EI",  # Changed default
                     exploration_weight: float = 0.1):    # Changed default -> List[Tuple]:  # Kept for API compatibility
        
    
        """
        True Bayesian Optimization with Integer encoding
        
        Note: kernel_type, kernel_nu, kernel_length_scale are kept in signature for
        compatibility but not used due to scikit-optimize 0.10.2 limitations.
        The most important improvements are:
        1. Integer encoding (vs Categorical)
        2. LCB acquisition function
        3. High exploration_weight (kappa=2.0+)
        """
        from skopt import Optimizer
        from skopt.space import Integer
        
        if targets is None:
            targets = ['fom']
        
        print(f"  Using GP-based Bayesian Optimization for {n_samples} samples")
        print(f"  Parameters: acquisition_function={acquisition_function}, exploration_weight={exploration_weight}")
        print(f"  Optimization targets: {', '.join(targets)}")
        
        if not previous_results or len(previous_results) < 3:
            print("  Not enough data for BO model, using LHS for initialization")
            return self.latin_hypercube_sampling(n_samples)
        
        # Create per-variable bidirectional mappings using variable-specific value lists
        var_maps = {}       # var -> {index: value}
        var_reverse_maps = {}  # var -> {value: index}
        for var in self.all_var_names:
            vals = self.variables_to_optimize.get(var, self.W_values)
            var_maps[var] = {i: v for i, v in enumerate(vals)}
            var_reverse_maps[var] = {v: i for i, v in enumerate(vals)}

        print(f"  Using Integer encoding with per-variable discrete values")

        space = [
            Integer(0, len(self.variables_to_optimize.get(var, self.W_values)) - 1, name=f"{var}_idx")
            for var in self.all_var_names
        ]

        # Extract X and y from previous results
        X = []
        y = []

        for result in previous_results:
            # Create a fixed-length point with correct dimensionality
            point = []
            valid_point = True

            for var in self.all_var_names:
                # First check if variable exists in results dictionary
                if hasattr(result, 'results') and var in result.results:
                    val = result.results[var]
                else:
                    # Try attribute access as fallback
                    val = getattr(result, var, None)

                # Verify value is in per-variable reverse map
                rev_map = var_reverse_maps[var]
                if val in rev_map:
                    point.append(rev_map[val])
                else:
                    # If not found, print warning and mark point as invalid
                    print(f"  Warning: Value {val} for {var} not in W_reverse_map")
                    valid_point = False
                    break
            
            # Only add points with correct dimensions and valid indices
            if valid_point and len(point) == len(self.all_var_names):
                X.append(point)
                
                # Extract objective value
                if 'composite' in targets:
                    try:
                        composite_value = self.optimizer.llm_agent._compute_composite_metric(
                            result.to_dict(), self.optimizer.llm_agent.target_metric
                        )
                        direction = self.optimizer.llm_agent.target_metric.get('direction', 'maximize')
                        objective = -composite_value if direction == 'maximize' else composite_value
                    except Exception as e:
                        print(f"  Warning: Error computing composite metric: {e}, using FOM")
                        # Try to get FOM from results dictionary first
                        if hasattr(result, 'results') and 'fom' in result.results:
                            objective = -result.results['fom']
                        else:
                            objective = -getattr(result, 'fom', 0.0)
                elif len(targets) == 1:
                    target = targets[0]
                    
                    # First try accessing from results dictionary
                    if hasattr(result, 'results') and target in result.results:
                        target_value = result.results[target]
                    # Fall back to direct attribute access
                    elif hasattr(result, target):
                        target_value = getattr(result, target)
                    else:
                        print(f"  Warning: Target '{target}' not found in result, using FOM")
                        # Get FOM from results dictionary
                        target_value = result.results.get('fom', 0.0) if hasattr(result, 'results') else 0.0
                    
                    objective = target_value if target == 'power_uw' else -target_value
                else:
                    objective = 0.0
                    for target in targets:
                        # Try both results dictionary and direct attribute access
                        if hasattr(result, 'results') and target in result.results:
                            value = result.results[target]
                        elif hasattr(result, target):
                            value = getattr(result, target)
                        else:
                            print(f"  Warning: Target '{target}' not found in result")
                            continue
                            
                        weight = weights.get(target, 1.0) if weights else 1.0
                        objective += weight * value if target == 'power_uw' else -weight * value
                
                y.append(objective)
            else:
                print(f"  Skipping point with invalid dimensions or values (expected {len(self.all_var_names)}, got {len(point)})")
        
        # Check if we have enough valid points to train a model
        if len(X) < 3:
            print(f"  Not enough valid historical data points ({len(X)} < 3), using LHS instead")
            return self.latin_hypercube_sampling(n_samples)
        
        print(f"  Historical data: {len(X)} designs with objectives ranging from {min(y):.3f} to {max(y):.3f}")
        
        # Set up acquisition function
        acq_func_kwargs = {}
        if acquisition_function.upper() in ["EI", "PI"]:
            acq_func_kwargs = {"xi": exploration_weight}
            print(f"  Acquisition: {acquisition_function} with xi={exploration_weight}")
        elif acquisition_function.upper() in ["LCB", "UCB"]:
            acq_func_kwargs = {"kappa": exploration_weight}
            print(f"  Acquisition: {acquisition_function} with kappa={exploration_weight}")
        
        # Create optimizer with default GP
        opt = Optimizer(
            dimensions=space,
            base_estimator="GP",
            acq_func=acquisition_function.upper(),
            acq_func_kwargs=acq_func_kwargs,
            n_initial_points=0,
            random_state=42
        )
        
        try:
            opt.tell(X, y)
        except Exception as e:
            print(f"  Error initializing optimizer: {e}")
            print(f"  Falling back to LHS sampling")
            return self.latin_hypercube_sampling(n_samples)
        
        # Collect suggested points
        suggested_points = []
        
        if previous_best:
            try:
                if isinstance(previous_best, tuple):
                    if len(previous_best) == len(self.all_var_names):
                        suggested_points.append(previous_best)
                else:
                    # Try to extract from results dictionary first
                    if hasattr(previous_best, 'results'):
                        prev_best_tuple = tuple(
                            previous_best.results.get(var, 0) 
                            for var in self.all_var_names
                        )
                    else:
                        # Fall back to direct attribute access
                        prev_best_tuple = tuple(
                            getattr(previous_best, var, 0)
                            for var in self.all_var_names
                        )
                    suggested_points.append(prev_best_tuple)
            except Exception as e:
                print(f"  Error adding previous best point: {e}")
        
        # Ask for new points
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(suggested_points) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                next_x = opt.ask()
                # Dynamically construct point based on actual dimension count
                point = tuple(var_maps[var][next_x[i]] for i, var in enumerate(self.all_var_names))
                if point not in suggested_points:
                    suggested_points.append(point)
            except Exception as e:
                print(f"  Error asking for next point: {e}")
                continue
        
        print(f"  Primary acquisition function suggested {len(suggested_points)} unique points")
        
        # Alternative acquisition functions
        if len(suggested_points) < n_samples:
            print(f"  Need {n_samples - len(suggested_points)} more points, trying alternative acquisition functions")
            
            alt_acq_funcs = [af for af in ["PI", "EI", "LCB"] if af != acquisition_function.upper()]
            
            for acq_func in alt_acq_funcs:
                if len(suggested_points) >= n_samples:
                    break
                    
                print(f"  Trying {acq_func}...")
                
                alt_kwargs = {"xi": exploration_weight} if acq_func in ["EI", "PI"] else {"kappa": exploration_weight}
                
                alt_opt = Optimizer(
                    dimensions=space,
                    base_estimator="GP",
                    acq_func=acq_func,
                    acq_func_kwargs=alt_kwargs,
                    n_initial_points=0,
                    random_state=42 + len(suggested_points)
                )
                
                try:
                    alt_opt.tell(X, y)
                except Exception as e:
                    print(f"  Error initializing alternative optimizer: {e}")
                    continue
                
                attempts = 0
                max_alt_attempts = (n_samples - len(suggested_points)) * 5
                
                while len(suggested_points) < n_samples and attempts < max_alt_attempts:
                    attempts += 1
                    try:
                        next_x = alt_opt.ask()
                        # Dynamically construct point based on actual dimension count
                        point = tuple(var_maps[var][next_x[i]] for i, var in enumerate(self.all_var_names))
                        if point not in suggested_points:
                            suggested_points.append(point)
                    except Exception as e:
                        print(f"  Error asking for next point from alternative optimizer: {e}")
                        continue
        
        # Random points fallback
        if len(suggested_points) < n_samples:
            print(f"  Still need {n_samples - len(suggested_points)} points, adding random samples")
            attempts = 0
            max_random_attempts = n_samples * 20
            
            while len(suggested_points) < n_samples and attempts < max_random_attempts:
                attempts += 1
                point = self._random_individual()
                if point not in suggested_points:
                    suggested_points.append(point)
        
        # LHS fallback
        if len(suggested_points) < n_samples:
            print(f"  Warning: Only generated {len(suggested_points)} unique points, filling with LHS")
            lhs_points = self.latin_hypercube_sampling(n_samples - len(suggested_points))
            for point in lhs_points:
                if point not in suggested_points:
                    suggested_points.append(point)
                if len(suggested_points) >= n_samples:
                    break
        
        print(f"  Total points generated: {len(suggested_points)}")
        
        return suggested_points[:n_samples]
    
    
    def _predict_with_surrogate(self, point: Tuple, known_points: List[Tuple],
                                known_foms: List[float]) -> Tuple[float, float]:
        """Simple surrogate prediction using weighted nearest neighbors"""
        if not known_points:
            return 0.0, 1.0
        
        # Calculate distances to known points
        distances = [self._discrete_distance(point, kp) for kp in known_points]
        
        # Inverse distance weighting for mean
        weights = [1.0 / (d + 1e-6) for d in distances]
        total_weight = sum(weights)
        mean = sum(w * f for w, f in zip(weights, known_foms)) / total_weight
        
        # Uncertainty based on minimum distance (exploration bonus)
        min_distance = min(distances)
        uncertainty = min_distance / (self.n_dims * len(self.W_values))
        
        return mean, uncertainty
    
    def _discrete_distance(self, point1: Tuple, point2: Tuple) -> float:
        """
        Calculate distance between two points considering only optimizable variables
        Fixed variables don't contribute to distance
        """
        if self.n_dims == 0:
            return 0.0
        
        # Create dicts for easy lookup
        p1_dict = {var: val for var, val in zip(self.all_var_names, point1)}
        p2_dict = {var: val for var, val in zip(self.all_var_names, point2)}
        
        distance = 0.0
        
        # Only count differences in optimizable variables
        for var in self.optimizable_vars:
            var_range = self.variables_to_optimize[var]
            
            val1 = p1_dict[var]
            val2 = p2_dict[var]
            
            # If values are different, add normalized distance
            if val1 != val2:
                try:
                    idx1 = var_range.index(val1)
                    idx2 = var_range.index(val2)
                    # Normalized by range size
                    distance += abs(idx1 - idx2) / (len(var_range) - 1) if len(var_range) > 1 else 0
                except ValueError:
                    # Value not in range (shouldn't happen), treat as maximum distance
                    distance += 1.0
        
        return distance



    def optuna_bayesian_optimization(self, n_samples: int, W_values: List[float] = None,
                                 previous_results: List = None, previous_best = None,
                                 targets: List[str] = None, weights: Dict[str, float] = None,
                                 n_ei_candidates: int = 30,
                                 n_startup_trials: int = 5,
                                 multivariate: bool = False,
                                 prior_weight: float = 1.0) -> List[Tuple]:
    
        """
        Bayesian Optimization using Optuna TPE with variable-specific ranges.
        Only optimizes variables in self.optimizable_vars; fixed variables use their assigned values.
        
        TPE is superior to GP for discrete/categorical spaces because:
        - Native discrete variable support (no rounding issues)
        - Models probability distributions directly
        - No kernel smoothness assumptions
        - Naturally generates diverse points
        
        Parameters:
        -----------
        n_samples: int
            Number of design points to generate
        W_values: List[float] (optional)
            Not used - kept for backward compatibility
        previous_results: List
            Historical evaluation results for TPE training
        previous_best: tuple or OTAResult
            Best design found so far
        targets: List[str]
            Optimization targets (e.g., ['composite'], ['fom'])
        weights: Dict[str, float]
            Weights for multi-objective optimization
        
        Returns:
        --------
        List[Tuple]: List of full design tuples (including fixed variables)
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
            import math  # ← ADD THIS IMPORT
        except ImportError:
            print("  ⚠️  Optuna not installed. Install with: pip install optuna")
            print("  Falling back to Latin Hypercube Sampling...")
            return self.latin_hypercube_sampling(n_samples)
        
        if targets is None:
            targets = ['fom']
        
        print(f"  Using Optuna TPE Bayesian Optimization for {n_samples} samples")
        print(f"  Optimizing {self.n_dims} variables (out of {self.total_vars} total)")
        print(f"  TPE Parameters:")
        print(f"    - n_ei_candidates: {n_ei_candidates} (exploration strength)")
        print(f"    - n_startup_trials: {n_startup_trials} (random before Bayesian)")
        print(f"    - multivariate: {multivariate} (parameter interactions)")
        print(f"    - prior_weight: {prior_weight} (trust in prior)")
        print(f"  Optimization targets: {', '.join(targets)}")
        
        # Print variable ranges
        for var in self.optimizable_vars:
            print(f"    {var}: {len(self.variables_to_optimize[var])} choices")
        for var in self.fixed_vars:
            print(f"    {var}: fixed at {self.variables_fixed[var]}")
        
        # Check if we have enough historical data
        if not previous_results or len(previous_results) < 3:
            print("  Not enough data for TPE model, using LHS for initialization")
            return self.latin_hypercube_sampling(n_samples)
        
        if self.n_dims == 0:
            print("  No variables to optimize! Returning fixed point.")
            fixed_point = tuple(self.variables_fixed[var] for var in self.all_var_names)
            return [fixed_point] * n_samples
        
        print(f"  Historical data: {len(previous_results)} designs available for TPE training")
        
        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create TPE sampler with LLM-controlled parameters
        sampler = TPESampler(
            n_startup_trials=min(n_startup_trials, len(previous_results)),
            n_ei_candidates=n_ei_candidates,
            seed=42,
            multivariate=multivariate,
            group=multivariate,
            warn_independent_sampling=False,
            constant_liar=True
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'ota_tpe_optimization_{id(self)}'
        )
        
        # Load historical evaluations into Optuna
        print(f"  Loading {len(previous_results)} historical evaluations into TPE model...")
        
        valid_count = 0
        invalid_count = 0
        
        for result in previous_results:
            # Calculate objective value
            objective_value = self._calculate_objective_value(result, targets, weights)
            
            # Skip if objective value is None or invalid
            if objective_value is None:
                invalid_count += 1
                continue
            
            try:
                # Validate objective value is a real number
                objective_value = float(objective_value)
                if math.isnan(objective_value) or math.isinf(objective_value):
                    invalid_count += 1
                    continue
            except (ValueError, TypeError):
                invalid_count += 1
                continue
            
            # Extract only optimizable variables for the trial
            params = {}
            has_none_param = False
            for var in self.optimizable_vars:
                if hasattr(result, "results"):
                    param_value = result.results.get(var)
                else:
                    param_value = result.get(var)
                
                # Check if parameter is None
                if param_value is None:
                    has_none_param = True
                    break
                
                params[var] = param_value
            
            # Skip if any parameter is None
            if has_none_param:
                invalid_count += 1
                continue
            
            # Create trial with only optimizable parameters
            try:
                trial = optuna.trial.create_trial(
                    params=params,
                    distributions={
                        var: optuna.distributions.CategoricalDistribution(
                            choices=self.variables_to_optimize[var]
                        )
                        for var in self.optimizable_vars
                    },
                    values=[objective_value]
                )
                
                study.add_trial(trial)
                valid_count += 1
            except Exception as e:
                print(f"  Warning: Failed to add trial: {e}")
                invalid_count += 1
        
        # Report statistics
        print(f"  Successfully loaded {valid_count}/{len(previous_results)} historical designs")
        if invalid_count > 0:
            print(f"  ⚠️  Skipped {invalid_count} designs with invalid objective/parameter values")
        
        # Check if we have enough valid data
        if valid_count == 0:
            print(f"  ⚠️  No valid historical data - Falling back to LHS")
            return self.latin_hypercube_sampling(n_samples)
        
        if valid_count < 3:
            print(f"  ⚠️  Only {valid_count} valid designs - Using LHS for better coverage")
            return self.latin_hypercube_sampling(n_samples)
        
        # Get objective statistics for monitoring
        all_objectives = [trial.value for trial in study.trials]
        print(f"  Historical objectives: min={min(all_objectives):.3f}, max={max(all_objectives):.3f}, mean={sum(all_objectives)/len(all_objectives):.3f}")
        
        # Collect suggested points
        suggested_points = []
        
        # Include previous best point for continuity
        if previous_best:
            if isinstance(previous_best, tuple):
                suggested_points.append(previous_best)
            else:
                prev_best_tuple = tuple(
                    (previous_best.results.get(var)
                     if hasattr(previous_best, "results")
                     else previous_best.get(var))
                    for var in self.all_var_names
                )
                suggested_points.append(prev_best_tuple)
            print(f"  Including previous best design in suggestions")
        
        # Ask TPE for suggestions
        print(f"  Asking TPE for {n_samples} design suggestions...")
        
        attempts = 0
        max_attempts = n_samples * 20
        
        while len(suggested_points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Ask Optuna for a suggestion
            trial = study.ask()
            
            # Get suggested parameters for optimizable variables only
            optimizable_values = []
            for var in self.optimizable_vars:
                var_range = self.variables_to_optimize[var]
                value = trial.suggest_categorical(var, var_range)
                optimizable_values.append(value)
            
            # Create full point with fixed values
            point = self._create_full_point(tuple(optimizable_values))
            
            # Check if unique
            if point not in suggested_points:
                suggested_points.append(point)
                study.tell(trial, values=[0.0])
            else:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        
        print(f"  TPE suggested {len(suggested_points)} unique points after {attempts} attempts")
        
        # If TPE couldn't generate enough unique points
        if len(suggested_points) < n_samples:
            needed = n_samples - len(suggested_points)
            print(f"  TPE provided {len(suggested_points)}/{n_samples} points, using LHS for {needed} more...")
            
            lhs_points = self.latin_hypercube_sampling(needed * 2)
            
            for point in lhs_points:
                if point not in suggested_points:
                    suggested_points.append(point)
                if len(suggested_points) >= n_samples:
                    break
        
        # Summary statistics
        print(f"  ✅ Generated {len(suggested_points)} intelligent points ({len(suggested_points)/n_samples*100:.1f}% intelligent)")
        
        return suggested_points[:n_samples]    
    
            
    def _calculate_objective_value(self, result, targets: List[str], 
                               weights: Dict[str, float] = None) -> float:
        
        """
        Calculate objective value for a design result
        Returns value to MAXIMIZE (negate if needed)
        Returns None if calculation fails
        """
        try:
            if 'composite' in targets:
                try:
                    composite_value = self.optimizer.llm_agent._compute_composite_metric(
                        result.to_dict(), self.optimizer.llm_agent.target_metric
                    )
                    if composite_value is None:
                        return None
                    
                    direction = self.optimizer.llm_agent.target_metric.get('direction', 'maximize')
                    return composite_value if direction == 'maximize' else -composite_value
                except Exception as e:
                    print(f"  Warning: Error computing composite metric: {e}")
                    # Fall through to FOM
            
            if len(targets) == 1:
                target = targets[0]
                if hasattr(result, target):
                    target_value = getattr(result, target)
                    if target_value is None:
                        return None
                    # For power, we want to minimize, so negate
                    return -target_value if target == 'power_uw' else target_value
                elif hasattr(result, 'results') and target in result.results:
                    target_value = result.results[target]
                    if target_value is None:
                        return None
                    return -target_value if target == 'power_uw' else target_value
                else:
                    # Try FOM as fallback
                    if hasattr(result, 'fom'):
                        fom_val = getattr(result, 'fom')
                        return fom_val if fom_val is not None else None
                    return None
            
            else:
                # Multi-objective weighted combination
                objective = 0.0
                for target in targets:
                    target_value = None
                    if hasattr(result, target):
                        target_value = getattr(result, target)
                    elif hasattr(result, 'results') and target in result.results:
                        target_value = result.results[target]
                    
                    if target_value is None:
                        return None  # If any target is None, return None
                    
                    weight = weights.get(target, 1.0) if weights else 1.0
                    objective += -weight * target_value if target == 'power_uw' else weight * target_value
                
                return objective
        
        except Exception as e:
            print(f"  Warning: Error calculating objective value: {e}")
            return None

            
    def adaptive_search(self, n_samples: int, previous_results: List = None,
               previous_best: Optional[Tuple] = None,
               targets: List[str] = None, weights: Dict[str, float] = None,
               explore_weight: float = 0.4, exploit_weight: float = 0.4, 
               random_weight: float = 0.2, radius: int = 2) -> List[Tuple]:
        
       
        """Adaptive search with configurable strategy weights and search radius"""
        print(f"  Using Adaptive Search for {n_samples} samples")
        print(f"  Parameters: explore_weight={explore_weight}, exploit_weight={exploit_weight}, random_weight={random_weight}, radius={radius}")
        print(f"  Targets: {targets}")
        
        # Normalize weights to sum to 1.0
        total_weight = explore_weight + exploit_weight + random_weight
        if total_weight <= 0:
            # Default to balanced if invalid weights
            explore_weight, exploit_weight, random_weight = 0.4, 0.4, 0.2
        else:
            explore_weight /= total_weight
            exploit_weight /= total_weight
            random_weight /= total_weight
        
        points = []
        
        # Strategy 1: Exploit best regions with configurable radius
        n_exploit = max(1, int(n_samples * exploit_weight))
        if previous_best:
            exploit_points = self._local_search_around_best(
                previous_best, n_exploit, radius=radius
            )
            points.extend(exploit_points)
        
        # Strategy 2: Explore diverse regions using the same weights
        n_explore = max(1, int(n_samples * explore_weight))
        explored_regions = self._get_unexplored_regions(
            previous_results, 
            targets, 
            weights,
            exploration_weight=explore_weight,
            exploitation_weight=exploit_weight
        )
        explore_points = self._sample_from_regions(explored_regions, n_explore)
        points.extend(explore_points)
        
        # Strategy 3: Random exploration
        n_random = n_samples - len(points)
        for _ in range(n_random):
            points.append(self._random_individual())
        
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                unique_points.append(p)
                seen.add(p)
            if len(unique_points) >= n_samples:
                break
        
        return unique_points[:n_samples]
    
    def _local_search_around_best(self, best_point: Tuple, n_samples: int,
                                  radius: int = 2) -> List[Tuple]:
        """
        Search around best point within radius, respecting variable-specific ranges.
        Only varies optimizable variables; fixed variables remain constant.
        """
        if self.n_dims == 0:
            # No optimizable variables, return the same point
            return [best_point] * n_samples
        
        points = []
        
        # Extract best point values into dict
        best_dict = {var: val for var, val in zip(self.all_var_names, best_point)}
        
        # Generate all combinations within radius
        attempts = 0
        max_attempts = n_samples * 10  # Try up to 10x to get enough unique points
        
        while len(points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Start with best point
            new_dict = dict(best_dict)
            
            # Decide how many optimizable dimensions to vary
            n_dims_to_vary = random.randint(1, min(3, self.n_dims))
            dims_to_vary = random.sample(self.optimizable_vars, n_dims_to_vary)
            
            # Vary selected optimizable dimensions
            for var in dims_to_vary:
                var_range = self.variables_to_optimize[var]
                current_val = best_dict[var]
                
                try:
                    current_idx = var_range.index(current_val)
                    
                    # Apply offset within radius
                    offset = random.randint(-radius, radius)
                    new_idx = max(0, min(len(var_range) - 1, current_idx + offset))
                    
                    new_dict[var] = var_range[new_idx]
                    
                except ValueError:
                    # Current value not in range (shouldn't happen), pick random nearby
                    new_dict[var] = random.choice(var_range)
            
            # Create full point in correct order
            point_tuple = tuple(new_dict[var] for var in self.all_var_names)
            
            # Add if unique
            if point_tuple not in points:
                points.append(point_tuple)
        
        # If we couldn't generate enough unique points, fill with random individuals
        while len(points) < n_samples:
            points.append(self._random_individual())
        
        return points[:n_samples]
    
    def _get_unexplored_regions(self, previous_results: List, targets=None, weights=None,
                               exploration_weight: float = 0.6, 
                               exploitation_weight: float = 0.4) -> List[Dict]:
        
        """
        Identify unexplored regions of design space with variable-specific ranges.
        Only explores optimizable variables; fixed variables are constant.
        
        Parameters:
        -----------
        previous_results: List
            List of previously evaluated designs
        targets: List[str], optional
            Target metrics to optimize (e.g., ['gain_db'], ['composite'])
        weights: Dict[str, float], optional
            Weights for multiple targets
        exploration_weight: float
            Weight for exploration score (0.0-1.0)
        exploitation_weight: float
            Weight for exploitation score (0.0-1.0)
        """
        if not previous_results:
            return [{'center': self._random_individual(), 'priority': 1.0}]
        
        print(f"  Finding unexplored regions using target(s): {targets}")
        print(f"  Region weights: exploration={exploration_weight:.2f}, exploitation={exploitation_weight:.2f}")
        
        # Create grid of regions
        regions = []
        
        # Sample some random regions
        for _ in range(10):
            regions.append({
                'center': self._random_individual(),
                'priority': 1.0,
                'exploration_score': 0.0,
                'exploitation_score': 0.0
            })
        
        # Calculate exploration score (distance from previous designs)
        for region in regions:
            # Calculate minimum distance to any previous design
            min_distance = float('inf')
            
            for result in previous_results:
                point = tuple(
                    result.results[var]
                    for var in self.all_var_names  # Use all_var_names
                )
                dist = self._discrete_distance(region['center'], point)
                min_distance = min(min_distance, dist)
            
            # Normalize by maximum possible distance (only considering optimizable dims)
            max_possible_dist = self.n_dims if self.n_dims > 0 else 1
            region['exploration_score'] = min(1.0, min_distance / max_possible_dist)
        
        # Calculate exploitation score (potential performance based on nearby designs)
        for region in regions:
            # Find nearby designs
            nearby_results = []
            for result in previous_results:
                point = tuple(
                    result.results[var]
                    for var in self.all_var_names  # Use all_var_names
                )
                dist = self._discrete_distance(region['center'], point)
                # Adjusted distance threshold based on optimizable dimensions
                if dist <= self.n_dims * 2 if self.n_dims > 0 else 1:
                    nearby_results.append((result, dist))
            
            if not nearby_results:
                region['exploitation_score'] = 0.5
                continue
            
            nearby_results.sort(key=lambda x: x[1])
            nearby_results = nearby_results[:5]
            
            # Calculate weighted average performance
            total_weight = 0.0
            weighted_performance = 0.0
            
            for result, dist in nearby_results:
                weight = 1.0 / (dist + 0.5)
                total_weight += weight
                
                # ✅ Use centralized metric computation
                if targets and 'composite' in targets:
                    performance = self.optimizer.llm_agent._compute_composite_metric(
                        result.to_dict(), self.optimizer.llm_agent.target_metric
                    )
                else:
                    # Use regular calculate_fitness for non-composite
                    performance = self.calculate_fitness(result, targets, weights)
                
                weighted_performance += performance * weight
            
            if total_weight > 0:
                region['exploitation_score'] = weighted_performance / total_weight
            else:
                region['exploitation_score'] = 0.5
        
        # Normalize weights if they don't sum to 1.0
        total_weight = exploration_weight + exploitation_weight
        if total_weight <= 0:
            # Default if weights are invalid
            norm_exploration = 0.6
            norm_exploitation = 0.4
        else:
            norm_exploration = exploration_weight / total_weight
            norm_exploitation = exploitation_weight / total_weight
        
        # Calculate final priority using the normalized input weights
        for region in regions:
            region['priority'] = (region['exploration_score'] * norm_exploration + 
                                 region['exploitation_score'] * norm_exploitation)
        
        # Return sorted by priority (highest first)
        return sorted(regions, key=lambda x: x['priority'], reverse=True)
    
    def _sample_from_regions(self, regions: List[Dict], n_samples: int) -> List[Tuple]:
        """Sample points from prioritized regions, respecting variable-specific ranges"""
        points = []
        
        total_priority = sum(r['priority'] for r in regions)
        if total_priority == 0:
            total_priority = 1.0
        
        for i, region in enumerate(regions):
            if len(points) >= n_samples:
                break
            
            # Sample more from high-priority regions
            n_from_region = max(1, int(n_samples * region['priority'] / total_priority))
            
            for _ in range(n_from_region):
                # Sample near region center using _get_neighbor (respects variable ranges)
                point = self._get_neighbor(region['center'])
                if point not in points:
                    points.append(point)
                
                if len(points) >= n_samples:
                    break
        
        # Fill remaining with random if needed
        while len(points) < n_samples:
            point = self._random_individual()
            if point not in points:
                points.append(point)
        
        return points[:n_samples]
    
    def multi_start_local_search(self, n_samples: int, 
                             previous_results: List = None,
                             targets: List[str] = None, 
                             weights: Dict[str, float] = None,
                             n_starts: int = 5,
                             search_radius: int = 1) -> List[Tuple]:
        """Multi-start local search with configurable radius and variable-specific ranges"""
        print(f"  Using Multi-Start Local Search: {n_starts} starting points")
        print(f"  Parameters: search_radius={search_radius}")
        print(f"  Targets: {targets}")
        print(f"  Optimizing {self.n_dims} variables")
        
        if self.n_dims == 0:
            print("  No variables to optimize! Returning fixed point.")
            fixed_point = tuple(self.variables_fixed[var] for var in self.all_var_names)
            return [fixed_point] * n_samples
        
        points = []
        samples_per_start = max(1, n_samples // n_starts)
        
        # Get starting points
        if previous_results and len(previous_results) >= n_starts:
            # ✅ Use centralized metric computation for ALL formulation types
            if targets and 'composite' in targets:
                sorted_results = sorted(
                    previous_results,
                    key=lambda x: self.optimizer.llm_agent._compute_composite_metric(
                        x.to_dict(), self.optimizer.llm_agent.target_metric
                    ),
                    reverse=(self.optimizer.target_metric['direction'] == 'maximize')
                )
            else:
                # Single target or multi-objective with weights
                sorted_results = self._safe_sort_results(
                    previous_results, 
                    key_fn =lambda x: self.calculate_fitness(x, targets, weights),
                    reverse=True
                )
    
            starts = [
                tuple(getattr(r, var) for var in self.all_var_names)  # Use all_var_names
                for r in sorted_results[:n_starts]
            ]
        else:
            # Random starts
            starts = [self._random_individual() for _ in range(n_starts)]
        
        # Local search from each start
        for start in starts:
            local_points = self._local_search_around_best(start, samples_per_start, radius=search_radius)
            points.extend(local_points)
        
        # Fill remaining budget
        while len(points) < n_samples:
            point = self._random_individual()
            if point not in points:
                points.append(point)
        
        return points[:n_samples]



def enhanced_generate_search_points(config, optimizer, n_samples, W_values, method='random',
                               previous_best=None, search_radius=None,
                               all_previous_results=None, targets=None, 
                               weights=None, algorithm_params=None,
                               optimization_config=None):
    
    """
    Enhanced search point generation with advanced algorithms and custom parameters.
    Now supports variable-specific ranges through optimization_config.
    
    Parameters:
    -----------
    optimization_config: dict, optional
        LLM-generated optimization configuration with:
        - 'variables_to_optimize': {var_name: [allowed_values]}
        - 'variables_fixed': {var_name: fixed_value}
    """

    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    var_names = list(config['variable'].keys()) 
    
    # Default empty params if none provided
    if algorithm_params is None:
        algorithm_params = {}
    
    # Create advanced search instance WITH optimization_config
    advanced = AdvancedSearchMethods(
        config, 
        W_values,
        optimization_config=optimization_config  # Pass the LLM config
    )
    
    # Add reference to optimizer
    advanced.optimizer = optimizer
    
    # Convert previous_best to tuple if it's an OTAResult
    prev_best_tuple = None
    if previous_best:
        prev_best_tuple = tuple(
            previous_best.results.get(var) if hasattr(previous_best, "results") 
            else previous_best.get(var) 
            for var in var_names
        )
    
    # Dispatch to appropriate method with parameters
    if method == 'lhs':
        return advanced.latin_hypercube_sampling(
            n_samples, 
            targets, 
            weights,
            seed=algorithm_params.get('seed')
        )
    
    elif method == 'genetic':
        return advanced.genetic_algorithm(
            n_samples, 
            all_previous_results, 
            targets, 
            weights,
            population_size=algorithm_params.get('population_size', 20),
            mutation_rate=algorithm_params.get('mutation_rate', 0.1),
            crossover_rate=algorithm_params.get('crossover_rate', 0.8),
            tournament_size=algorithm_params.get('tournament_size', 3)
        )
    
    elif method == 'annealing':
        return advanced.simulated_annealing(
            n_samples, 
            prev_best_tuple, 
            targets, 
            weights,
            initial_temperature=algorithm_params.get('initial_temperature', 1.0),
            cooling_rate=algorithm_params.get('cooling_rate', 0.95)
        )
    
    elif method == 'adaptive':
        return advanced.adaptive_search(
            n_samples, 
            all_previous_results, 
            prev_best_tuple, 
            targets, 
            weights,
            explore_weight=algorithm_params.get('explore_weight', 0.4),
            exploit_weight=algorithm_params.get('exploit_weight', 0.4),
            random_weight=algorithm_params.get('random_weight', 0.2)
        )
    
    elif method == 'bayesian':
        return advanced.true_bayesian_optimization(
            n_samples,
            W_values, 
            all_previous_results, 
            previous_best,
            targets, 
            weights,
            acquisition_function=algorithm_params.get('acquisition_function', 'EI'),
            exploration_weight=algorithm_params.get('exploration_weight', 0.1)
        )

    elif method == 'optuna':
        return advanced.optuna_bayesian_optimization(
            n_samples,
            W_values,
            all_previous_results,
            previous_best,
            targets,
            weights,
            n_ei_candidates=algorithm_params.get('n_ei_candidates', 30),
            n_startup_trials=algorithm_params.get('n_startup_trials', 5),
            multivariate=algorithm_params.get('multivariate', True),
            prior_weight=algorithm_params.get('prior_weight', 1.0)
        )
    
    elif method == 'multistart':
        return advanced.multi_start_local_search(
            n_samples, 
            all_previous_results, 
            targets, 
            weights,
            n_starts=algorithm_params.get('n_starts', 5),
            search_radius=algorithm_params.get('search_radius', 1)
        )
    
    elif method == 'grid':
        # Grid search with variable-specific ranges
        if optimization_config:
            # Only grid over optimizable variables
            grids = [
                advanced.variables_to_optimize[var] 
                for var in advanced.optimizable_vars
            ]
            
            from itertools import product
            points = []
            for optimizable_values in product(*grids):
                point = advanced._create_full_point(optimizable_values)
                points.append(point)
                if len(points) >= n_samples:
                    return points[:n_samples]
            return points
        else:
            # Original grid search over all variables
            from itertools import product
            points = []
            for combo in product(W_values, repeat=len(var_names)):
                points.append(combo)
                if len(points) >= n_samples:
                    return points[:n_samples]
            return points
    
    elif method == 'refined' and prev_best_tuple:
        # Original refined search (now respects variable ranges)
        return advanced._local_search_around_best(
            prev_best_tuple, 
            n_samples, 
            radius=search_radius or 1
        )
    
    else:  # random
        # Use advanced search's random sampling (respects variable ranges)
        return advanced.random_sampling(n_samples)
