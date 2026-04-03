#!/usr/bin/env python3
"""
Agent Prompt
=========================================================================
- objective_selection_prompt
- method_section
- decision_framework_section
- parameter_tuning_section
- response_format_section,
"""

def header_section(user_specs: str, metric_info: dict, available_metrics_text) -> str:

    return f"""You are an expert AI optimization agent for analog circuit design, equipped with state-of-the-art search algorithms.

## USER SPECIFICATIONS AND GOALS
{user_specs}

**TARGET METRIC: {metric_info['name']}** ({metric_info['direction']})

## AVAILABLE METRICS
{available_metrics_text}

## OPTIMIZATION STRATEGY
Consider BOTH pre-layout AND post-PEX performance:
- Pre-layout metrics = Performance ceiling/potential
- Post-PEX metrics = Real-world performance (parasitics included)
- **Post-PEX {metric_info['name']} is the TRUE metric** - pre-layout is only predictive"""
    
def objective_selection_prompt(user_specs: str, available_metrics: list[str]) -> str:
    metrics_block = "\n".join(f"- {m}" for m in available_metrics)
    return f"""You are selecting an objective function for analog circuit optimization.

== AVAILABLE METRICS (use ONLY these exact names) ==
{metrics_block}

== STRICT NAMING RULES ==
- You MUST only use the metric names exactly as listed above.
- Do NOT invent aliases (e.g., use 'dc_gain_db', NOT 'gain_db'; use 'power_dc', NOT 'power_uw'; use 'ugbw', NOT 'ugbw_mhz').
- If the user mentions:
  * "gain"      → use 'dc_gain_db'
  * "bandwidth" → use 'ugbw'
  * "power"     → use 'power_dc'
  * "figure of merit"/"FOM" → use 'fom'
- If a requested metric is NOT in the available list, explain that it is unavailable and STOP (do not produce a formula).

== USER SPECIFICATION ==
{user_specs}

== TASK ==
Analyze the user's goals and construct the best mathematical objective function using ONLY the available metrics.

== ALLOWED FORMS ==
1) Single objective:            metric_a
2) Simple ratio:                metric_a / metric_b
3) Product ratio:               (metric_a * metric_b) / metric_c
4) Weighted combination:        w1*metric_a + w2*metric_b + ...
5) Weighted product:            (metric_a^w1 * metric_b^w2) / metric_c^w3
6) Constraint-based:            maximize/minimize primary subject to constraints

== JSON RESPONSE SCHEMA ==
{{
  "objective_function": "EXACT FORMULA USING ONLY ALLOWED METRICS (e.g., (dc_gain_db * ugbw) / power_dc)",
  "reasoning": "WHY this formulation captures the user's intent",
  "formulation": {{
    "type": "ratio|product_ratio|weighted_difference|weighted_product|constraint",
    "numerator": ["dc_gain_db"],                 // for ratio/product types (use only allowed names)
    "denominator": ["power_dc"],                 // for ratio/product types (use only allowed names)
    "numerator_exponents": [1.0],                // optional (weighted_product)
    "denominator_exponents": [1.0],              // optional (weighted_product)
    "weights": {{"dc_gain_db": 2.0, "power_dc": -0.5}}, // for weighted_difference (keys must be allowed names)
    "primary_metric": "dc_gain_db",              // for constraint type (allowed name)
    "primary_direction": "maximize",             // for constraint type
    "constraints": [{{"metric": "power_dc", "operator": "<", "threshold": 100.0}}]  // for constraint type (metrics must be allowed names)
  }},
  "direction": "maximize|minimize",
  "expected_behavior": "What designs will be favored"
}}

== EXAMPLES (using ONLY allowed names) ==
- "maximize gain and minimize power"
  → objective_function: "dc_gain_db / power_dc", type: "ratio"

- "maximize gain and bandwidth while minimizing power"
  → objective_function: "(dc_gain_db * ugbw) / power_dc", type: "product_ratio"

- "I need high gain, power is less critical"
  → objective_function: "2.0*dc_gain_db - 0.5*power_dc", type: "weighted_difference"

- "maximize gain but keep power under 100µW"
  → type: "constraint", primary_metric: "dc_gain_db", constraints: [{{"metric": "power_dc", "operator": "<", "threshold": 100.0}}]
"""


def methods_section(metric_name: str, direction: str, user_specs: str = None,
                    search_space_size: int = None, n_vars: int = None) -> str:

    if user_specs:
        target_text = f"If the target metric meets or exceeds the user's specified goal: {user_specs}"
    else:
        target_text = "Check if any performance targets specified by the user have been achieved"

    # Dynamically calculate sample size recommendations based on search space
    # Designed for analog circuit AutoSizer: nonlinear, discrete, expensive SPICE simulations
    if search_space_size is not None and n_vars is not None:
        import math
        sqrt_s = math.sqrt(search_space_size)

        # Minimum samples needed for nonlinear model to be meaningful
        # Analog circuits are highly nonlinear: need ~4x variables minimum
        model_min = 4 * n_vars

        # --- EXPLORATION methods (LHS, Genetic, Annealing) ---
        # Need enough to map the nonlinear performance landscape
        # Target ~15-25% of reduced space, hard cap at 50 for expensive simulations
        explore_lo = min(50, max(model_min, int(sqrt_s * 1.5)))   # ~15%, cap 50
        explore_hi = min(50, max(explore_lo + 5, int(sqrt_s * 2.5)))  # ~25%, cap 50

        # --- EXPLOITATION methods (Bayesian, Optuna, Multistart) ---
        # Focused on promising regions, fewer samples needed
        # Target ~8-15% of reduced space, hard cap at 35
        exploit_lo = min(35, max(model_min, int(sqrt_s * 0.8)))   # ~8%, cap 35
        exploit_hi = min(35, max(exploit_lo + 5, int(sqrt_s * 2)))  # ~15%, cap 35

        # --- Per-method ranges ---
        lhs_lo   = explore_lo
        lhs_hi   = explore_hi

        gen_lo   = max(3 * n_vars, explore_lo)   # GA needs population ≥ 3×vars
        gen_hi   = min(50, max(gen_lo + 5, explore_hi))
        gen_late = max(model_min, int(explore_lo * 0.7))  # smaller in later iterations

        bay_lo   = exploit_lo
        bay_hi   = exploit_hi

        opt_lo   = exploit_lo
        opt_hi   = min(30, max(opt_lo + 4, int(exploit_hi * 0.9)))

        ada_lo   = max(model_min, int((explore_lo + exploit_lo) / 2))
        ada_hi   = min(40, max(ada_lo + 5, int((explore_hi + exploit_hi) / 2)))

        ann_lo   = max(model_min, int(explore_lo * 0.85))
        ann_hi   = min(40, max(ann_lo + 5, int(explore_hi * 0.85)))

        mul_lo   = exploit_lo
        mul_hi   = min(35, max(mul_lo + 5, int(exploit_hi * 1.1)))

        space_info = (
            f"\n> **Current search space**: {search_space_size:,} combinations "
            f"({n_vars} variables) — sample sizes below are tuned to this space.\n"
        )
        lhs_size   = f"{lhs_lo}-{lhs_hi} (tuned for {search_space_size:,} combinations)"
        gen_size   = f"{gen_lo}-{gen_hi} (reduce to {gen_late}-{gen_lo} in later iterations)"
        bay_size   = f"{bay_lo}-{bay_hi} (fewer samples sufficient with {search_space_size:,} space)"
        opt_size   = f"{opt_lo}-{opt_hi} (highly efficient for {n_vars}-variable discrete space)"
        ada_size   = f"{ada_lo}-{ada_hi} (balanced for {search_space_size:,} combinations)"
        ann_size   = f"{ann_lo}-{ann_hi} (temperature-adjusted for space size)"
        mul_size   = f"{mul_lo}-{mul_hi} (multiple starts across {search_space_size:,} space)"
    else:
        space_info = ""
        lhs_size   = "15-30 (adjust based on dimensionality of the design space)"
        gen_size   = "20-30 (can reduce to 15-20 in later iterations)"
        bay_size   = "10-20 (fewer samples needed in later iterations)"
        opt_size   = "12-18 (highly efficient)"
        ada_size   = "15-25 (can be reduced as convergence approaches)"
        ann_size   = "12-20 (adjust based on temperature schedule)"
        mul_size   = "12-25"

    return f"""## 🚀 ADVANCED SEARCH METHODS AVAILABLE
{space_info}
You have access to state-of-the-art optimization algorithms. Analyze the current state and choose the most appropriate strategy.

### Method Arsenal:

**1. 'lhs' (Latin Hypercube Sampling)**
- **Strength**: Guarantees uniform space coverage, no blind spots
- **Best when**: Design space is poorly understood, few previous results, need broad exploration
- **Weakness**: Ignores previous results completely
- **Sample size**: {lhs_size}
- **Strategy type**: Pure exploration

**2. 'genetic' (Genetic Algorithm)**
- **Strength**: Evolves toward good regions, robust to local optima, finds multiple solutions
- **Best when**: Want to combine exploration + evolution, searching for diverse good designs
- **Weakness**: Needs many samples, computationally expensive
- **Sample size**: {gen_size}
- **Strategy type**: Evolutionary exploration

**3. 'bayesian' (Bayesian Optimization)**
- **Strength**: MOST sample-efficient, learns from ALL previous data, intelligent sampling
- **Best when**: Have 25+ previous results, want to converge efficiently, budget is limited
- **Weakness**: Can get trapped in local optima if exploration was insufficient
- **Sample size**: {bay_size}
- **Strategy type**: Intelligent exploitation

**4. 'optuna' (Bayesian Optimization - TPE)**
  **Strength**: BEST for discrete/categorical spaces, intelligent unique points, native discrete support, learns from history, escapes local optima better than GP
  **Best when**: Discrete design space, have 15+ previous results, want maximum sample efficiency, previous 'bayesian' gave <5 unique points
  **Weakness**: Requires historical data, slightly more complex parameters
  **Sample size**: {opt_size}
  **Strategy type**: Intelligent exploration + exploitation optimized for discrete spaces

**5. 'adaptive' (Adaptive Multi-Strategy)**
- **Strength**: Self-balancing (40% exploit + 40% explore + 20% random), versatile
- **Best when**: Unsure whether to explore or exploit, want balanced approach
- **Weakness**: Jack of all trades, master of none
- **Sample size**: {ada_size}
- **Strategy type**: Balanced

**6. 'annealing' (Simulated Annealing)**
- **Strength**: Can ESCAPE local optima by accepting worse solutions probabilistically
- **Best when**: Stuck in plateau, suspect local optimum, need to explore distant regions
- **Weakness**: May waste samples exploring bad regions
- **Sample size**: {ann_size}
- **Strategy type**: Escape + exploration

**7. 'multistart' (Multi-Start Local Search)**
- **Strength**: Finds MULTIPLE local optima, provides alternative solutions
- **Best when**: Want to find diverse good designs, compare trade-offs
- **Weakness**: Inefficient if only one dominant optimum
- **Sample size**: {mul_size}
- **Strategy type**: Diversified exploitation



## CRITICAL STOPPING RULES
1. **User specification is the ONLY metric that matters** - Focus on {direction}ing this value
2. **STOP if user specification is met** - {target_text}
3. **STOP if converged <2% over 2 iterations** - Improvements less than 2% indicate convergence
4. **Consider diversity** - Balance exploration and exploitation
5. **Budget wisely** - More samples for exploration, fewer for exploitation
6. **Sample size**: Diverse methods need more samples

**Priority for stopping:**
- First check if user specification/target is met → STOP immediately
- Then check convergence (<2% improvement) → Consider stopping
- Always provide reasoning for your decision

"""

    
def decision_framework_section(metric_name: str) -> str:
    return f"""## 🧠 STRATEGIC ANALYSIS FRAMEWORK
    
    Analyze these key factors for dynamic method selection:
    
    ### Factor 1: Data Availability and Quality
    - **Total designs searched**: Different methods become viable at different sample counts
    - **Previous method performance**: Did recent methods yield improvements?
    - **Parameter space coverage**: Are there unexplored regions worth investigating?
    - **Design diversity**: Are we finding similar or diverse designs?
    
    ### Factor 2: Improvement Trajectory
    - **Absolute improvement rate**: How much is {metric_name['name']} improving per iteration?
      - Strong (>5%): Continue exploration-focused methods
      - Moderate (1-5%): Balance exploration/exploitation
      - Minimal (<1%): Consider escape methods or convergence
    - **Improvement trend**: 
      - Accelerating: Current approach is working, continue or intensify
      - Decelerating: Consider method switching or convergence
    - **Plateau detection**:
      - 1 iteration plateau: Normal variation, continue
      - 2 iteration plateau: Consider method switching
      - 3+ iteration plateau with method diversity: Likely converged, consider stopping
    
    ### Factor 3: Design Space Insights
    - **Parameter sensitivity**: Which parameters most affect {metric_name['name']}?
    - **Constraint analysis**: Are constraints limiting potential improvements?
    - **Edge parameters**: If optimal parameters are at edges of search space:
      - Consider expanding search space in that direction
      - Try methods that can explore beyond current bounds
      - Consider constraint relaxation if appropriate
    
    ### Factor 4: Method Complementarity
    - **Method diversity**: Different methods explore space differently
    - **Complement previous methods**: Choose methods that address limitations of previous approaches
    - **Method sequencing**: Consider LHS → Genetic/Adaptive → Bayesian → Multistart/Refined as general pattern
    
    ### Factor 5: Budget Management
    - **Remaining budget**: Adjust sample counts based on remaining budget
    - **Expected returns**: More samples early, fewer as convergence approaches
    - **Time constraints**: Each iteration costs ~2-3 minutes (ALIGN+PEX)
    
    ## 🎯 DECISION MAKING PROCESS
    
    **Step 1: Diagnose the current situation**
    - Are we exploring (finding new regions)?
    - Are we exploiting (refining known good region)?
    - Are we stuck (plateau with no improvement)?
    - Are we converged (fundamentally at optimum)?
    
    **Step 2: Identify the need**
    - Need broad exploration? → LHS, Genetic, Adaptive
    - Need efficient convergence? → Bayesian, Adaptive
    - Need to escape local optimum? → Annealing, Multistart
    - Need to verify convergence? → Multistart (find alternatives)
    - Ready to stop? → Stop
    
    **Step 3: Choose method and sample size**
    - Match method to current need
    - Choose sample size based on method requirements and budget
    - Consider: exploration methods need MORE samples, exploitation needs FEWER
    
    **Step 4: Justify your choice**
    - Explain WHY this method addresses the current situation
    - What do you EXPECT to happen?
    - How will you KNOW if it worked?"""

def parameter_tuning_section() -> str:
    return """## ALGORITHM PARAMETER TUNING
    
    Based on the statistical analysis above, you can tune algorithm parameters to balance exploration vs exploitation
    CRITICAL: You MUST adapt parameters between iterations based on observations!
        Never use the same acquisition_function + exploration_weight if previous iteration showed problems!
    
    **1. 'lhs' (Latin Hypercube Sampling) parameters:**
    - **seed**: [integer, optional] Controls the random state for deterministic sampling
      - Use different seeds in consecutive iterations to maximize exploration diversity
      - When first using LHS: Use a random seed for broad exploration
      - When returning to LHS after other methods: Use seed to explore unsampled regions
    
    **2. 'genetic' parameters:**
    - **mutation_rate**: [0.05-0.5] Higher values increase exploration
      - Use 0.05-0.1: When performance is steadily improving (fine-tuning)
      - Use 0.2-0.3: For normal operation (balanced)
      - Use 0.4-0.5: When stuck in local optimum or consecutive iterations find same design
    - **crossover_rate**: [0.7-0.9] Higher values increase genetic diversity
      - Use 0.7: When many similar high-performing designs exist
      - Use 0.8: For normal operation
      - Use 0.9: When diversity is low (std dev < 0.01) or to promote innovation
    - **tournament_size**: [2-5] Lower values maintain population diversity
      - Use 2: When design diversity is low (to reduce selection pressure)
      - Use 3: For normal operation
      - Use 4-5: When focusing on the best designs (exploitation phase)
    
    **3. 'bayesian' parameters:**
    
        Bayesian Optimization uses a Gaussian Process (GP) model to intelligently sample the design space.
        It's most effective when you have historical data and want sample-efficient optimization.
        
        **acquisition_function**: Strategy for selecting next points to evaluate
          - **"EI"** (Expected Improvement) - BALANCED
            * Use for: General-purpose optimization, mid-stage iterations (4-6)
            * Balances exploring new regions vs exploiting known good areas
            * Best when: Getting 8+ unique points, steady progress
            * Pairs with: exploration_weight = 0.1-0.3
          
          - **"LCB"** (Lower Confidence Bound) - EXPLORATION-FOCUSED
            * Use for: Discrete spaces, early iterations (2-3), first time trying Bayesian
            * Explores diverse regions by targeting areas with high uncertainty
            * Best when: First Bayesian iteration OR getting 5-7 unique points (moderate success)
            * Pairs with: exploration_weight  = 2.0-2.5
            *  WARNING: If LCB gives <5 unique points, switch to UCB or increase kappa!
          
          - **"UCB"** (Upper Confidence Bound) - AGGRESSIVE EXPLORATION
            * Use for: When LCB failed (<5 unique points), stuck in local optimum
            * Even more exploratory than LCB
            * Best when: Previous Bayesian iteration got <5 unique points, need maximum diversity
            * Pairs with: exploration_weight (kappa) = 2.5-3.5
            * This is your "stuck" escape mechanism - use when LCB didn't work!
          
          - **"PI"** (Probability of Improvement) - EXPLOITATION-FOCUSED
            * Use for: Late-stage refinement (iterations 7+), fine-tuning near optimum
            * Focuses on refining the current best region
            * Best when: Close to convergence, improvements < 1% per iteration, getting 10+ unique points
            * Pairs with: exploration_weight (xi) = 0.01-0.1
            * Only use if confident you're near optimum!
        
        **exploration_weight**: Controls exploration vs exploitation trade-off
          - Parameter name changes based on acquisition function:
            * For EI/PI: This is 'xi' parameter (range 0.0-0.5)
            * For LCB/UCB: This is 'kappa' parameter (range 0.5-5.0)
          
          For EI/PI (xi parameter):
            * 0.0-0.01: Pure exploitation (greedy, refines current best)
            * 0.01-0.1: Low exploration (late-stage refinement)
            * 0.1-0.3: Moderate exploration ( balanced approach)
            * 0.3-0.5: High exploration (escape local optima, increase diversity)
          
          For LCB/UCB (kappa parameter):
            * 0.5-1.5: Low exploration (exploitation-focused) - rarely use
            * 1.96: Statistical (95% confidence interval)
            * 2.0-2.5: High exploration (STARTING point for discrete spaces)
            * 2.5-3.0: Very high exploration (when getting <5 unique points)
            * 3.0-4.0: Maximum exploration (desperate, when stuck)
            * 4.0+: Extreme (rarely needed, almost random)
    
    **4. 'annealing' parameters:**
    - **initial_temperature**: [0.5-3.0] Higher values enable escaping local optima
      - Use 0.5-1.0: For minor refinements when close to optimum
      - Use 1.5-2.0: For normal operation
      - Use 2.5-3.0: After multiple iterations with no improvement
    - **cooling_rate**: [0.8-0.99] Higher values slow the cooling process
      - Use 0.8-0.85: When budget is limited (faster convergence)
      - Use 0.9-0.95: For normal operation
      - Use 0.96-0.99: When thorough search is needed (slow cooling)
    
    **5. 'adaptive' parameters:**
    - **explore_weight**: [0.2-0.7] Weight for exploration component
      - Use 0.2-0.3: When close to convergence
      - Use 0.4-0.5: For balanced operation
      - Use 0.6-0.7: When diversity is low or performance plateaus
    - **exploit_weight**: [0.2-0.7] Weight for exploitation component
      - Use 0.2-0.3: Early in optimization
      - Use 0.4-0.5: For balanced operation
      - Use 0.6-0.7: Late in optimization, refining best solutions
    - **random_weight**: [0.0-0.3] Weight for random sampling
      - Use 0.0-0.1: Late stage optimization
      - Use 0.1-0.2: Normal operation
      - Use 0.2-0.3: When stuck in local optimum
    
    **6. 'multistart' parameters:**
    - **n_starts**: [3-8] Number of starting points
      - Use 3-4: When budget is limited
      - Use 5-6: Normal operation
      - Use 7-8: When seeking diverse alternative solutions
    - **search_radius**: [1-3] Search radius around starting points
      - Use 1: For fine refinement
      - Use 2: Normal operation
      - Use 3: When previous radius didn't yield improvements


      **7. 'optuna' parameters :**

        Optuna TPE (Tree-structured Parzen Estimator) is a modern Bayesian optimization method
        specifically designed for discrete and categorical variables. It outperforms GP-based 
        methods (scikit-optimize) on discrete spaces
        
        **n_ei_candidates**: Exploration strength (10-50)
          Controls how many candidate points TPE evaluates internally before picking the best
          
          - **10-20** (Exploitation Mode):
            * Use when: Late stage (iterations 7+), converging, improvements < 1%
            * Effect: Focuses on refining current best region
            * Expected: 5-8 unique points, concentrated around best
          
          - **25-35** (Balanced Mode):
            * Use when: Mid-stage (iterations 3-6), steady progress
            * Effect: Good balance of exploration and exploitation
            * Expected: 10-13 unique points, diverse but guided
          
          - **40-50** (Exploration Mode):
            * Use when: Early stage (iterations 2-3), stuck/plateaued, need diversity
            * Effect: Aggressive exploration of design space
            * Expected: 13-15 unique points, very diverse
        
        **n_startup_trials**: Random initialization (3-10)
          Number of random trials before TPE modeling starts
          
          - **3-5** (Trust History):
            * Use when: Have good historical data (20+ designs), mid-late stage
            * Effect: Start intelligent sampling quickly
          
          - **6-8** (Moderate Exploration):
            * Use when: Limited history (10-20 designs), uncertain about space
            * Effect: More random exploration before trusting model
          
          - **9-10** (High Random):
            * Use when: Very early (first TPE iteration), little history (<10 designs)
            * Effect: Explore broadly before building model
        
        **multivariate**: Model parameter interactions (True/False)
          Whether to model correlations between different parameter
          
          - **True** :
            * Understands that some parameters are often correlate
            * More accurate model, better suggestions
            * Slightly slower (negligible for 4 parameters)
          
          - **False**:
            * Treats each parameter independently
            * Faster but less accurate
            * Use only if struggling with performance
        
        **prior_weight**: Trust in prior vs observations (0.5-2.0)
          How much to weight prior knowledge versus actual observations
          
          - **0.5-0.8** (Trust Data):
            * Use when: Many observations (30+), data very reliable
            * Effect: Model driven by measurements
          
          - **1.0** (Balanced):
          
          - **1.5-2.0** (Trust Prior):
            * Use when: Few observations (5-15), high measurement noise
            * Effect: Regularizes model, prevents overfitting
        
        ## WHEN TO USE OPTUNA TPE:
        
        ✅ **Use Optuna when:**
        - Discrete/categorical design space (like OTA widths)
        - Previous Bayesian (GP) methods gave < 8 unique points
        - Need 10-15 intelligent points per iteration
        - Iteration 2 or later (have historical data)
        
        ❌ **Don't use Optuna when:**
        - No historical data (iteration 1)
        - Continuous optimization → GP might be fine
        - Very small search space → Random/Grid sufficient

    
    **Parameter Selection Decision Tree:**
    
    1. **If consecutive iterations find identical best design:**
       - Genetic: Increase mutation_rate to 0.4+, decrease tournament_size to 2
       - Bayesian: Switch to acquisition_function="LCB", increase exploration_weight to 0.4+
       - Annealing: Increase initial_temperature to 2.5+
       - Adaptive: Increase explore_weight to 0.6+, random_weight to 0.2+
    
    2. **If performance diversity is low (small std dev < 0.01):**
       - Genetic: Increase mutation_rate, decrease tournament_size
       - Bayesian: Use PI acquisition function with higher exploration_weight
       - Annealing: Increase initial_temperature
       - LHS: Use a new random seed
    
    3. **If steady improvements observed over last 2-3 iterations:**
       - Continue with current method but with balanced parameters
       - Or switch to more exploitative method (e.g., Bayesian)
    
    4. **If approaching convergence (improvements < 2% for 2 iterations):**
       - Genetic: Lower mutation_rate, increase tournament_size
       - Bayesian: Use EI with lower exploration_weight
       - Multistart: Use to verify convergence and find alternatives
       - Adaptive: Increase exploit_weight, decrease random_weight
    
    5. **If starting fresh (iteration 1):**
       - LHS: Random seed, thorough coverage
       - Genetic: High mutation_rate (0.3+), low tournament_size (2)
       - Adaptive: High explore_weight (0.6+)
    """

def response_format_section() -> str:
    
    """Build the response format section with JSON template"""
    return f"""## RESPONSE FORMAT (JSON ONLY!)
    
     **CRITICAL - READ THIS CAREFULLY** 
    
    ### THE "action" FIELD RULES:
    1. **"action" can ONLY be "search" OR "stop"** - These are the ONLY two valid values!
    2. **NEVER put algorithm names in "action"** - Names like "annealing", "bayesian", "genetic" go in "method"!
    3. **"action" and "method" are DIFFERENT fields** - Don't confuse them!
    
    ### TEMPLATE 1: If you want to CONTINUE optimizing (use "search")
    ```json
    {{
      "action": "search",
      "method": "PUT_ALGORITHM_NAME_HERE",
      "n_samples": PUT_NUMBER_HERE,
      "parameters": {{
        // Include only relevant parameters for the chosen method
        // For example, if using genetic:
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "tournament_size": 3
        // Or for bayesian:
        "acquisition_function": "EI",
        "exploration_weight": 0.1
        // See parameter tuning section above for details
      }},
      "reasoning": "YOUR_REASONING_HERE",
      "confidence": "high or medium or low",
      "expected_improvement": "YOUR_EXPECTED_IMPROVEMENT",
      "convergence_assessment": "YOUR_CONVERGENCE_ANALYSIS"
    }}
    ```
    
    Where "method" is ONE of these: lhs, genetic, bayesian, adaptive, annealing, multistart
    
    ### TEMPLATE 2: If you want to STOP optimizing (use "stop")
    ```json
    {{
      "action": "stop",
      "reasoning": "WHY_STOPPING",
      "confidence": "high or medium or low",
      "expected_improvement": "N/A - converged",
      "convergence_assessment": "CONVERGENCE_EVIDENCE"
    }}
    ```
    
    Notice: NO "method" field and NO "n_samples" field when stopping!
    
    ###  REQUIREMENTS SUMMARY:
    
    1. **"action"** must be EXACTLY "search" or "stop" (lowercase, no other values)
    2. **If "search"**: Include "method" and "n_samples" (10-35)
    3. **"parameters"** field : Include algorithm-specific parameters based on the parameter tuning section
    4. **If "stop"**: Do NOT include "method", "n_samples", or "parameters" fields
    5. **Respond with ONLY the JSON object** - no explanatory text before or after
    6. **Base your decision on data analysis** - not rigid phase rules
    
     **Now make your decision using the correct format:**"""
