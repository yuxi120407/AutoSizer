"""
Extract useful optimization feedback information from LLM prompt debug file
This extracts key data to help LLM decide on search space regeneration
"""

import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def prepare_format_params(feedback_dict, current_config, netlist, original_config):
    """
    Extract all parameters from feedback_dict, current_config, and original_config
    Returns dict ready for CIRCUIT_REUNDERSTANDING_PROMPT.format(**params)
    
    Parameters:
    -----------
    feedback_dict: dict - Optimization feedback
    current_config: dict - Current optimization configuration
    netlist: str - Circuit netlist
    original_config: dict - Original YAML configuration with all variables and ranges
    """
    
    basic_info = feedback_dict.get('basic_info', {})
    convergence = feedback_dict.get('convergence_analysis', {})
    variable_impact = feedback_dict.get('variable_impact', {})
    issues = feedback_dict.get('search_space_issues', {})
    iteration_history = feedback_dict.get('iteration_history', [])
    stats = feedback_dict.get('statistical_analysis', {})
    
    # =========================================================================
    # ORIGINAL SEARCH SPACE (from YAML config)
    # =========================================================================
    
    # Get original variables from YAML
    original_variables = original_config.get('variable', {})
    available_vars = list(original_variables.keys())
    
    # Get value ranges for each original variable
    original_value_ranges = {}
    for var_name in available_vars:
        # Check for variable-specific values first
        specific_key = f"{var_name}_values"
        if specific_key in original_config:
            original_value_ranges[var_name] = original_config[specific_key]
        elif var_name.startswith('W_') and 'W_values' in original_config:
            original_value_ranges[var_name] = original_config['W_values']
        elif var_name.startswith('L_') and 'L_values' in original_config:
            original_value_ranges[var_name] = original_config['L_values']
        elif var_name.startswith('R_') and 'R_values' in original_config:
            original_value_ranges[var_name] = original_config['R_values']
    
    # Format available variables section
    available_vars_lines = []
    for var_name in available_vars:
        if var_name in original_value_ranges:
            values = original_value_ranges[var_name]
            available_vars_lines.append(f"- {var_name}: {values} ({len(values)} values)")
        else:
            available_vars_lines.append(f"- {var_name}: (no range defined)")
    available_variables = '\n'.join(available_vars_lines)
    
    # Format value ranges section
    value_ranges_lines = []
    for var_name, values in original_value_ranges.items():
        value_ranges_lines.append(f"- {var_name}: {values}")
    value_ranges_str = '\n'.join(value_ranges_lines) if value_ranges_lines else "No ranges defined"
    
    # Get fixed parameters (params that are NOT in variable:)
    original_params = original_config.get('params', {})
    fixed_params_lines = []
    for param_name, param_value in original_params.items():
        fixed_params_lines.append(f"- {param_name}: {param_value} (FIXED, cannot be optimized)")
    fixed_parameters = '\n'.join(fixed_params_lines) if fixed_params_lines else "None"
    
    # =========================================================================
    # CURRENT SEARCH SPACE (from optimization config)
    # =========================================================================
    
    # Current optimized variables
    optimized_vars_section = ""
    optimized_vars_details = []
    if current_config and 'variables_to_optimize' in current_config:
        for var, var_info in current_config['variables_to_optimize'].items():
            if isinstance(var_info, dict):
                values = var_info.get('search_space', [])
                rationale = var_info.get('rationale', 'N/A')
                
                # Check if values are subset of original
                if var in original_value_ranges:
                    original_values = original_value_ranges[var]
                    is_subset = all(v in original_values for v in values)
                    subset_status = "✓" if is_subset else "⚠️ WARNING: Values outside original range!"
                else:
                    subset_status = "⚠️ WARNING: Variable not in original config!"
                
                optimized_vars_section += f"\n• {var}: {len(values)} choices from {values} {subset_status}"
                optimized_vars_section += f"\n  Rationale: {rationale}"
                
                optimized_vars_details.append({
                    'name': var,
                    'values': values,
                    'count': len(values),
                    'is_subset': subset_status.startswith('✓')
                })
            else:
                # Old format: just list of values
                values = var_info
                optimized_vars_section += f"\n• {var}: {len(values)} choices from {values}"
                optimized_vars_details.append({
                    'name': var,
                    'values': values,
                    'count': len(values)
                })
    else:
        optimized_vars_section = "(None)"
    
    # Current fixed variables
    fixed_vars_section = ""
    fixed_vars_details = []
    if current_config and 'variables_fixed' in current_config:
        for var, var_info in current_config['variables_fixed'].items():
            if isinstance(var_info, dict):
                value = var_info.get('fixed_value', 'N/A')
                rationale = var_info.get('rationale', 'N/A')
                
                # Check if value is in original range
                if var in original_value_ranges:
                    original_values = original_value_ranges[var]
                    is_valid = value in original_values
                    valid_status = "✓" if is_valid else "⚠️ WARNING: Value outside original range!"
                else:
                    valid_status = "⚠️ WARNING: Variable not in original config!"
                
                fixed_vars_section += f"\n• {var}: Fixed at {value} {valid_status}"
                fixed_vars_section += f"\n  Rationale: {rationale}"
                
                fixed_vars_details.append({
                    'name': var,
                    'value': value,
                    'is_valid': valid_status.startswith('✓')
                })
            else:
                # Old format: just value
                value = var_info
                fixed_vars_section += f"\n• {var}: {value}"
                fixed_vars_details.append({
                    'name': var,
                    'value': value
                })
    else:
        fixed_vars_section = "(None)"
    
    # Calculate current search space size
    current_search_space_size = 1
    for var_detail in optimized_vars_details:
        current_search_space_size *= var_detail['count']
    
    # Calculate original search space size
    original_search_space_size = 1
    for var_name, values in original_value_ranges.items():
        original_search_space_size *= len(values)
    
    # Search space comparison
    if current_search_space_size > 0 and original_search_space_size > 0:
        reduction_factor = original_search_space_size / current_search_space_size
        search_space_comparison = (
            f"Original: {original_search_space_size:,} combinations\n"
            f"Current:  {current_search_space_size:,} combinations\n"
            f"Reduction: {reduction_factor:.1f}× smaller"
        )
    else:
        search_space_comparison = "Unable to calculate"
    
    # =========================================================================
    # OPTIMIZATION RESULTS
    # =========================================================================
    
    # Performance progression
    progression_section = ""
    if iteration_history:
        improvements = convergence.get('improvements', [])
        for i, iter_data in enumerate(iteration_history):
            improvement = f" (+{improvements[i-1]:.1f}%)" if i > 0 and i-1 < len(improvements) else ""
            progression_section += f"\nIter {iter_data['iteration']} [{iter_data['method']}]: {iter_data['best_fom']:.4f}{improvement}"
    else:
        progression_section = "(No data)"
    
    # Variable impact
    impact_section = ""
    if variable_impact:
        for var, analysis in variable_impact.items():
            impact_section += f"\n\n{var}:"
            
            if 'most_common_in_top' in analysis:
                common = analysis['most_common_in_top']
                impact_section += f"\n  Top designs use: " + ", ".join([f"{val} ({cnt}x)" for val, cnt in common[:3]])
            
            if 'boundary_clustering' in analysis:
                boundary = analysis['boundary_clustering']
                at_min = boundary['at_min_boundary']
                at_max = boundary['at_max_boundary']
                total = boundary['total_top_designs']
                
                if at_min >= total * 0.4:
                    impact_section += f"\n  ⚠️ {at_min}/{total} at LOWER boundary → Consider expanding range"
                if at_max >= total * 0.4:
                    impact_section += f"\n  ⚠️ {at_max}/{total} at UPPER boundary → Consider expanding range"
            
            if stats:
                last_iter = max(stats.keys(), key=int)
                if 'parameter_distributions' in stats[last_iter]:
                    param_dist = stats[last_iter]['parameter_distributions'].get(var, {})
                    total_designs = stats[last_iter].get('total_designs', 1)
                    if param_dist and max(param_dist.values()) == total_designs:
                        dominant_value = [k for k, v in param_dist.items() if v == max(param_dist.values())][0]
                        impact_section += f"\n  ⚠️ 100% use {dominant_value} → Low diversity"
    else:
        impact_section = "(No data)"
    
    # Issues
    issues_section = ""
    if issues and issues.get('issues'):
        for issue in issues['issues']:
            issues_section += f"\n• [{issue['severity'].upper()}] {issue['description']}"
    else:
        issues_section = "None"
    
    # Top designs
    top_designs_section = ""
    if iteration_history:
        all_designs = []
        for iter_data in iteration_history:
            all_designs.extend(iter_data.get('top_designs', []))
        all_designs.sort(key=lambda x: x['fom'], reverse=True)
        
        for i, design in enumerate(all_designs[:10], 1):
            params_str = ", ".join([f"{k}={v}" for k, v in design['parameters'].items()])
            top_designs_section += f"\n{i}. FOM={design['fom']:.4f} → {params_str}"
    else:
        top_designs_section = "(No data)"
    
    # =========================================================================
    # RETURN ALL PARAMETERS
    # =========================================================================
    
    return {
        # Circuit info
        'netlist': netlist,
        
        # Original search space (constraints)
        'available_variables': available_variables,
        'fixed_parameters': fixed_parameters,
        'value_ranges': value_ranges_str,
        'original_search_space_size': f"{original_search_space_size:,}",
        
        # Current search space
        'optimized_vars_section': optimized_vars_section,
        'fixed_vars_section': fixed_vars_section,
        'current_search_space': f"{current_search_space_size:,}",
        'search_space_comparison': search_space_comparison,
        
        # Optimization results
        'progression_section': progression_section,
        'convergence_status': convergence.get('status', 'unknown').upper(),
        'convergence_reason': convergence.get('reason', 'N/A'),
        'best_fom': iteration_history[-1]['best_fom'] if iteration_history else 'N/A',
        'stagnant': 'Yes' if convergence.get('stagnant') else 'No',
        'impact_section': impact_section,
        'issues_section': issues_section,
        'top_designs_section': top_designs_section,
        
        # Iteration info
        'iterations_completed': basic_info.get('iterations_completed', 'N/A'),
        'total_designs': basic_info.get('total_designs', 'N/A'),
        'target_metric': basic_info.get('target_metric', 'FOM'),
    }

# def prepare_format_params(feedback_dict, current_config, netlist, original_config):
#     """
#     Extract all parameters from feedback_dict and current_config
#     Returns dict ready for CIRCUIT_REUNDERSTANDING_PROMPT.format(**params)
#     """
    
#     basic_info = feedback_dict.get('basic_info', {})
#     convergence = feedback_dict.get('convergence_analysis', {})
#     variable_impact = feedback_dict.get('variable_impact', {})
#     issues = feedback_dict.get('search_space_issues', {})
#     iteration_history = feedback_dict.get('iteration_history', [])
#     stats = feedback_dict.get('statistical_analysis', {})
    
#     # Optimized variables
#     optimized_vars_section = ""
#     if current_config and 'variables_to_optimize' in current_config:
#         for var, values in current_config['variables_to_optimize'].items():
#             optimized_vars_section += f"\n• {var}: {len(values)} choices → {values}"
#     else:
#         optimized_vars_section = "(None)"
    
#     # Fixed variables
#     fixed_vars_section = ""
#     if current_config and 'variables_fixed' in current_config:
#         for var, value in current_config['variables_fixed'].items():
#             fixed_vars_section += f"\n• {var}: {value}"
#     else:
#         fixed_vars_section = "(None)"
    
#     # Performance progression
#     progression_section = ""
#     if iteration_history:
#         improvements = convergence.get('improvements', [])
#         for i, iter_data in enumerate(iteration_history):
#             improvement = f" (+{improvements[i-1]:.1f}%)" if i > 0 and i-1 < len(improvements) else ""
#             progression_section += f"\nIter {iter_data['iteration']} [{iter_data['method']}]: {iter_data['best_fom']:.4f}{improvement}"
#     else:
#         progression_section = "(No data)"
    
#     # Variable impact
#     impact_section = ""
#     if variable_impact:
#         for var, analysis in variable_impact.items():
#             impact_section += f"\n\n{var}:"
            
#             if 'most_common_in_top' in analysis:
#                 common = analysis['most_common_in_top']
#                 impact_section += f"\n  Top designs use: " + ", ".join([f"{val} ({cnt}x)" for val, cnt in common[:3]])
            
#             if 'boundary_clustering' in analysis:
#                 boundary = analysis['boundary_clustering']
#                 at_min = boundary['at_min_boundary']
#                 at_max = boundary['at_max_boundary']
#                 total = boundary['total_top_designs']
                
#                 if at_min >= total * 0.4:
#                     impact_section += f"\n  {at_min}/{total} at LOWER boundary"
#                 if at_max >= total * 0.4:
#                     impact_section += f"\n  {at_max}/{total} at UPPER boundary"
            
#             if stats:
#                 last_iter = max(stats.keys(), key=int)
#                 if 'parameter_distributions' in stats[last_iter]:
#                     param_dist = stats[last_iter]['parameter_distributions'].get(var, {})
#                     total_designs = stats[last_iter].get('total_designs', 1)
#                     if param_dist and max(param_dist.values()) == total_designs:
#                         dominant_value = [k for k, v in param_dist.items() if v == max(param_dist.values())][0]
#                         impact_section += f"\n 100% use {dominant_value}"
#     else:
#         impact_section = "(No data)"
    
#     # Issues
#     issues_section = ""
#     if issues and issues.get('issues'):
#         for issue in issues['issues']:
#             issues_section += f"\n• [{issue['severity'].upper()}] {issue['description']}"
#     else:
#         issues_section = "None"
    
#     # Top designs
#     top_designs_section = ""
#     if iteration_history:
#         all_designs = []
#         for iter_data in iteration_history:
#             all_designs.extend(iter_data.get('top_designs', []))
#         all_designs.sort(key=lambda x: x['fom'], reverse=True)
        
#         for i, design in enumerate(all_designs[:10], 1):
#             params_str = ", ".join([f"{k}={v}" for k, v in design['parameters'].items()])
#             top_designs_section += f"\n{i}. FOM={design['fom']:.4f} → {params_str}"
#     else:
#         top_designs_section = "(No data)"
    
#     return {
#         'iterations_completed': basic_info.get('iterations_completed', 'N/A'),
#         'total_designs': basic_info.get('total_designs', 'N/A'),
#         'netlist': netlist,
#         'optimized_vars_section': optimized_vars_section,
#         'fixed_vars_section': fixed_vars_section,
#         'current_search_space': current_config.get('search_space_summary', {}).get('reduced_search_space', 'N/A') if current_config else 'N/A',
#         'progression_section': progression_section,
#         'convergence_status': convergence.get('status', 'unknown').upper(),
#         'convergence_reason': convergence.get('reason', 'N/A'),
#         'best_fom': iteration_history[-1]['best_fom'] if iteration_history else 'N/A',
#         'stagnant': 'Yes' if convergence.get('stagnant') else 'No',
#         'impact_section': impact_section,
#         'issues_section': issues_section,
#         'top_designs_section': top_designs_section,
#         'width_values': basic_info.get('width_values', []),
#         'target_metric': basic_info.get('target_metric', 'composite')
#     }



def extract_useful_section(text):
    start_marker = "## USER SPECIFICATIONS AND GOALS"
    end_marker = "## 🚀 ADVANCED SEARCH METHODS AVAILABLE"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx].strip()
    return None

    
def extract_optimization_feedback(prompt_file_path: str) -> Dict:
    """
    Extract all useful information from LLM prompt file for search space regeneration
    
    Returns a structured dictionary with:
    - Current state summary
    - Iteration history with convergence analysis
    - Variable impact analysis
    - Statistical summaries
    - Recommendations for search space adaptation
    """
    
    with open(prompt_file_path, 'r') as f:
        content = f.read()
    
    feedback = {}
    
    # ========================================================================
    # 1. EXTRACT BASIC STATE INFO
    # ========================================================================
    feedback['basic_info'] = extract_basic_info(content)
    
    # ========================================================================
    # 2. EXTRACT ITERATION HISTORY
    # ========================================================================
    feedback['iteration_history'] = extract_iteration_history(content)
    
    # ========================================================================
    # 3. EXTRACT STATISTICAL ANALYSIS
    # ========================================================================
    feedback['statistical_analysis'] = extract_statistical_analysis(content)
    
    # ========================================================================
    # 4. ANALYZE CONVERGENCE
    # ========================================================================
    feedback['convergence_analysis'] = analyze_convergence(feedback['iteration_history'])
    
    # ========================================================================
    # 5. ANALYZE VARIABLE IMPACT
    # ========================================================================
    feedback['variable_impact'] = analyze_variable_impact(
        feedback['iteration_history'], 
        feedback['statistical_analysis']
    )
    
    # ========================================================================
    # 6. DETECT SEARCH SPACE ISSUES
    # ========================================================================
    feedback['search_space_issues'] = detect_search_space_issues(
        feedback['variable_impact'],
        feedback['iteration_history']
    )
    
    # ========================================================================
    # 7. GENERATE RECOMMENDATIONS
    # ========================================================================
    feedback['recommendations'] = generate_recommendations(feedback)
    
    return feedback


def extract_basic_info(content: str) -> Dict:
    """Extract basic state information"""
    info = {}
    
    # Extract total designs searched
    match = re.search(r'Total designs searched:\s*(\d+)', content)
    if match:
        info['total_designs'] = int(match.group(1))
    
    # Extract iterations completed
    match = re.search(r'Iterations completed:\s*(\d+)', content)
    if match:
        info['iterations_completed'] = int(match.group(1))
    
    # Extract budget status
    match = re.search(r'Budget status.*?:\s*(.+?)(?:\n|$)', content)
    if match:
        info['budget_status'] = match.group(1).strip()
    
    # Extract width values
    match = re.search(r'Width values:\s*\[([^\]]+)\]', content)
    if match:
        info['width_values'] = [float(x.strip()) for x in match.group(1).split(',')]
    
    # Extract target metric
    match = re.search(r'\*\*TARGET METRIC:\s*(.+?)\*\*', content)
    if match:
        info['target_metric'] = match.group(1).strip()
    
    return info


def extract_iteration_history(content: str) -> List[Dict]:
    """Extract iteration-by-iteration results"""
    iterations = []
    
    # Find all iteration blocks
    iter_pattern = r'\*\*Iteration\s+(\d+)\*\*\s+\[(\w+)\]:\s*\n- Designs searched:\s*(\d+)\n- Algorithm parameters:\s*(.+?)\n- Pre-layout.*?:\s*([\d.]+)'
    
    for match in re.finditer(iter_pattern, content):
        iter_num = int(match.group(1))
        method = match.group(2)
        designs_searched = int(match.group(3))
        params_str = match.group(4)
        best_fom = float(match.group(5))
        
        # Extract top 5 designs for this iteration
        top_designs = extract_top_designs(content, iter_num)
        
        iteration_data = {
            'iteration': iter_num,
            'method': method,
            'designs_searched': designs_searched,
            'parameters': parse_params(params_str),
            'best_fom': best_fom,
            'top_designs': top_designs
        }
        
        iterations.append(iteration_data)
    
    return iterations

def extract_top_designs(content: str, iter_num: int) -> List[Dict]:
    """Extract top designs from an iteration - works for any circuit type"""
    designs = []
    
    # Find the iteration block
    iter_start = content.find(f'**Iteration {iter_num}**')
    if iter_start == -1:
        print(f"  ⚠️ Could not find '**Iteration {iter_num}**' in content")
        return designs
    
    # Find next iteration or end
    next_iter = content.find(f'**Iteration {iter_num + 1}**', iter_start)
    if next_iter == -1:
        next_iter = len(content)
    
    iter_block = content[iter_start:next_iter]
    
    # Find "Top N designs" section - ALLOW LEADING WHITESPACE
    top_section_match = re.search(
        r'\s*\*\*Top \d+ designs found.*?\[Pre-layout rankings\]:\*\*\s*\n(.*?)(?=\n\n\*\*|\Z)',
        iter_block,
        re.DOTALL
    )
    
    if not top_section_match:
        print(f"\n{'='*70}")
        print(f"⚠️ NO 'Top N designs' SECTION FOUND for iteration {iter_num}")
        print(f"{'='*70}")
        print("Full iteration block content:")
        print(iter_block)
        print(f"{'='*70}\n")
        return designs
    
    designs_text = top_section_match.group(1)
    print(f"\n=== Extracted designs text for iteration {iter_num} ===")
    print(designs_text[:500])
    print("=" * 50)
    
    # Parse design lines - they have leading spaces
    for line in designs_text.split('\n'):
        line = line.strip()  # Remove leading/trailing spaces
        if not line:
            continue
        
        # Parse: "1. FOM=0.024  (W_pmos_base=1.05, W_nmos_base=0.84 | dc_gain_db=...)"
        match = re.match(r'^(\d+)\.\s+FOM=([\d.]+)\s+\((.*?)\)', line)
        if not match:
            continue
        
        rank = int(match.group(1))
        fom = float(match.group(2))
        rest = match.group(3)
        
        # Split by '|' to separate parameters from metrics
        if '|' in rest:
            params_part = rest.split('|')[0]
        else:
            params_part = rest
        
        # Parse parameters: "W_pmos_base=1.05, W_nmos_base=0.84"
        params = {}
        for param_match in re.finditer(r'(\w+)=([\d.]+)', params_part):
            key = param_match.group(1)
            val = param_match.group(2)
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val
        
        if params:
            designs.append({
                'rank': rank,
                'fom': fom,
                'parameters': params
            })
    
    # Debug: Confirm extraction
    print(f"  ✓ Extracted {len(designs)} designs for iteration {iter_num}")
    if designs:
        print(f"    Sample: FOM={designs[0]['fom']:.4f}, params={designs[0]['parameters']}")
    else:
        print(f"  ⚠️ No designs extracted - check parsing logic")
        print(f"  Raw designs_text preview: {designs_text[:200]}")
    
    return designs[:10]
    
# def extract_top_designs(content: str, iter_num: int) -> List[Dict]:
#     """Extract top designs from an iteration - works for any circuit type"""
#     designs = []
    
#     # Find the iteration block
#     iter_start = content.find(f'**Iteration {iter_num}**')
#     if iter_start == -1:
#         return designs
    
#     # Find next iteration or end
#     next_iter = content.find(f'**Iteration {iter_num + 1}**', iter_start)
#     if next_iter == -1:
#         next_iter = len(content)
    
#     iter_block = content[iter_start:next_iter]
    
#     # Find "Top N designs" section
#     # Pattern: "[timestamp] INFO: Top 3 designs (by FOM):"
#     top_section_match = re.search(
#         r'Top \d+ designs \(by (?:FOM|fom)\):\s*\n(.*?)(?=\n\n|\n\[|\*\*|\Z)',
#         iter_block,
#         re.DOTALL
#     )
    
#     if not top_section_match:
#         print(f"  ⚠️ No 'Top N designs' section found for iteration {iter_num}")
#         return designs
    
#     designs_text = top_section_match.group(1)
#     print(f"\n=== Extracted designs text for iteration {iter_num} ===")
#     print(designs_text[:500])
#     print("=" * 50)
    
#     # Parse design lines with timestamp prefix
#     # Format: "[2026-01-09 04:33:58] INFO:   1. W_pmos_base=0.84 W_nmos_base=0.84 | dc_gain_db=22.16dB ... fom=0.024"
#     # Or without timestamp: "  1. W_pmos_base=0.84 W_nmos_base=0.84 | ..."
    
#     for line in designs_text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove timestamp prefix if present: "[2026-01-09 04:33:58] INFO:"
#         line = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s+\w+:\s*', '', line)
        
#         # Now parse: "1. W_pmos_base=0.84 W_nmos_base=0.84 | dc_gain_db=22.16dB ... fom=0.024"
#         match = re.match(r'^(\d+)\.\s+(.*)', line)
#         if not match:
#             continue
        
#         rank = int(match.group(1))
#         rest = match.group(2)
        
#         # Split by '|' to separate parameters from metrics
#         if '|' in rest:
#             params_part, metrics_part = rest.split('|', 1)
#         else:
#             params_part = rest
#             metrics_part = ""
        
#         # Extract FOM from metrics part
#         fom = None
#         fom_match = re.search(r'fom=([\d.]+)', metrics_part)
#         if fom_match:
#             fom = float(fom_match.group(1))
        
#         # Parse parameters (before '|')
#         params = {}
#         # Pattern: "W_pmos_base=0.84 W_nmos_base=0.84"
#         for param_match in re.finditer(r'(\w+)=([\d.]+)', params_part):
#             key = param_match.group(1)
#             val = param_match.group(2)
#             try:
#                 params[key] = float(val)
#             except ValueError:
#                 params[key] = val
        
#         if fom is not None and params:
#             designs.append({
#                 'rank': rank,
#                 'fom': fom,
#                 'parameters': params
#             })
    
#     return designs[:10]  # Return up to 10 designs
    
# def extract_top_designs(content: str, iter_num: int) -> List[Dict]:
#     """Extract top 5 designs from an iteration"""
#     designs = []
    
#     # Find the iteration block
#     iter_start = content.find(f'**Iteration {iter_num}**')
#     if iter_start == -1:
#         return designs
    
#     # Find next iteration or end
#     next_iter = content.find(f'**Iteration {iter_num + 1}**', iter_start)
#     if next_iter == -1:
#         next_iter = len(content)
    
#     iter_block = content[iter_start:next_iter]
    
#     # Extract design lines
#     design_pattern = r'(\d+)\.\s+\(dc_gain_db \* ugbw\) / power_dc=([\d.]+)\s+\((.+?)\)'
    
#     for match in re.finditer(design_pattern, iter_block):
#         rank = int(match.group(1))
#         fom = float(match.group(2))
#         params_str = match.group(3)
        
#         # Parse parameters
#         params = {}
#         for param_match in re.finditer(r'(W_\w+)=([\d.]+)', params_str):
#             params[param_match.group(1)] = float(param_match.group(2))
        
#         designs.append({
#             'rank': rank,
#             'fom': fom,
#             'parameters': params
#         })
    
#     return designs[:5]


def parse_params(params_str: str) -> Dict:
    """Parse algorithm parameters string"""
    params = {}
    
    # Split by comma
    for part in params_str.split(','):
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to number
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except:
                # Keep as string
                params[key] = value
    
    return params


def extract_statistical_analysis(content: str) -> Dict:
    """Extract statistical analysis per iteration"""
    stats = {}
    
    # Find statistical analysis section
    stats_start = content.find('###  STATISTICAL ANALYSIS OF SEARCH RESULTS')
    if stats_start == -1:
        return stats
    
    stats_content = content[stats_start:]
    
    # Find all iteration headers more carefully - some have content, some don't
    # Pattern: **Iteration N [METHOD] Statistical Analysis:**
    # Then look for the NEXT line that has content (not just another header)
    
    all_iter_positions = []
    for match in re.finditer(r'\*\*Iteration\s+(\d+)\s+\[(\w+)\]\s+Statistical Analysis:\*\*', stats_content):
        iter_num = int(match.group(1))
        method = match.group(2)
        start_pos = match.end()
        
        # Check if there's actual content after this header (not just whitespace and another header)
        next_content = stats_content[start_pos:start_pos+500]
        
        # Skip if next non-empty line is another header
        lines_after = [l.strip() for l in next_content.split('\n') if l.strip()]
        if lines_after and not lines_after[0].startswith('**Iteration'):
            all_iter_positions.append((iter_num, method, start_pos))
    
    # Process each iteration that has content
    for i, (iter_num, method, start_pos) in enumerate(all_iter_positions):
        # Find end position (next iteration or end)
        if i < len(all_iter_positions) - 1:
            end_pos = all_iter_positions[i + 1][2] - 50  # Back up a bit
        else:
            # Last one - find next section or end
            next_section = stats_content.find('## AVAILABLE SEARCH ALGORITHMS', start_pos)
            end_pos = next_section if next_section != -1 else len(stats_content)
        
        iter_block = stats_content[start_pos:end_pos]
        
        # Extract statistics
        iter_stats = extract_iteration_statistics(iter_block)
        iter_stats['method'] = method
        
        stats[iter_num] = iter_stats
    
    return stats


def extract_iteration_statistics(block: str) -> Dict:
    """Extract statistics from an iteration block"""
    stats = {}
    
    # Extract total designs
    match = re.search(r'- Total designs:\s*(\d+)', block)
    if match:
        stats['total_designs'] = int(match.group(1))
    
    # Extract distribution statistics
    stat_names = ['Mean', 'Median', 'Min', 'Max', 'Std Dev']
    for stat_name in stat_names:
        match = re.search(rf'-\s+{stat_name}:\s*([\d.]+)', block)
        if match:
            stats[stat_name.lower().replace(' ', '_')] = float(match.group(1))
    
    # Extract percentiles
    percentile_names = ['10th', '25th', '50th', '75th', '90th']
    stats['percentiles'] = {}
    for pct in percentile_names:
        match = re.search(rf'-\s+{pct}\s+percentile:\s*([\d.]+)', block)
        if match:
            stats['percentiles'][pct] = float(match.group(1))
    
    # Extract parameter distributions
    stats['parameter_distributions'] = extract_parameter_distributions(block)
    
    return stats


def extract_parameter_distributions(block: str) -> Dict:
    """Extract how each parameter was distributed"""
    distributions = {}
    
    # Pattern: W_in_base values: {1.26: 5, 2.52: 5, ...}
    param_pattern = r'(W_\w+)\s+values:\s*\{([^}]+)\}'
    
    for match in re.finditer(param_pattern, block):
        param_name = match.group(1)
        values_str = match.group(2)
        
        # Parse value:count pairs
        value_counts = {}
        for pair in values_str.split(','):
            if ':' in pair:
                val, count = pair.split(':')
                value_counts[float(val.strip())] = int(count.strip())
        
        distributions[param_name] = value_counts
    
    return distributions


def analyze_convergence(iteration_history: List[Dict]) -> Dict:
    """Analyze convergence trends"""
    analysis = {}
    
    if not iteration_history:
        return analysis
    
    # Extract FOM progression
    fom_progression = [iter_data['best_fom'] for iter_data in iteration_history]
    analysis['fom_progression'] = fom_progression
    
    # Calculate improvements
    improvements = []
    for i in range(1, len(fom_progression)):
        prev_fom = fom_progression[i-1]
        curr_fom = fom_progression[i]
        improvement_pct = ((curr_fom - prev_fom) / prev_fom * 100) if prev_fom > 0 else 0
        improvements.append(improvement_pct)
    
    analysis['improvements'] = improvements
    
    # Convergence detection
    if len(improvements) >= 2:
        recent_improvements = improvements[-2:]
        avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
        
        if avg_recent_improvement < 2.0:
            analysis['status'] = 'converging'
            analysis['reason'] = f'Recent improvements < 2% ({avg_recent_improvement:.2f}%)'
        elif avg_recent_improvement > 10.0:
            analysis['status'] = 'improving'
            analysis['reason'] = f'Strong improvements ({avg_recent_improvement:.2f}%)'
        else:
            analysis['status'] = 'progressing'
            analysis['reason'] = f'Moderate improvements ({avg_recent_improvement:.2f}%)'
    else:
        analysis['status'] = 'early'
        analysis['reason'] = 'Too few iterations to assess'
    
    # Check for stagnation
    if len(fom_progression) >= 3:
        last_3 = fom_progression[-3:]
        if last_3[0] == last_3[1] == last_3[2]:
            analysis['stagnant'] = True
            analysis['stagnant_value'] = last_3[0]
        else:
            analysis['stagnant'] = False
    
    return analysis


def analyze_variable_impact(iteration_history: List[Dict], 
                           statistical_analysis: Dict) -> Dict:
    """Analyze which variables have most impact"""
    impact = {}
    
    # Collect all top designs across iterations
    all_top_designs = []
    for iter_data in iteration_history:
        all_top_designs.extend(iter_data['top_designs'])
    
    if not all_top_designs:
        return impact
    
    # Get variable names
    if all_top_designs:
        var_names = list(all_top_designs[0]['parameters'].keys())
    else:
        return impact
    
    # Analyze each variable
    for var in var_names:
        var_analysis = {}
        
        # Collect values from top designs
        top_values = [design['parameters'][var] for design in all_top_designs[:10]]
        
        # Most common value in top designs
        value_counter = Counter(top_values)
        var_analysis['most_common_in_top'] = value_counter.most_common(3)
        
        # Range used in top designs
        var_analysis['range_in_top'] = (min(top_values), max(top_values))
        
        # Check if clustered at boundaries
        all_possible_values = set()
        for iter_num, stats in statistical_analysis.items():
            if 'parameter_distributions' in stats and var in stats['parameter_distributions']:
                all_possible_values.update(stats['parameter_distributions'][var].keys())
        
        if all_possible_values:
            min_possible = min(all_possible_values)
            max_possible = max(all_possible_values)
            
            # Count designs at boundaries
            at_min = sum(1 for v in top_values if v == min_possible)
            at_max = sum(1 for v in top_values if v == max_possible)
            
            var_analysis['boundary_clustering'] = {
                'at_min_boundary': at_min,
                'at_max_boundary': at_max,
                'total_top_designs': len(top_values)
            }
            
            # Recommendation
            if at_max >= len(top_values) * 0.4:  # 40% at upper boundary
                var_analysis['recommendation'] = 'expand_upper_range'
            elif at_min >= len(top_values) * 0.4:  # 40% at lower boundary
                var_analysis['recommendation'] = 'expand_lower_range'
            elif max(top_values) - min(top_values) < (max_possible - min_possible) * 0.3:
                var_analysis['recommendation'] = 'narrow_range'
            else:
                var_analysis['recommendation'] = 'keep_current'
        
        impact[var] = var_analysis
    
    return impact


def detect_search_space_issues(variable_impact: Dict, 
                               iteration_history: List[Dict]) -> Dict:
    """Detect potential issues with current search space"""
    issues = []
    
    # Check for boundary clustering
    for var, analysis in variable_impact.items():
        if 'boundary_clustering' in analysis:
            boundary = analysis['boundary_clustering']
            total = boundary['total_top_designs']
            
            if boundary['at_max_boundary'] >= total * 0.4:
                issues.append({
                    'type': 'upper_boundary_limit',
                    'variable': var,
                    'severity': 'high',
                    'description': f"{var} has {boundary['at_max_boundary']}/{total} top designs at upper boundary",
                    'action': 'expand_upper_range'
                })
            
            if boundary['at_min_boundary'] >= total * 0.4:
                issues.append({
                    'type': 'lower_boundary_limit',
                    'variable': var,
                    'severity': 'high',
                    'description': f"{var} has {boundary['at_min_boundary']}/{total} top designs at lower boundary",
                    'action': 'expand_lower_range'
                })
    
    # Check for repeated best design
    if len(iteration_history) >= 3:
        last_3_best = [iter_data['best_fom'] for iter_data in iteration_history[-3:]]
        if last_3_best[0] == last_3_best[1] == last_3_best[2]:
            issues.append({
                'type': 'stagnation',
                'severity': 'medium',
                'description': f"Best FOM unchanged for 3 iterations ({last_3_best[0]:.4f})",
                'action': 'expand_search_space_or_change_strategy'
            })
    
    return {'issues': issues, 'count': len(issues)}


def generate_recommendations(feedback: Dict) -> Dict:
    """Generate actionable recommendations for search space adaptation"""
    recommendations = {
        'should_regenerate': False,
        'actions': [],
        'priority': 'low'
    }
    
    # Check convergence status
    convergence = feedback.get('convergence_analysis', {})
    if convergence.get('status') == 'converging':
        recommendations['actions'].append({
            'action': 'consider_stopping',
            'reason': convergence.get('reason', 'Converging')
        })
    
    # Check for issues
    issues = feedback.get('search_space_issues', {})
    if issues.get('count', 0) > 0:
        recommendations['should_regenerate'] = True
        recommendations['priority'] = 'high' if issues['count'] >= 2 else 'medium'
        
        for issue in issues['issues']:
            recommendations['actions'].append({
                'action': issue['action'],
                'variable': issue.get('variable'),
                'reason': issue['description']
            })
    
    # Variable-specific recommendations
    variable_impact = feedback.get('variable_impact', {})
    for var, analysis in variable_impact.items():
        if analysis.get('recommendation') and analysis['recommendation'] != 'keep_current':
            recommendations['should_regenerate'] = True
            recommendations['actions'].append({
                'action': analysis['recommendation'],
                'variable': var,
                'reason': f"Based on top design clustering"
            })
    
    return recommendations


def print_feedback_summary(feedback: Dict):
    """Print a human-readable summary of the feedback"""
    print("\n" + "="*80)
    print("OPTIMIZATION FEEDBACK SUMMARY")
    print("="*80)
    
    # Basic info
    info = feedback.get('basic_info', {})
    print(f"\n📊 Current State:")
    print(f"   Total designs: {info.get('total_designs', 'N/A')}")
    print(f"   Iterations: {info.get('iterations_completed', 'N/A')}")
    print(f"   Budget: {info.get('budget_status', 'N/A')}")
    
    # Convergence
    convergence = feedback.get('convergence_analysis', {})
    print(f"\n📈 Convergence Status: {convergence.get('status', 'unknown').upper()}")
    print(f"   {convergence.get('reason', 'N/A')}")
    if 'fom_progression' in convergence:
        foms = convergence['fom_progression']
        print(f"   FOM progression: {' → '.join(f'{x:.2f}' for x in foms[-5:])}")
    
    # Issues
    issues = feedback.get('search_space_issues', {})
    if issues.get('count', 0) > 0:
        print(f"\n⚠️  Search Space Issues ({issues['count']} found):")
        for issue in issues['issues']:
            print(f"   • [{issue['severity'].upper()}] {issue['description']}")
            print(f"     → Recommended action: {issue['action']}")
    
    # Recommendations
    recs = feedback.get('recommendations', {})
    print(f"\n💡 Recommendations:")
    print(f"   Should regenerate search space: {recs.get('should_regenerate', False)}")
    print(f"   Priority: {recs.get('priority', 'N/A').upper()}")
    if recs.get('actions'):
        print(f"   Actions:")
        for action in recs['actions']:
            var = f" ({action.get('variable')})" if action.get('variable') else ""
            print(f"      • {action['action']}{var}: {action['reason']}")
    
    print("\n" + "="*80)


def save_feedback_json(feedback: Dict, output_path: str):
    """Save feedback as JSON file"""
    with open(output_path, 'w') as f:
        json.dump(feedback, f, indent=2)
    print(f"\n💾 Feedback saved to: {output_path}")

# def check_user_specs_met(design_dict: dict, user_specs: str, verbose: bool = False) -> bool:
#     import re
    
#     if not user_specs:
#         return False
    
#     # Extract all constraints: metric operator value
#     pattern = r'(\w+)\s*([><=]+)\s*([\d.]+)'
#     constraints = re.findall(pattern, user_specs)  # Don't use .lower()! Keeps metric names intact
    
#     if not constraints:
#         return False
    
#     all_met = True
#     for metric_name, operator, value_str in constraints:
#         target = float(value_str)
        
#         # Try to get the metric (check exact name first, then variants)
#         actual = None
#         if metric_name in design_dict:
#             actual = design_dict[metric_name]
#         elif f'{metric_name}_db' in design_dict:
#             actual = design_dict[f'{metric_name}_db']
#         elif f'{metric_name}_dc' in design_dict:
#             actual = design_dict[f'{metric_name}_dc']
        
#         if actual is None:
#             if verbose: 
#                 print(f"❌ {metric_name}: not found in design (looked for: {metric_name}, {metric_name}_db, {metric_name}_dc)")
#             all_met = False
#             continue
        
#         # Evaluate constraint
#         met = False
#         if operator == '>':
#             met = actual > target
#         elif operator == '>=':
#             met = actual >= target
#         elif operator == '<':
#             met = actual < target
#         elif operator == '<=':
#             met = actual <= target
#         else:
#             if verbose:
#                 print(f"⚠️  Unknown operator '{operator}' for {metric_name}")
#             all_met = False
#             continue
        
#         if verbose: 
#             print(f"{'✅' if met else '❌'} {metric_name} {operator} {target}: actual = {actual:.3e}")
        
#         if not met:
#             all_met = False
    
#     return all_met


def check_user_specs_met(design_dict: dict, user_specs: str, verbose: bool = False, fom_only_check: bool = True) -> bool:
    import re
    
    if not user_specs:
        return False
    
    # Extract all constraints: metric operator value
    pattern = r'(\w+)\s*([><=]+)\s*([\d.]+)'
    constraints = re.findall(pattern, user_specs)
    
    if not constraints:
        return False
    
    # Filter to only FOM constraint if requested
    if fom_only_check:
        constraints = [(m, o, v) for m, o, v in constraints if m.lower() == 'fom']
        if not constraints:
            if verbose:
                print(f"⚠️  No FOM constraint found in user specs")
            return False
    
    all_met = True
    for metric_name, operator, value_str in constraints:
        target = float(value_str)
        
        # Try to get the metric (check exact name first, then variants)
        actual = None
        if metric_name in design_dict:
            actual = design_dict[metric_name]
        elif f'{metric_name}_db' in design_dict:
            actual = design_dict[f'{metric_name}_db']
        elif f'{metric_name}_dc' in design_dict:
            actual = design_dict[f'{metric_name}_dc']
        
        if actual is None:
            if verbose: 
                print(f"❌ {metric_name}: not found in design (looked for: {metric_name}, {metric_name}_db, {metric_name}_dc)")
            all_met = False
            continue
        
        # Evaluate constraint
        met = False
        if operator == '>':
            met = actual > target
        elif operator == '>=':
            met = actual >= target
        elif operator == '<':
            met = actual < target
        elif operator == '<=':
            met = actual <= target
        else:
            if verbose:
                print(f"⚠️  Unknown operator '{operator}' for {metric_name}")
            all_met = False
            continue
        
        if verbose: 
            print(f"{'✅' if met else '❌'} {metric_name} {operator} {target}: actual = {actual:.3e}")
        
        if not met:
            all_met = False
    
    return all_met
# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Example usage
    input_file = "/mnt/user-data/uploads/llm_prompt_debug.txt"
    output_file = "/mnt/user-data/outputs/optimization_feedback.json"
    
    print("Extracting optimization feedback...")
    feedback = extract_optimization_feedback(input_file)
    
    # Print summary
    print_feedback_summary(feedback)
    
    # Save to JSON
    save_feedback_json(feedback, output_file)
    
    print("\n✅ Extraction complete!")