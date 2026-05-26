#!/usr/bin/env python3
"""
Controlled Iterative OTA Optimization
=====================================
Step-by-step optimization with configurable search sizes per iteration
"""

import subprocess
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import random
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence, List, Optional, Mapping, Dict, Any
from advanced_search_methods_test import enhanced_generate_search_points, AdvancedSearchMethods
import yaml, string
import math
from circuit_sim.vco_characterization import simulate_vco
from circuit_sim.switched_capacitor_sim import simulate_switched_capacitor
from circuit_sim.bandgap_reference_sim import simulate_bandgap_reference
from circuit_sim.ldo_regulator_sim import simulate_ldo_regulator
from circuit_sim.nand_gate_sim import simulate_nand_gate
from circuit_sim.xor_gate_sim import simulate_xor_gate
from circuit_sim.resistive_load_amp_sim import simulate_resistive_load_amp
from circuit_sim.diode_load_amp_sim import simulate_diode_load_amp

from circuit_sim.ring_osc_sim import simulate_ring_oscillator
from circuit_sim.sim_hpf import simulate_folder_cascode_ota_with_hpf
from circuit_sim.sim_lpf import simulate_folder_cascode_ota_with_lpf
from circuit_sim.sim_bandpf import simulate_folder_cascode_ota_with_bpf

def aggregate_trial_metrics(results_dir, n_trials):
    """Aggregate metrics across all trials"""
    all_metrics = []
    
    for trial_idx in range(n_trials):
        #summary_file = results_dir / f"optimization_summary_trial_{trial_idx}.json"
        summary_file = os.path.join(results_dir, f"optimization_summary_trial_{trial_idx}.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
                all_metrics.append(data['metrics'])
    
    if not all_metrics:
        return None
    
    # Calculate averages and std
    import numpy as np
    
    best_foms = [m['best_fom'] for m in all_metrics]
    evals_to_best = [m['evals_to_best'] for m in all_metrics]
    times = [m['time_to_best_seconds'] for m in all_metrics]
    successes = [m['success'] for m in all_metrics]
    
    return {
        'avg_best_fom': np.mean(best_foms),
        'std_best_fom': np.std(best_foms),
        'avg_evals_to_best': np.mean(evals_to_best),
        'std_evals_to_best': np.std(evals_to_best),
        'avg_time_seconds': np.mean(times),
        'std_time_seconds': np.std(times),
        'success_rate_percent': (sum(successes) / len(successes)) * 100
    }

def parse_user_specs(user_specs_metric):
    """Parse user_specs_metric string into list of constraints"""
    import re
    constraints = []
    parts = user_specs_metric.split(' AND ')
    
    for part in parts:
        part = part.strip()
        # Match patterns like "fom > 1.5" or "power_dc < 50.0"
        match = re.match(r'(\w+)\s*([<>=]+)\s*([\d.e+-]+)', part)
        if match:
            metric, operator, value = match.groups()
            constraints.append({
                'metric': metric,
                'operator': operator,
                'target': float(value)
            })
    
    return constraints


def calculate_constraint_satisfaction_score(design_dict, user_specs_metric):
    """
    Calculate a score based on how well constraints are satisfied
    Returns: (num_satisfied, total_violation_score)
    """
    constraints = parse_user_specs(user_specs_metric)
    
    num_satisfied = 0
    total_violation = 0
    
    for constraint in constraints:
        metric = constraint['metric']
        target = constraint['target']
        op = constraint['operator']
        actual = design_dict.get(metric)
        
        if actual is None:
            total_violation += 1e6  # Large penalty for missing metrics
            continue
        
        satisfied = False
        violation = 0
        
        if op == '>':
            satisfied = actual > target
            violation = max(0, target - actual) / abs(target) if not satisfied else 0
        elif op == '<':
            satisfied = actual < target
            violation = max(0, actual - target) / abs(target) if not satisfied else 0
        elif op == '>=':
            satisfied = actual >= target
            violation = max(0, target - actual) / abs(target) if not satisfied else 0
        elif op == '<=':
            satisfied = actual <= target
            violation = max(0, actual - target) / abs(target) if not satisfied else 0
        
        if satisfied:
            num_satisfied += 1
        else:
            total_violation += violation
    
    return num_satisfied, total_violation


def multi_objective_sort_key_hybrid(design, user_specs_metric):
    """
    Hybrid ranking: hard constraint on satisfaction + soft optimization of FOM
    
    Priority 1: All specs met? (binary: yes=1, no=0)
    Priority 2 (if no): How many specs met + how bad violations
    Priority 3: Maximize FOM
    """
    design_dict = design.to_dict() if hasattr(design, 'to_dict') else design
    num_satisfied, violation = calculate_constraint_satisfaction_score(design_dict, user_specs_metric)
    fom = getattr(design, 'fom', 0) if hasattr(design, 'fom') else design.get('fom', 0)
    
    if fom is None:
        fom = 0
    
    total_constraints = len(parse_user_specs(user_specs_metric))
    
    # Critical: Does it meet ALL constraints?
    all_satisfied = (num_satisfied == total_constraints)
    
    if all_satisfied:
        # If all specs met, rank purely by FOM (optimize performance)
        return (1, 0, fom)  # (all_met=1, violation=0, fom)
    else:
        # If specs not met, rank by how close we are (feasibility)
        # Prioritize: more specs met > less violation > higher FOM
        feasibility_score = num_satisfied / max(total_constraints, 1) - violation * 0.1
        return (0, feasibility_score, fom)  # (all_met=0, feasibility, fom)


def single_objective_sort_key_fom(design):
    """
    Original FOM-only ranking (for ablation comparison)
    """
    fom = getattr(design, 'fom', 0) if hasattr(design, 'fom') else design.get('fom', 0)
    if fom is None:
        fom = 0
    return fom
    
# @dataclass
# class OTAResult:
#     """Generic container for circuit simulation results."""
#     results: Dict[str, Any] = field(default_factory=dict)
#     area: Optional[float] = None
#     fom_per_area: Optional[float] = None
    
#     # Add this method to allow attribute access
#     def __getattr__(self, name):
#         """Allow accessing results dict keys as attributes."""
#         if name in ('results', 'area', 'fom_per_area'):
#             # These are actual dataclass fields, use default behavior
#             return object.__getattribute__(self, name)
#         # Try to get from results dict
#         try:
#             return self.results[name]
#         except KeyError:
#             raise AttributeError(f"'OTAResult' object has no attribute '{name}'")
    
#     # --- Methods ---
#     def calculate_fom_per_area(self):
#         """Compute FOM/Area if both values exist."""
#         fom = self.results.get("fom")
#         if fom is not None and self.area:
#             self.fom_per_area = (fom / self.area) * 100
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Return all stored values as a dict."""
#         d = dict(self.results)
#         d["area"] = self.area
#         d["fom_per_area"] = self.fom_per_area
#         return d
    
#     def __str__(self) -> str:
#         """Pretty summary string — adapt automatically to whatever keys exist."""
#         lines = []
#         # Design variables (widths, params, etc.)
#         var_keys = [k for k in self.results.keys() if "base" in k or k.lower() in ("l", "ibias")]
#         if var_keys:
#             vars_str = " ".join(f"{k}={self.results[k]:.2f}" for k in var_keys)
#             lines.append(vars_str)
#         # Metrics (all other numeric keys)
#         metric_keys = [k for k in self.results.keys() if k not in var_keys]
#         if metric_keys:
#             metrics_str = " ".join(f"{k}={(self.results[k] if self.results[k] is not None else 0):.4f}" for k in metric_keys)
#             lines.append(metrics_str)
#         # Area info
#         if self.area is not None:
#             area_str = f"area={self.area:.2f}µm²"
#             if self.fom_per_area is not None:
#                 area_str += f" FOM/Area={self.fom_per_area:.4f}"
#             lines.append(area_str)
#         return " | ".join(lines) if lines else "Empty OTAResult"


@dataclass
class OTAResult:
    """Generic container for circuit simulation results."""
    results: Dict[str, Any] = field(default_factory=dict)
    area: Optional[float] = None
    fom_per_area: Optional[float] = None
    
    def __getattr__(self, name):
        """Allow accessing results dict keys as attributes."""
        if name in ('results', 'area', 'fom_per_area'):
            return object.__getattribute__(self, name)
        try:
            return self.results[name]
        except KeyError:
            raise AttributeError(f"'OTAResult' object has no attribute '{name}'")
    
    def calculate_fom_per_area(self):
        """Compute FOM/Area if both values exist."""
        fom = self.results.get("fom")
        if fom is not None and self.area:
            self.fom_per_area = (fom / self.area) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all stored values as a dict."""
        d = dict(self.results)
        d["area"] = self.area
        d["fom_per_area"] = self.fom_per_area
        return d
    
    def __str__(self) -> str:
        """Pretty summary string — adapt automatically to whatever keys exist."""
        lines = []
        
        # Design variables (widths, params, etc.)
        var_keys = [k for k in self.results.keys() if "base" in k or k.lower() in ("l", "ibias")]
        if var_keys:
            var_parts = []
            for k in var_keys:
                value = self.results[k]
                if value is None:
                    var_parts.append(f"{k}=None")
                elif isinstance(value, str):
                    var_parts.append(f"{k}={value}")
                else:
                    try:
                        var_parts.append(f"{k}={float(value):.2f}")
                    except:
                        var_parts.append(f"{k}={value}")
            if var_parts:
                vars_str = " ".join(var_parts)
                lines.append(vars_str)
        
        # Metrics (all other keys) - HANDLE ALL TYPES SAFELY
        metric_keys = [k for k in self.results.keys() if k not in var_keys]
        if metric_keys:
            metric_parts = []
            for k in metric_keys:
                value = self.results[k]
                
                # Handle None
                if value is None:
                    metric_parts.append(f"{k}=None")
                
                # Handle strings (like "NA")
                elif isinstance(value, str):
                    metric_parts.append(f"{k}={value}")
                
                # Handle boolean
                elif isinstance(value, bool):
                    metric_parts.append(f"{k}={value}")
                
                # Handle numbers (int or float)
                elif isinstance(value, (int, float)):
                    try:
                        # Convert to float and format with 4 decimals
                        metric_parts.append(f"{k}={float(value):.4f}")
                    except (ValueError, TypeError, OverflowError):
                        # If conversion/formatting fails, use default string
                        metric_parts.append(f"{k}={value}")
                
                # Handle any other type
                else:
                    metric_parts.append(f"{k}={str(value)}")
            
            if metric_parts:
                metrics_str = " ".join(metric_parts)
                lines.append(metrics_str)
        
        # Area info
        if self.area is not None:
            try:
                area_str = f"area={float(self.area):.2f}µm²"
                if self.fom_per_area is not None:
                    area_str += f" FOM/Area={float(self.fom_per_area):.4f}"
                lines.append(area_str)
            except:
                lines.append(f"area={self.area}")
        
        return " | ".join(lines) if lines else "Empty OTAResult"

@dataclass
class IterationResult:
    """Store results from one complete iteration"""
    iteration: int
    pre_layout: OTAResult
    post_pex: Optional[OTAResult]
    gds_path: Optional[Path]
    pex_netlist_path: Optional[Path]
    degradation_percent: Optional[dict]
    num_designs_searched: int
    timestamp: str
    method: str = 'unknown'  # Track search method used
    elapsed_time: float = 0.0  # ← Add this
    cumulative_time: float = 0.0  # ← Add this


class ControlledOTAOptimizer:
    """Manages controlled step-by-step optimization"""
    
    def __init__(self, config, user_specs: str = None, llm_agent=None, optimization_config=None, ranking_method='fom_only'):

        self.start_time = None  # ← Add this
        self.iteration_times = []
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.config = config

        self.ranking_method = ranking_method  # Make sure this line exists
        self.user_specs_metric = self.config['user_specs_metric']  # Make sure this exists

        
        self.var_names = list(self.config['variable'].keys())

        # Store optimization config from LLM
        self.optimization_config = optimization_config

        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_history = []
        self.all_searched_designs = []  # Track all designs ever searched

        # Parse objective - need LLM agent here
        self.user_specs = user_specs
        self.llm_agent = llm_agent  # Assign BEFORE parsing
        
        # Get target metric from LLM agent (already extracted during LLM agent init)
        if self.llm_agent and hasattr(self.llm_agent, 'target_metric'):
            self.target_metric = self.llm_agent.target_metric
            self.objective_metric = self.target_metric['metric_key']
            self.maximize = (self.target_metric['direction'] == 'maximize')
        else:
            # Fallback to default
            self.target_metric = {
                'metric_key': 'fom',
                'direction': 'maximize',
                'is_composite': False,
                'formulation_type': 'ratio'
            }
            self.objective_metric = 'fom'
            self.maximize = True
            
    def sort_designs(self, designs):
        """
        Sort designs based on ranking method with robust error handling
        
        Parameters:
        -----------
        designs: list
            List of design results (OTAResult objects)
        
        Returns:
        --------
        list: Sorted designs (valid designs only)
        """
        # Handle None, empty, or non-iterable designs
        if designs is None:
            print("  Warning: designs is None")
            return []
        
        try:
            if not hasattr(designs, '__iter__'):
                print("  Warning: designs is not iterable")
                return []
            
            if len(designs) == 0:
                print("  Warning: designs is empty")
                return []
        except Exception as e:
            print(f"  Warning: Cannot process designs: {e}")
            return []
        
        # Filter out None FOM values
        valid_results = []
        invalid_results = []
        
        for r in designs:
            if r is None:
                invalid_results.append(r)
                continue
            
            try:
                fom_value = getattr(r, 'fom', None)
                if fom_value is not None:
                    valid_results.append(r)
                else:
                    invalid_results.append(r)
            except Exception as e:
                print(f"  Warning: Error accessing FOM for design: {e}")
                invalid_results.append(r)
        
        if invalid_results:
            print(f"  Warning: {len(invalid_results)}/{len(designs)} designs failed to produce valid FOM")
        
        if not valid_results:
            print(" Error: All designs failed to produce valid FOM!")
            return []
        
        # Sort based on ranking method
        print("########################")
        print(self.ranking_method)
        print("########################")
        try:
            if self.ranking_method == 'hybrid':
                results_sorted = sorted(
                    valid_results,
                    key=lambda x: multi_objective_sort_key_hybrid(x, self.user_specs_metric),
                    reverse=True
                )
                print(f"✅ Sorted {len(valid_results)} designs using HYBRID ranking")
            else:  # 'fom_only'
                results_sorted = sorted(
                    valid_results,
                    key=lambda x: single_objective_sort_key_fom(x),
                    reverse=True
                )
                print(f"✅ Sorted {len(valid_results)} designs using FOM-ONLY ranking")
            
            return results_sorted
    
        except Exception as e:
            print(f"❌ Error during sorting: {e}")
            import traceback
            traceback.print_exc()
            print("   Falling back to unsorted valid results")
            return valid_results
        
    def log(self, message, level="INFO"):
        """Logging with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if level == "SECTION":
            print(f"\n{'='*100}")
            print(f"[{timestamp}] {message}")
            print(f"{'='*100}")
        else:
            print(f"[{timestamp}] {level}: {message}")

    def _placeholders_in_format(self, fmt_string: str):
        names = set()
        for _, field_name, _, _ in string.Formatter().parse(fmt_string):
            if field_name:
                names.add(field_name.split('!')[0].split(':')[0].split('.')[0])
        return names


    # def make_provider(self, trial_values, var_names):
    #     trial = {name: val for name, val in zip(var_names, trial_values)}
    #     def provider(name):
    #         if name in trial:
    #             return trial[name]
    #         return None  # return None for anything not in trial
    #     return provider

    def make_provider(self, trial_values, var_names):
        """
        Build a callable that returns a value for a given variable name.
        Priority:
          1. trial_values (if provided)
          2. Variable-specific values (e.g., W_mp_values, nfin_tail_values)
          3. Generic W_values/L_values/nfin_values (fallback)
        """
        if trial_values is not None:
            # Use provided trial values
            mapping = dict(zip(var_names, trial_values))
            return lambda name: mapping.get(name)

        # Build from config value ranges
        print(f"\n{'='*80}")
        print(f"⚠️  WARNING: RANDOM VALUE PROVIDER MODE ACTIVATED")
        print(f"{'='*80}")
        print(f"No specific trial_values provided - will use RANDOM selection from value ranges")
        print(f"This should NOT happen during normal LLM-guided optimization!")
        print(f"Variables: {var_names}")
        print(f"{'='*80}\n")

        def provider(name):
            # PRIORITY 1: Check for variable-specific values
            specific_key = f"{name}_values"
            if specific_key in self.config:
                values = self.config[specific_key]
                if values:
                    value = random.choice(values)
                    print(f"⚠️  WARNING: Using RANDOM value for '{name}' from '{specific_key}': {value}")
                    return value

            # PRIORITY 2: Check for generic W_values/L_values/nfin_values
            if name.startswith('W_'):
                if 'W_values' in self.config:
                    value = random.choice(self.config['W_values'])
                    print(f"⚠️  WARNING: Using RANDOM value for '{name}' from 'W_values': {value}")
                    return value
            elif name.startswith('L_') or name == 'L':
                if 'L_values' in self.config:
                    value = random.choice(self.config['L_values'])
                    print(f"⚠️  WARNING: Using RANDOM value for '{name}' from 'L_values': {value}")
                    return value
            elif name.startswith('m_'):
                if 'm_values' in self.config:
                    value = random.choice(self.config['m_values'])
                    print(f"⚠️  WARNING: Using RANDOM value for '{name}' from 'm_values': {value}")
                    return value
            elif name.startswith('nfin_'):
                if 'nfin_values' in self.config:
                    value = random.choice(self.config['nfin_values'])
                    print(f"⚠️  WARNING: Using RANDOM value for '{name}' from 'nfin_values': {value}")
                    return value

            # PRIORITY 3: Fallback to default
            print(f"❌ ERROR: No value found for '{name}'!")
            return None
        
        return provider


    def _build_instantiation_pins(self, subckt_pins, testbench_signals):

        inst_connections = []
        for pin in subckt_pins:
            pin_str = str(pin)
            pin_upper = pin_str.upper()
            
            # Match pin to testbench signal
            matched = False
            for expected_pin, signal in testbench_signals.items():
                if str(expected_pin).upper() == pin_upper: #if expected_pin in pin_upper:
                    inst_connections.append(signal)
                    matched = True
                    break
            
            # if not matched:
            #     # This is an internal node - leave it unconnected (will float internally)
            #     # We need to create a dummy node for it
            #     inst_connections.append(pin.lower() + '_internal')

            if not matched:
                # never create a fake ground node name
                if pin_upper in ("0", "GND"):
                    inst_connections.append("0")
                else:
                    inst_connections.append(pin_str.lower() + "_internal")
        
        inst_pins = ' '.join(inst_connections)
        return inst_pins
    


    def parse_pex_subcircuit(self, netlist_path, target_name=None):
        """
        Parse PEX netlist to extract subcircuit name and pin order.
        Handles .subckt definitions written on one line or with '+' continuation lines.
        """
        with open(netlist_path, "r") as f:
            content = f.read()
    
        lines = content.splitlines()
        subckt_name = None
        pins = []
    
        for i, raw in enumerate(lines):
            line = raw.strip()
    
            # Skip comments and blank lines
            if not line or line.startswith(('*', ';', '//')):
                continue
    
            # Detect the .subckt start
            if line.lower().startswith(".subckt"):
                parts = line.split()
                if len(parts) < 3:
                    continue  # malformed line
                subckt_name = parts[1]
                pins = parts[2:]
    
                # Handle continuation lines starting with '+'
                j = i + 1
                while j < len(lines):
                    cont = lines[j].lstrip()
                    if cont.startswith('+'):
                        extra = cont[1:].strip()
                        if extra:
                            pins.extend(extra.split())
                        j += 1
                    else:
                        break
                break  # stop after first subckt found
    
        if not subckt_name:
            self.log("No .subckt definition found.", "ERROR")
            return None, None, None, None, content
    
        # Normalize pins
        pins = [("0" if str(p).strip().upper() in ("0", "GND") else str(p).strip()) for p in pins]
    
        # Separate I/O and internal pins (based on config)
        I_O_pins_cfg = [str(x).strip().upper() for x in self.config.get("I_O_pins", [])]
        io_pins = []
        internal_pins = []
        for p in pins:
            pu = p.upper()
            if pu in ("0", "GND") or any(name == pu for name in I_O_pins_cfg):
                io_pins.append(p)
            else:
                internal_pins.append(p)
    
        self.log(f"PEX subcircuit: {subckt_name}", "INFO")
        self.log(f"  I/O pins: {io_pins}", "INFO")
        self.log(f"  Internal nodes (will be left floating): {internal_pins}", "INFO")
    
        return subckt_name, pins, io_pins, internal_pins, content


    def simulate_ota_config(self, trial_values=None, netlist_path=None):
        """
        Fully generic simulator runner for arbitrary circuits.
        Supports both single-point (OTA) and multi-point (VCO) characterization.
        
        Parameters:
        -----------
        trial_values: list or dict
            Values for optimization variables
        netlist_path: str, optional
            Path to PEX netlist for post-layout simulation
            
        Returns:
        --------
        OTAResult object with simulation results
        """
        import numpy as np
        import subprocess
        import re
        import os


        original_dir = os.getcwd()
    
        var_names = list(self.config['variable'].keys())
        variables = {name: None for name in var_names}
        
        pdk_lib_path       = self.config["pdk_lib_path"]
        subckt_tmpl        = self.config["ota_subckt_template"]
        tb_tmpl            = self.config["testbench_template"]
        params             = dict(self.config.get("params", {}))
        
        # Get scaling rules - all are optional
        width_scales       = dict(self.config.get("width_scales", {}))
        length_scales      = dict(self.config.get("length_scales", {}))
        nfin_scales        = dict(self.config.get("nfin_scales", {}))
        
        subckt_name        = self.config.get("subckt_name")
        subckt_pins        = list(self.config.get("subckt_pins", []))
        I_O_pins           = list(self.config.get("I_O_pins", []))
        testbench_signals  = dict(self.config.get("testbench_signals", {}))
        metrics            = list(self.config.get("metrics", []))
        netlist_filename   = self.config.get("netlist_filename", "circuit_sim.spice")
        Pex_netlist_filename   = self.config.get("netlist_filename", "circuit_sim_pex.spice")
        
        timeout_sec        = self.config.get("timeout_sec", 120)
        value_provider = self.make_provider(trial_values, var_names)
        
        results_dir = self.config.get("results_dir", "./results")
        os.makedirs(results_dir, exist_ok=True)
        netlist_path_full = os.path.join(results_dir, netlist_filename)
        
        def resolve_key(k, fmt, provider):
            if k not in fmt or fmt[k] is None:
                if callable(provider):
                    val = provider(k)
                    if val is not None:
                        fmt[k] = val
            return fmt.get(k)
    
        def apply_scales(scales_dict, fmt, provider, scale_type=""):
            """
            Apply scaling transformations from scales_dict
            
            Parameters:
            -----------
            scales_dict: dict
                Dictionary of {final_name: [base_var, scale_factor]}
            fmt: dict
                Formatting dictionary to update
            provider: callable
                Value provider function
            scale_type: str
                Optional label for error messages ("width" or "length")
            """
            if not scales_dict:
                return
                
            for final_name, pair in scales_dict.items():
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    raise ValueError(
                        f"{scale_type.capitalize() + ' s' if scale_type else 'S'}cale '{final_name}' "
                        f"must be [unit_key, factor], got: {pair}"
                    )
                unit_key, factor = pair
                
                resolve_key(unit_key, fmt, provider)
                
                if final_name in fmt and fmt[final_name] is not None:
                    continue
                    
                if unit_key not in fmt or fmt[unit_key] is None:
                    raise KeyError(
                        f"Scale for '{final_name}' requires '{unit_key}', which is missing and "
                        f"value_provider did not supply."
                    )
                val = fmt[unit_key] * factor
                fmt[final_name] = round(val, 2)

    
        # =======================================================================
        # MAIN SIMULATION FLOW
        # =======================================================================
    
        if netlist_path is None:
            # PRE-LAYOUT PATH

            fmt = {}
            fmt.update(params)
            fmt.update(variables)
            fmt["pdk_lib_path"] = pdk_lib_path
            fmt["subckt_name"]  = subckt_name
            fmt.update(testbench_signals)

            # Expand group variables to individual transistor variables
            group_mapping = self.config.get('group_mapping', {})
            if group_mapping:
                # First: resolve all group variable values via the provider
                for group_var in group_mapping.keys():
                    resolve_key(group_var, fmt, value_provider)
                # Also resolve any non-group variables (passives etc.)
                for var in var_names:
                    resolve_key(var, fmt, value_provider)
                # Then: expand each group value to all its individual transistor variables
                for group_var, individual_vars in group_mapping.items():
                    group_value = fmt.get(group_var)
                    if group_value is not None:
                        for ind_var in individual_vars:
                            fmt[ind_var] = group_value
        
            # Determine required placeholders
            required = set()
            required |= self._placeholders_in_format(subckt_tmpl)
            required |= self._placeholders_in_format(tb_tmpl)
            
            # Add width_scales requirements
            if width_scales:
                for final_name, pair in width_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)

            # Add length_scales requirements
            if length_scales:
                for final_name, pair in length_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)

            # Add nfin_scales requirements
            if nfin_scales:
                for final_name, pair in nfin_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)
        
            # Resolve everything
            for k in required:
                resolve_key(k, fmt, value_provider)
        
            # Apply scales
            if width_scales:
                apply_scales(width_scales, fmt, value_provider, scale_type="width")

            if length_scales:
                apply_scales(length_scales, fmt, value_provider, scale_type="length")

            if nfin_scales:
                apply_scales(nfin_scales, fmt, value_provider, scale_type="nfin")
            
            # ===================================================================
            # CIRCUIT-SPECIFIC COMPUTED VALUES (BEFORE MISSING CHECK)
            # ===================================================================

            # bandgap need resistor device
            if 'pnp_model_path' in required:
                fmt['pnp_model_path'] = self.config["pnp_model_path"]
            
            # VCO: Calculate cap_total if needed
            if 'cap_total' in required:
                v_ctrl_raw = fmt.get('v_ctrl')
                if v_ctrl_raw is None:
                    v_ctrl_raw = params.get('v_ctrl', 0.9)
                
                vdd_raw = fmt.get('vdd')
                if vdd_raw is None:
                    vdd_raw = params.get('vdd', 1.8)
                
                cap_base_raw = fmt.get('cap_base')
                if cap_base_raw is None:
                    cap_base_raw = params.get('cap_base', 5e-15)
                
                cap_variable_max_raw = fmt.get('cap_variable_max')
                if cap_variable_max_raw is None:
                    cap_variable_max_raw = params.get('cap_variable_max', 50e-15)
                
                v_ctrl = float(v_ctrl_raw)
                vdd = float(vdd_raw)
                cap_base = float(cap_base_raw)
                cap_variable_max = float(cap_variable_max_raw)
                
                cap_variable = (v_ctrl / vdd) * cap_variable_max
                fmt['cap_total'] = cap_base + cap_variable
            
            # ===================================================================
    
            # Build instance pins
            inst_pins = self._build_instantiation_pins(subckt_pins, testbench_signals)
            fmt["inst_pins"] = inst_pins
            
            tb_keys = self._placeholders_in_format(tb_tmpl)
            tb_keys.discard("ota_subckt")  
        
            # Skip placeholder check for circuits with dedicated sim functions
            _dedicated_types = {'ring_oscillator', 'vco', 'switched_capacitor', 'bandgap_reference',
                                'ldo_regulator', 'xor_gate', 'nand_gate', 'resistive_load_amp',
                                'diode_load_amp', 'fold_cascode_ota_hp', 'fold_cascode_ota_lp',
                                'fold_cascode_ota_bandp'}
            if self.config.get('circuit_type') not in _dedicated_types:
                # Final missing check
                missing = sorted(
                    k for k in (self._placeholders_in_format(subckt_tmpl) | tb_keys)
                    if k not in fmt or fmt[k] is None
                )
                if missing:
                    print(f"\n{'='*80}")
                    print("ERROR: Missing placeholders")
                    print(f"{'='*80}")
                    print(f"Missing: {missing}")
                    print(f"\nAvailable keys ({len(fmt)}):")
                    for key in sorted(fmt.keys()):
                        print(f"  - {key}: {fmt[key]}")
                    print(f"{'='*80}\n")
                    raise KeyError("Missing placeholder values: " + ", ".join(missing))
        
            # Render netlist (skip for circuits with dedicated sim functions)
            if self.config.get('circuit_type') not in _dedicated_types:
                subckt_text = subckt_tmpl.format(**fmt)
                netlist = tb_tmpl.format(ota_subckt=subckt_text, **fmt)

                with open(netlist_path_full, "w") as f:
                    f.write(netlist)
    
        elif trial_values is not None and netlist_path is not None:
            # POST-PEX PATH
            
            subckt_name, subckt_pins, io_pins, internal_pins, ota_netlist = self.parse_pex_subcircuit(netlist_path)
    
            inst_pins = self._build_instantiation_pins(subckt_pins, testbench_signals)
            
            fmt = {
                "pdk_lib_path": self.config["pdk_lib_path"],
                "subckt_name":  self.config.get("subckt_name", subckt_name),
                "inst_pins":    inst_pins,
                **self.config.get("params", {}),
            }
        
            variables = {var: trial_values[i] for i, var in enumerate(var_names)}
            fmt.update(variables)
            
            required = set()
            required |= self._placeholders_in_format(tb_tmpl)
            
            # Add width_scales requirements
            if width_scales:
                for final_name, pair in width_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)

            # Add length_scales requirements
            if length_scales:
                for final_name, pair in length_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)

            # Add nfin_scales requirements
            if nfin_scales:
                for final_name, pair in nfin_scales.items():
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        unit_key, _ = pair
                        required.add(unit_key)
                        required.add(final_name)
            
            # Resolve everything
            for k in required:
                resolve_key(k, fmt, value_provider)
            
            # Apply scales
            if width_scales:
                apply_scales(width_scales, fmt, value_provider, scale_type="width")

            if length_scales:
                apply_scales(length_scales, fmt, value_provider, scale_type="length")

            if nfin_scales:
                apply_scales(nfin_scales, fmt, value_provider, scale_type="nfin")

            # bandgap need resistor device
            if 'pnp_model_path' in required:
                fmt['pnp_model_path'] = self.config["pnp_model_path"]
            
            # VCO: Calculate cap_total if needed
            if 'cap_total' in required:
                v_ctrl_raw = fmt.get('v_ctrl')
                if v_ctrl_raw is None:
                    v_ctrl_raw = params.get('v_ctrl', 0.9)
                
                vdd_raw = fmt.get('vdd')
                if vdd_raw is None:
                    vdd_raw = params.get('vdd', 1.8)
                
                cap_base_raw = fmt.get('cap_base')
                if cap_base_raw is None:
                    cap_base_raw = params.get('cap_base', 5e-15)
                
                cap_variable_max_raw = fmt.get('cap_variable_max')
                if cap_variable_max_raw is None:
                    cap_variable_max_raw = params.get('cap_variable_max', 50e-15)
                
                v_ctrl = float(v_ctrl_raw)
                vdd = float(vdd_raw)
                cap_base = float(cap_base_raw)
                cap_variable_max = float(cap_variable_max_raw)
                
                cap_variable = (v_ctrl / vdd) * cap_variable_max
                fmt['cap_total'] = cap_base + cap_variable
            
            tb_keys = self._placeholders_in_format(tb_tmpl)
            tb_keys.discard("ota_subckt")  
            
            # Final missing check
            missing = sorted(
                k for k in tb_keys
                if k not in fmt or fmt[k] is None
            )
            if missing:
                print(f"\n{'='*80}")
                print("ERROR: Missing placeholders")
                print(f"{'='*80}")
                print(f"Missing: {missing}")
                print(f"\nAvailable keys ({len(fmt)}):")
                for key in sorted(fmt.keys()):
                    print(f"  - {key}: {fmt[key]}")
                print(f"{'='*80}\n")
                raise KeyError("Missing placeholder values: " + ", ".join(missing))
            
            netlist = tb_tmpl.format(ota_subckt=ota_netlist, **fmt)
    
            with open(netlist_path_full, "w") as f:
                f.write(netlist)


        # =======================================================================
        # DETECT VCO vs OTA (BEFORE RUNNING NGSPICE)
        # =======================================================================
        
        is_vco = ('v_ctrl_min' in params and 'v_ctrl_max' in params and 
                  'num_v_ctrl_points' in params and netlist_path is None)

        is_switched_cap = (self.config.get('circuit_type') == 'switched_capacitor' and 
                   netlist_path is None)

        is_bandgap = (self.config.get('circuit_type') == 'bandgap_reference' and 
              netlist_path is None)

        is_ldo = (self.config.get('circuit_type') == 'ldo_regulator' and 
          netlist_path is None)

        is_xor = (self.config.get('circuit_type') == 'xor_gate' and 
           netlist_path is None)

    
        is_nand = (self.config.get('circuit_type') == 'nand_gate' and 
           netlist_path is None)

        is_rload_amp = (self.config.get('circuit_type') == 'resistive_load_amp' and 
                netlist_path is None)

        is_diode_load_amp = (self.config.get('circuit_type') == 'diode_load_amp' and 
                     netlist_path is None)

        is_ring_osc = (self.config.get('circuit_type') == 'ring_oscillator')
        
        if is_vco:
            return self._simulate_vco_multipoint(fmt, params, subckt_tmpl, 
                                                 timeout_sec, variables)

        elif is_switched_cap:
            # SWITCHED CAPACITOR PATH
            return self._simulate_switched_capacitor(fmt, params, variables)

        elif is_bandgap:
            return self._simulate_bandgap_reference(fmt, params, variables)

        elif is_ldo:
            # LDO PATH
            return self._simulate_ldo_regulator(fmt, params, variables)

        elif is_nand:
            return self._simulate_nand_gate(fmt, params, variables)

        elif is_xor:
            return self._simulate_xor_gate(fmt, params, variables)

        elif is_rload_amp:
            return self._simulate_resistive_load_amp(fmt, params, variables)

        elif is_diode_load_amp:
            return self._simulate_diode_load_amp(fmt, params, variables)

        elif is_ring_osc:
            return self._simulate_ring_oscillator(fmt, params, variables)

        elif self.config.get('circuit_type') == 'fold_cascode_ota_hp':
            return self._simulate_filter_circuit(fmt, params, variables, 'hpf')

        elif self.config.get('circuit_type') == 'fold_cascode_ota_lp':
            return self._simulate_filter_circuit(fmt, params, variables, 'lpf')

        elif self.config.get('circuit_type') == 'fold_cascode_ota_bandp':
            return self._simulate_filter_circuit(fmt, params, variables, 'bpf')

        # =======================================================================
        # RUN NGSPICE
        # =======================================================================
        
        result = subprocess.run(
            ["ngspice", "-b", netlist_filename],
            capture_output=True, text=True, timeout=timeout_sec,
            cwd=results_dir,
        )
    
        if result.returncode != 0:
            raise RuntimeError(f"ngspice error:\n{result.stderr}")
    
        stdout = result.stdout
    
        
        # OTA: Standard single-point extraction
        def _grab(name, text):
            m = re.search(rf"measure:\s*{re.escape(name)}\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
                          text, flags=re.IGNORECASE)
            if m: return float(m.group(1))
            m = re.search(rf"\b{re.escape(name)}\b\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
                          text, flags=re.IGNORECASE)
            return float(m.group(1)) if m else None

        results = {name: _grab(name, stdout) for name in metrics}
        
        # =======================================================================
        # POST-PROCESS METRICS
        # =======================================================================
        
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": __import__('math'), "pow": pow}
        
        processed = {}
        
        for name in metrics:
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale    = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            unit     = spec.get("unit", "")
            expr     = spec.get("expr")
        
            val = None
        
            try:
                if expr:
                    # Evaluate expression using processed results
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    #val = round(float(val), decimals)
                    val = float(val)
            except Exception:
                val = None
        
            processed[name] = val
    
        results = processed
    
        # =======================================================================
        # BUILD FINAL RESULTS
        # =======================================================================
        
        final_results = {}
    
        # Include design variables
        for k, _ in variables.items():
            final_results[k] = fmt[k]
    
        # Include metrics
        for k, v in results.items():
            final_results[k] = v
            
        return OTAResult(final_results)


    def _simulate_vco_multipoint(self, fmt, params, subckt_tmpl, timeout_sec, variables):
        """
        VCO multi-point characterization wrapper
        Uses the working characterize_vco function
        """
        import sys
        sys.path.append('.')  # Make sure we can import
        
        # Import your working function
        
        
        # Extract parameters
        W_inv_p = fmt.get('W_inv_p', 1.0)
        W_inv_n = fmt.get('W_inv_n', 0.5)
        L_inv_p = fmt.get('L_inv_p', 0.15)
        L_inv_n = fmt.get('L_inv_n', 0.15)
        
        vdd = params.get('vdd', 1.8)
        temp = params.get('temp', 27)
        v_ctrl_min = params.get('v_ctrl_min', 0.0)
        v_ctrl_max = params.get('v_ctrl_max', 1.8)
        num_points = params.get('num_v_ctrl_points', 10)
        pdk_lib_path = params.get('pdk_lib_path', self.config['pdk_lib_path'])

        results_dir = self.config.get('results_dir')
        
        print(f"\n{'='*80}")
        print(f"VCO CHARACTERIZATION")
        print(f"{'='*80}")
        print(f"W_inv_p={W_inv_p}, W_inv_n={W_inv_n}, L_inv_p={L_inv_p}, L_inv_n={L_inv_n}")
        
        # Call your working characterization function
        results = simulate_vco(
            pdk_lib_path=pdk_lib_path,
            W_inv_p_base=W_inv_p,
            W_inv_n_base=W_inv_n,
            L_inv_p_base=L_inv_p,
            L_inv_n_base=L_inv_n,
            vdd=vdd,
            temp=temp,
            v_ctrl_min=v_ctrl_min,
            v_ctrl_max=v_ctrl_max,
            num_v_ctrl_points=num_points,
            results_dir=results_dir
        )
        
        if results is None:
            # Return zeros if failed
            results = {
                'tuning_range_percent': 0.0,
                'freq_at_max': 0.0,
                'freq_at_min': 0.0,
                'power_uw': 0.0,
                'vco_gain_MHz_per_V': 0.0,
                'vco_gain_linearity': 0.0,
                'fom': 0.0,
            }
        
        # Post-process metrics (apply scaling and compute FOM expr)
        import math as _math
        metrics = self.config.get("metrics", [])
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": _math, "pow": pow, "abs": abs}

        processed = {}
        for name in metrics:
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            expr = spec.get("expr")

            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = float(raw_val) * scale
                if val is not None:
                    val = float(val)
            except Exception as e:
                print(f"  FOM calc error for {name}: {e}")
                val = None
            processed[name] = val

        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in processed.items():
            if v is not None:
                final_results[k] = v

        if 'fom' in final_results:
            print(f"  FOM = {final_results['fom']:.4f}")

        return OTAResult(final_results)


    def _simulate_switched_capacitor(self, fmt, params, variables):
        """
        Switched Capacitor simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        
        # Extract parameters - CONVERT TO FLOAT
        W_op1 = float(fmt.get('W_op1', 10.0))
        W_op2 = float(fmt.get('W_op2', 10.0))
        W_sw = float(fmt.get('W_sw', 5.0))
        L_op = float(fmt.get('L_op', 0.5))
        L_sw = float(fmt.get('L_sw', 0.15))
        
        C_samp = float(params.get('C_samp', 2e-12))
        C_hold = float(params.get('C_hold', 2e-12))
        C_load = float(params.get('C_load', 2e-12))
        m_op = int(params.get('m_op', 4))
        m_sw = int(params.get('m_sw', 2))
        
        vdd = float(params.get('vdd', 1.8))
        vin = float(params.get('vin', 0.9))
        ibias = float(params.get('ibias', 10e-6))  # <-- Add float() here
        temp = int(params.get('temp', 27))
        pdk_lib_path = self.config['pdk_lib_path']

        results_dir = self.config['results_dir']  # ADD THIS
        
        # print(f"\n{'='*80}")
        # print(f"SWITCHED CAPACITOR SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_op1={W_op1}, W_op2={W_op2}, W_sw={W_sw}, L_op={L_op}, L_sw={L_sw}")
        
        # Call simulation (uses vin_ac parameter, not vin)
        result = simulate_switched_capacitor(
            pdk_lib_path=pdk_lib_path,
            W_op1=W_op1, W_op2=W_op2, W_sw=W_sw,
            L_op=L_op, L_sw=L_sw,
            m_op=m_op, m_sw=m_sw,
            C_samp=C_samp, C_hold=C_hold, C_load=C_load,
            vdd=vdd, 
            vin_ac=0.25,  # AC amplitude
            ibias=ibias, 
            temp_nom=temp,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'gain_db': 0.0,
                'thd_db': 0.0,
                'power_uw': 9999.0,
                'phase_margin_deg': 0.0,
                'ugbw_mhz': 0.0,
            }
        else:
            # Extract metrics - all values come from actual simulation now!
            results = {
                'gain_db': result.gain_db if result.gain_db is not None else 0.0,
                'thd_db': result.thd_db if result.thd_db is not None else 0.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
                'phase_margin_deg': result.phase_margin_deg if result.phase_margin_deg is not None else 0.0,
                'ugbw_mhz': result.ugbw_mhz if result.ugbw_mhz is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    # Evaluate FOM expression
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v

        print(f"{'='*80}")
        #print(f"SWITCHED CAPACITOR SIMULATION")
        print(f"W_op1={W_op1:.2f} W_op2={W_op2:.2f} W_sw={W_sw:.2f} L_op={L_op:.2f} L_sw={L_sw:.2f} | "
              f"Gain={results.get('gain_db', 0):.2f}dB THD={results.get('thd_db', 0):.1f}dB "
              f"Power={results.get('power_uw', 0):.0f}µW PM={results.get('phase_margin_deg', 0):.1f}° "
              f"UGBW={results.get('ugbw_mhz', 0):.1f}MHz FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # Pretty print results (matching your original format)
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"SC Gain:      {results.get('gain_db', 0):.2f} dB")
        # print(f"THD:          {results.get('thd_db', 0):.1f} dB")
        # print(f"Power:        {results.get('power_uw', 0):.0f} µW")
        # print(f"Phase Margin: {results.get('phase_margin_deg', 0):.1f}°")
        # print(f"UGBW:         {results.get('ugbw_mhz', 0):.1f} MHz")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)


    def _simulate_bandgap_reference(self, fmt, params, variables):
        """
        Bandgap Reference simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        
        # Extract parameters
        W_mp = float(fmt.get('W_mp', 10.0))
        L_mp = float(fmt.get('L_mp', 2.0))
        W_mn = float(fmt.get('W_mn', 10.0))
        L_mn = float(fmt.get('L_mn', 1.0))
        L_ra = float(fmt.get('L_ra', 31.2))
        L_rb = float(fmt.get('L_rb', 132.6))
        
        # Fixed parameters
        m_mp = int(params.get('m_mp', 3))
        m_mn = int(params.get('m_mn', 2))
        m_q1 = int(params.get('m_q1', 1))
        m_q2 = int(params.get('m_q2', 8))
        m_q3 = int(params.get('m_q3', 1))
        W_mp_startup = float(params.get('W_mp_startup', 5))
        W_mn_startup = float(params.get('W_mn_startup', 1))
        L_startup = float(params.get('L_startup', 7))
        m_mp_startup = int(params.get('m_mp_startup', 1))
        m_mn_startup = int(params.get('m_mn_startup', 1))
        
        vdd = float(params.get('vdd', 2.0))
        temp = int(params.get('temp', 27))
        vdd_min = float(params.get('vdd_min', 1.8))
        vdd_max = float(params.get('vdd_max', 3.3))
        
        pdk_lib_path = self.config['pdk_lib_path']
        pnp_model_path = self.config.get('pnp_model_path', '')

        results_dir = self.config['results_dir']  # ADD THIS
        
        # print(f"\n{'='*80}")
        # print(f"BANDGAP REFERENCE SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_mp={W_mp}, L_mp={L_mp}, W_mn={W_mn}, L_mn={L_mn}")
        # print(f"L_ra={L_ra}, L_rb={L_rb}")
        
        # Call simulation
        result = simulate_bandgap_reference(
            pdk_lib_path=pdk_lib_path,
            pnp_model_path=pnp_model_path,
            W_mp=W_mp, W_mn=W_mn,
            W_mp_startup=W_mp_startup, W_mn_startup=W_mn_startup,
            L_mp=L_mp, L_mn=L_mn, L_startup=L_startup,
            m_mp=m_mp, m_mn=m_mn,
            m_mp_startup=m_mp_startup, m_mn_startup=m_mn_startup,
            m_q1=m_q1, m_q2=m_q2, m_q3=m_q3,
            L_ra=L_ra, L_rb=L_rb,
            vdd=vdd, temp=temp,
            vdd_range=(vdd_min, vdd_max),
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'vref': 0.0,
                #'tc_ppm': 9999.0,
                'power_uw': 9999.0,
                'line_regulation_percent': 100.0,
                'psrr_100hz_db': 0.0,
            }
        else:
            results = {
                'vref': result.vref if result.vref is not None else 0.0,
                #'tc_ppm': result.tc if result.tc is not None else 9999.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
                'line_regulation_percent': result.line_regulation if result.line_regulation is not None else 100.0,
                'psrr_100hz_db': result.psrr_100hz if result.psrr_100hz is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    #val = round(float(val), decimals)
                    val = float(val)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"Vref:         {results.get('vref', 0):.6f} V")
        # #print(f"TC:           {results.get('tc_ppm', 0):.1f} ppm/°C")
        # print(f"Power:        {results.get('power_uw', 0):.1f} µW")
        # print(f"Line Reg:     {results.get('line_regulation_percent', 0):.3f} %/V")
        # print(f"PSRR @100Hz:  {results.get('psrr_100hz_db', 0):.1f} dB")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")

        print(f"{'='*80}")
        #print(f"BANDGAP REFERENCE SIMULATION")
        print(f"W_mp={W_mp:.2f} L_mp={L_mp:.2f} W_mn={W_mn:.2f} L_mn={L_mn:.2f} L_ra={L_ra:.2f} L_rb={L_rb:.2f} | "
              f"Vref={results.get('vref', 0):.6f}V Power={results.get('power_uw', 0):.1f}µW "
              f"LineReg={results.get('line_regulation_percent', 0):.3f}%/V "
              f"PSRR={results.get('psrr_100hz_db', 0):.1f}dB FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
            
        return OTAResult(final_results)


    
    def _simulate_ldo_regulator(self, fmt, params, variables):
        """
        LDO Regulator simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        
        # Extract optimization variables
        W_pass = float(fmt.get('W_pass', 100.0))
        L_pass = float(fmt.get('L_pass', 0.5))
        W_diff = float(fmt.get('W_diff', 10.0))
        W_load = float(fmt.get('W_load', 20.0))
        W_bias = float(fmt.get('W_bias', 5.0))
        L_amp = float(fmt.get('L_amp', 1.0))
        
        # Fixed parameters
        m_pass = int(params.get('m_pass', 10))
        m_diff = int(params.get('m_diff', 4))
        m_load = int(params.get('m_load', 4))
        L_r1 = float(params.get('L_r1', 156))
        L_r2 = float(params.get('L_r2', 156))
        
        vdd = float(params.get('vdd', 1.8))
        vref = float(params.get('vref', 0.6))
        vout_target = float(params.get('vout_target', 1.2))
        iload = float(params.get('iload', 0.01))
        ibias = float(params.get('ibias', 1e-5))
        temp = int(params.get('temp', 27))
        temp_min = int(params.get('temp_min', -40))
        temp_max = int(params.get('temp_max', 125))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config['results_dir']  # ADD THIS
        
        # print(f"\n{'='*80}")
        # print(f"LDO REGULATOR SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_pass={W_pass}, L_pass={L_pass}, W_diff={W_diff}")
        # print(f"W_load={W_load}, W_bias={W_bias}, L_amp={L_amp}")
        
        # Call simulation
        result = simulate_ldo_regulator(
            pdk_lib_path=pdk_lib_path,
            W_pass=W_pass, L_pass=L_pass, m_pass=m_pass,
            W_diff=W_diff, W_load=W_load, W_bias=W_bias,
            L_amp=L_amp, m_diff=m_diff, m_load=m_load,
            L_r1=L_r1, L_r2=L_r2,
            vref=vref, vdd=vdd, vout_target=vout_target,
            iload=iload, ibias=ibias,
            temp_nom=temp,
            temp_range=(temp_min, temp_max),
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'vout': 0.0,
                'dropout_mv': 9999.0,
                'line_reg_mv_per_v': 9999.0,
                'load_reg_mv_per_ma': 9999.0,
                'psrr_db': 0.0,
                'power_uw': 9999.0,
            }
        else:
            results = {
                'vout': result.vout if result.vout is not None else 0.0,
                'dropout_mv': result.dropout_mv if result.dropout_mv is not None else 9999.0,
                'line_reg_mv_per_v': result.line_reg if result.line_reg is not None else 9999.0,
                'load_reg_mv_per_ma': result.load_reg if result.load_reg is not None else 9999.0,
                'psrr_db': result.psrr_db if result.psrr_db is not None else 0.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)  # Keep this for reference but don't use it
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                # Remove this rounding step entirely
                # if val is not None:
                #     val = round(float(val), decimals)
                
                # Keep full precision
                if val is not None:
                    val = float(val)  # Just convert to float, don't round
                    
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed

        # CRITICAL: Ensure 'fom' key always exists, even if None
        if 'fom' not in results:
            results['fom'] = None
            print(f"  Warning: FOM not calculated, setting to None")
        
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v

        results_filtered = {k: v for k, v in results.items() if v is not None}
    
        print(f"{'='*80}")
        print(f"Params: W_pass={W_pass:.2f} L_pass={L_pass:.2f} W_diff={W_diff:.2f} W_load={W_load:.2f} W_bias={W_bias:.2f} L_amp={L_amp:.2f}")
        
        # Build metrics string, filtering out None values
        metrics_parts = []
        if 'vout' in results_filtered:
            metrics_parts.append(f"Vout={results_filtered['vout']:.4f}V")
        if 'dropout_mv' in results_filtered:
            metrics_parts.append(f"Dropout={results_filtered['dropout_mv']:.1f}mV")
        if 'line_reg_mv_per_v' in results_filtered:
            metrics_parts.append(f"LineReg={results_filtered['line_reg_mv_per_v']:.2f}mV/V")
        if 'load_reg_mv_per_ma' in results_filtered:
            metrics_parts.append(f"LoadReg={results_filtered['load_reg_mv_per_ma']:.3f}mV/mA")
        if 'psrr_db' in results_filtered:
            metrics_parts.append(f"PSRR={results_filtered['psrr_db']:.1f}dB")
        if 'power_uw' in results_filtered:
            metrics_parts.append(f"Power={results_filtered['power_uw']:.1f}µW")
        if 'fom' in results_filtered:
            metrics_parts.append(f"FOM={results_filtered['fom']:.4f}")
        
        metrics_str = " | ".join(metrics_parts) if metrics_parts else "Simulation failed - no metrics available"
        print(f"Metrics: {metrics_str}")
        print(f"{'='*80}")
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"Vout:         {results.get('vout', 0):.4f} V")
        # print(f"Dropout:      {results.get('dropout_mv', 0):.1f} mV")
        # print(f"Line Reg:     {results.get('line_reg_mv_per_v', 0):.2f} mV/V")
        # print(f"Load Reg:     {results.get('load_reg_mv_per_ma', 0):.3f} mV/mA")
        # print(f"PSRR:         {results.get('psrr_db', 0):.1f} dB")
        # print(f"Power:        {results.get('power_uw', 0):.1f} µW")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)


    def _simulate_nand_gate(self, fmt, params, variables):
        """
        NAND Gate simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        
        # Extract optimization variables
        W_pmos = float(fmt.get('W_pmos', 1.0))
        W_nmos = float(fmt.get('W_nmos', 0.5))
        L_pmos = float(fmt.get('L_pmos', 0.15))
        L_nmos = float(fmt.get('L_nmos', 0.15))
        
        # Fixed parameters
        vdd = float(params.get('vdd', 1.8))
        temp = int(params.get('temp', 27))
        freq = float(params.get('freq', 100e6))
        c_load = float(params.get('c_load', 10e-15))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config.get('results_dir', './results')
        
        # print(f"\n{'='*80}")
        # print(f"NAND GATE SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_pmos={W_pmos}, W_nmos={W_nmos}, L_pmos={L_pmos}, L_nmos={L_nmos}")
        
        # Call simulation
        result = simulate_nand_gate(
            pdk_lib_path=pdk_lib_path,
            W_pmos=W_pmos, W_nmos=W_nmos,
            L_pmos=L_pmos, L_nmos=L_nmos,
            vdd=vdd, temp=temp,
            freq=freq, c_load=c_load,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'tphl_ps': 9999.0,
                'tplh_ps': 9999.0,
                'avg_delay_ps': 9999.0,
                'power_total_uw': 9999.0,
                'energy_per_transition_fj': 9999.0,
                'noise_margin_high': 0.0,
                'noise_margin_low': 0.0,
            }
        else:
            results = {
                'tphl_ps': result.tphl_ps if result.tphl_ps is not None else 9999.0,
                'tplh_ps': result.tplh_ps if result.tplh_ps is not None else 9999.0,
                'avg_delay_ps': result.avg_delay_ps if result.avg_delay_ps is not None else 9999.0,
                'power_total_uw': result.power_total_uw if result.power_total_uw is not None else 9999.0,
                'energy_per_transition_fj': result.energy_per_transition_fj if result.energy_per_transition_fj is not None else 9999.0,
                'noise_margin_high': result.noise_margin_high if result.noise_margin_high is not None else 0.0,
                'noise_margin_low': result.noise_margin_low if result.noise_margin_low is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v

        print(f"{'='*80}")
        print(f"W_pmos={W_pmos:.2f} W_nmos={W_nmos:.2f} L_pmos={L_pmos:.2f} L_nmos={L_nmos:.2f} | "
              f"tPHL={results.get('tphl_ps', 0):.2f}ps tPLH={results.get('tplh_ps', 0):.2f}ps "
              f"AvgDelay={results.get('avg_delay_ps', 0):.2f}ps Power={results.get('power_total_uw', 0):.3f}µW "
              f"Energy={results.get('energy_per_transition_fj', 0):.3f}fJ NMH={results.get('noise_margin_high', 0):.4f}V "
              f"NML={results.get('noise_margin_low', 0):.4f}V FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"tPHL:         {results.get('tphl_ps', 0):.2f} ps")
        # print(f"tPLH:         {results.get('tplh_ps', 0):.2f} ps")
        # print(f"Avg Delay:    {results.get('avg_delay_ps', 0):.2f} ps")
        # print(f"Total Power:  {results.get('power_total_uw', 0):.3f} µW")
        # print(f"Energy/Trans: {results.get('energy_per_transition_fj', 0):.3f} fJ")
        # print(f"NMH:          {results.get('noise_margin_high', 0):.4f} V")
        # print(f"NML:          {results.get('noise_margin_low', 0):.4f} V")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)

    def _simulate_xor_gate(self, fmt, params, variables):
        """
        XOR Gate simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        # Extract optimization variables
        W_pmos = float(fmt.get('W_pmos', 1.0))
        W_nmos = float(fmt.get('W_nmos', 0.5))
        L_pmos = float(fmt.get('L_pmos', 0.15))
        L_nmos = float(fmt.get('L_nmos', 0.15))
        
        # Fixed parameters
        vdd = float(params.get('vdd', 1.8))
        temp = int(params.get('temp', 27))
        freq = float(params.get('freq', 100e6))
        c_load = float(params.get('c_load', 10e-15))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config.get('results_dir', './results')
        
        # print(f"\n{'='*80}")
        # print(f"XOR GATE SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_pmos={W_pmos}, W_nmos={W_nmos}, L_pmos={L_pmos}, L_nmos={L_nmos}")
        
        # Call simulation
        result = simulate_xor_gate(
            pdk_lib_path=pdk_lib_path,
            W_pmos=W_pmos, W_nmos=W_nmos,
            L_pmos=L_pmos, L_nmos=L_nmos,
            vdd=vdd, temp=temp,
            freq=freq, c_load=c_load,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'avg_delay_ps': 9999.0,
                'power_total_uw': 9999.0,
                'energy_per_transition_fj': 9999.0,
                'noise_margin_high': 0.0,
                'noise_margin_low': 0.0,
            }
        else:
            results = {
                'avg_delay_ps': result.avg_delay_ps if result.avg_delay_ps is not None else 9999.0,
                'power_total_uw': result.power_total_uw if result.power_total_uw is not None else 9999.0,
                'energy_per_transition_fj': result.energy_per_transition_fj if result.energy_per_transition_fj is not None else 9999.0,
                'noise_margin_high': result.noise_margin_high if result.noise_margin_high is not None else 0.0,
                'noise_margin_low': result.noise_margin_low if result.noise_margin_low is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v


        print(f"{'='*80}")
        #print(f"XOR GATE SIMULATION")
        print(f"W_pmos={W_pmos:.1f} W_nmos={W_nmos:.1f} L_pmos={L_pmos:.1f} L_nmos={L_nmos:.1f} | "
              f"AvgDelay={results.get('avg_delay_ps', 0):.2f}ps Power={results.get('power_total_uw', 0):.3f}µW "
              f"Energy={results.get('energy_per_transition_fj', 0):.3f}fJ NMH={results.get('noise_margin_high', 0):.4f}V "
              f"NML={results.get('noise_margin_low', 0):.4f}V FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"Avg Delay:    {results.get('avg_delay_ps', 0):.2f} ps")
        # print(f"Total Power:  {results.get('power_total_uw', 0):.3f} µW")
        # print(f"Energy/Trans: {results.get('energy_per_transition_fj', 0):.3f} fJ")
        # print(f"NMH:          {results.get('noise_margin_high', 0):.4f} V")
        # print(f"NML:          {results.get('noise_margin_low', 0):.4f} V")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)


    
    # def _simulate_resistive_load_amp(self, fmt, params, variables):
    #     """
    #     Resistive Load Amplifier simulation wrapper
    #     """
    #     import sys
    #     import math
    #     sys.path.append('.')
        
        
        
    #     # Extract optimization variables
    #     W_input = float(fmt.get('W_input', 5.0))
    #     L_input = float(fmt.get('L_input', 0.5))
    #     R_load = float(fmt.get('R_load', 50e3))
        
    #     # Fixed parameters
    #     vdd = float(params.get('vdd', 1.8))
    #     vbias = float(params.get('vbias', 0.6))
    #     temp = int(params.get('temp', 27))
    #     c_load = float(params.get('c_load', 10e-15))
        
    #     pdk_lib_path = self.config['pdk_lib_path']
    #     results_dir = self.config.get('results_dir', './results')
        
    #     # print(f"\n{'='*80}")
    #     # print(f"RESISTIVE LOAD AMPLIFIER SIMULATION")
    #     # print(f"{'='*80}")
    #     # print(f"W_input={W_input}, L_input={L_input}, R_load={R_load/1e3:.1f}kΩ")
        
    #     # Call simulation
    #     result = simulate_resistive_load_amp(
    #         pdk_lib_path=pdk_lib_path,
    #         W_input=W_input,
    #         L_input=L_input,
    #         R_load=R_load,
    #         vdd=vdd,
    #         vbias=vbias,
    #         temp=temp,
    #         c_load=c_load,
    #         results_dir=results_dir
    #     )
        
    #     if result is None:
    #         print("✗ Simulation failed")
    #         results = {
    #             'gain_db': 0.0,
    #             'bandwidth_mhz': 0.0,
    #             'power_uw': 9999.0,
    #             'output_swing_v': 0.0,
    #         }
    #     else:
    #         results = {
    #             'gain_db': result.gain_db if result.gain_db is not None else 0.0,
    #             'bandwidth_mhz': result.bandwidth_mhz if result.bandwidth_mhz is not None else 0.0,
    #             'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
    #             'output_swing_v': result.output_swing_v if result.output_swing_v is not None else 0.0,
    #         }
        
    #     # Post-process with metric_post (FOM calculated here)
    #     metric_post = self.config.get("metric_post", {})
    #     safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
    #     processed = {}
    #     for name in self.config.get('metrics', []):
    #         spec = metric_post.get(name, {})
    #         raw_val = results.get(name, None)
    #         scale = float(spec.get("scale", 1))
    #         decimals = spec.get("decimals", 2)
    #         expr = spec.get("expr")
            
    #         val = None
    #         try:
    #             if expr:
    #                 val = eval(expr, safe_globals, processed)
    #             elif raw_val is not None:
    #                 val = raw_val * scale
                
    #             if val is not None:
    #                 val = round(float(val), decimals)
    #         except Exception as e:
    #             print(f"  Warning: Could not calculate {name}: {e}")
    #             val = None
            
    #         processed[name] = val
        
    #     results = processed
        
    #     # Package final results
    #     final_results = {}
    #     for k in variables.keys():
    #         final_results[k] = fmt[k]
    #     for k, v in results.items():
    #         final_results[k] = v


    #     print(f"{'='*80}")
    #     print(f"W_input={W_input:.1f} L_input={L_input:.1f} R_load={R_load/1e3:.1f}kΩ | "
    #           f"Gain={results.get('gain_db', 0):.2f}dB BW={results.get('bandwidth_mhz', 0):.2f}MHz "
    #           f"Power={results.get('power_uw', 0):.3f}µW OutSwing={results.get('output_swing_v', 0):.4f}V "
    #           f"FOM={results.get('fom', 0):.4f}")
    #     print(f"{'='*80}")
        
    #     # print(f"\n{'='*80}")
    #     # print(f"FINAL PERFORMANCE SUMMARY")
    #     # print(f"{'='*80}")
    #     # print(f"Gain:         {results.get('gain_db', 0):.2f} dB")
    #     # print(f"Bandwidth:    {results.get('bandwidth_mhz', 0):.2f} MHz")
    #     # print(f"Power:        {results.get('power_uw', 0):.3f} µW")
    #     # print(f"Out Swing:    {results.get('output_swing_v', 0):.4f} V")
    #     # print(f"FOM:          {results.get('fom', 0):.4f}")
    #     # print(f"{'='*80}\n")
            
    #     return OTAResult(final_results)


    def _simulate_resistive_load_amp(self, fmt, params, variables):

        """
        Resistive Load Amplifier simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        # Extract optimization variables
        W_input = float(fmt.get('W_input', 5.0))
        L_input = float(fmt.get('L_input', 0.5))
        R_load = float(fmt.get('R_load', 50e3))
        
        # Fixed parameters
        vdd = float(params.get('vdd', 1.8))
        vbias = float(params.get('vbias', 0.6))
        temp = int(params.get('temp', 27))
        c_load = float(params.get('c_load', 10e-15))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config.get('results_dir', './results')
        
        # Call simulation
        result = simulate_resistive_load_amp(
            pdk_lib_path=pdk_lib_path,
            W_input=W_input,
            L_input=L_input,
            R_load=R_load,
            vdd=vdd,
            vbias=vbias,
            temp=temp,
            c_load=c_load,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'gain_db': 0.0,
                'bandwidth_mhz': 0.0,
                'power_uw': 9999.0,
                'output_swing_v': 0.0,
            }
        else:
            results = {
                'gain_db': result.gain_db if result.gain_db is not None else 0.0,
                'bandwidth_mhz': result.bandwidth_mhz if result.bandwidth_mhz is not None else 0.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
                'output_swing_v': result.output_swing_v if result.output_swing_v is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
            #print(f"DEBUG: Added variable {k} = {fmt[k]}")  # Debug each addition
        
        for k, v in results.items():
            final_results[k] = v
            #print(f"DEBUG: Added metric {k} = {v}")  # Debug each addition
        
    
        print(f"{'='*80}")
        print(f"W_input={W_input:.1f} L_input={L_input:.1f} R_load={R_load/1e3:.1f}kΩ | "
              f"Gain={results.get('gain_db', 0):.2f}dB BW={results.get('bandwidth_mhz', 0):.2f}MHz "
              f"Power={results.get('power_uw', 0):.3f}µW OutSwing={results.get('output_swing_v', 0):.4f}V "
              f"FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # ========================================
        # DEBUG SECTION - CREATING OTAResult
        # ========================================
        
        result_obj = OTAResult(results=final_results)
        # ========================================
            
        return result_obj
    


    def _simulate_diode_load_amp(self, fmt, params, variables):
        """
        Diode Load Amplifier simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        
        
        # Extract optimization variables
        W_input = float(fmt['W_input'])
        W_load = float(fmt['W_load'])
        L_input = float(fmt['L_input'])
        L_load = float(fmt['L_load'])
        
        # Fixed parameters
        vdd = float(params.get('vdd', 1.8))
        vbias = float(params.get('vbias', 0.6))
        temp = int(params.get('temp', 27))
        c_load = float(params.get('c_load', 10e-15))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config.get('results_dir', './results')
        
        # print(f"\n{'='*80}")
        # print(f"DIODE LOAD AMPLIFIER SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_input={W_input}, W_load={W_load}, L_input={L_input}, L_load={L_load}")
        
        # Call simulation
        result = simulate_diode_load_amp(
            pdk_lib_path=pdk_lib_path,
            W_input=W_input,
            W_load=W_load,
            L_input=L_input,
            L_load=L_load,
            vdd=vdd,
            vbias=vbias,
            temp=temp,
            c_load=c_load,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'gain_db': 0.0,
                'bandwidth_mhz': 0.0,
                'power_uw': 9999.0,
                'output_swing_v': 0.0,
            }
        else:
            results = {
                'gain_db': result.gain_db if result.gain_db is not None else 0.0,
                'bandwidth_mhz': result.bandwidth_mhz if result.bandwidth_mhz is not None else 0.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
                'output_swing_v': result.output_swing_v if result.output_swing_v is not None else 0.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v


        print(f"{'='*80}")
        print(f"W_input={W_input:.1f} W_load={W_load:.1f} L_input={L_input:.1f} L_load={L_load:.1f} | "
              f"Gain={results.get('gain_db', 0):.2f}dB BW={results.get('bandwidth_mhz', 0):.2f}MHz "
              f"Power={results.get('power_uw', 0):.3f}µW OutSwing={results.get('output_swing_v', 0):.4f}V "
              f"FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"Gain:         {results.get('gain_db', 0):.2f} dB")
        # print(f"Bandwidth:    {results.get('bandwidth_mhz', 0):.2f} MHz")
        # print(f"Power:        {results.get('power_uw', 0):.3f} µW")
        # print(f"Out Swing:    {results.get('output_swing_v', 0):.4f} V")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)


    
    def _simulate_ring_oscillator(self, fmt, params, variables):
        """
        3-Stage Ring Oscillator simulation wrapper
        """
        import sys
        import math
        sys.path.append('.')
        
        # Extract optimization variables
        L_inv_values = float(fmt['L_inv'])
        W_pmos0_values = float(fmt['W_pmos0'])
        W_nmos0_values = float(fmt['W_nmos0'])
        W_pmos1_values = float(fmt['W_pmos1'])
        W_nmos1_values = float(fmt['W_nmos1'])
        W_pmos2_values = float(fmt['W_pmos2'])
        W_nmos2_values = float(fmt['W_nmos2'])


        # Fixed parameters
        vdd = float(params.get('vdd', 1.8))
        temp = int(params.get('temp', 27))
        c_load = float(params.get('c_load', 10e-15))
        
        pdk_lib_path = self.config['pdk_lib_path']
        results_dir = self.config.get('results_dir', './results')
        
        # print(f"\n{'='*80}")
        # print(f"3-STAGE RING OSCILLATOR SIMULATION")
        # print(f"{'='*80}")
        # print(f"W_inv={W_inv}, L_inv={L_inv}")
        
        # Call simulation
        result = simulate_ring_oscillator(
            pdk_lib_path=pdk_lib_path,
            L_inv=L_inv_values,
            W_pmos0=W_pmos0_values,
            W_nmos0=W_nmos0_values,
            W_pmos1=W_pmos1_values,
            W_nmos1=W_nmos1_values,
            W_pmos2=W_pmos2_values,
            W_nmos2=W_nmos2_values,
            vdd=vdd,
            temp=temp,
            c_load=c_load,
            results_dir=results_dir
        )
        
        if result is None:
            print("✗ Simulation failed")
            results = {
                'frequency_mhz': 0.0,
                'power_uw': 9999.0,
                'delay_per_stage_ps': 9999.0,
            }
        else:
            results = {
                'frequency_mhz': result.frequency_mhz if result.frequency_mhz is not None else 0.0,
                'power_uw': result.power_uw if result.power_uw is not None else 9999.0,
                'delay_per_stage_ps': result.delay_per_stage_ps if result.delay_per_stage_ps is not None else 9999.0,
            }
        
        # Post-process with metric_post (FOM calculated here)
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}
        
        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            decimals = spec.get("decimals", 2)
            expr = spec.get("expr")
            
            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = raw_val * scale
                
                if val is not None:
                    val = round(float(val), decimals)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            
            processed[name] = val
        
        results = processed
        
        # Package final results
        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in results.items():
            final_results[k] = v


        print(f"{'='*80}")
        print(f"W_pmos0={W_pmos0_values:.2f} W_nmos0={W_nmos0_values:.2f} W_pmos1={W_pmos1_values:.2f} W_nmos1={W_nmos1_values:.2f} W_pmos2={W_pmos2_values:.2f} W_nmos2={W_nmos2_values:.2f} L_inv={L_inv_values:.2f}  | "
              f"Freq={results.get('frequency_mhz', 0):.2f}MHz Power={results.get('power_uw', 0):.3f}µW "
              f"Delay/Stage={results.get('delay_per_stage_ps', 0):.2f}ps FOM={results.get('fom', 0):.4f}")
        print(f"{'='*80}")
        
        # print(f"\n{'='*80}")
        # print(f"FINAL PERFORMANCE SUMMARY")
        # print(f"{'='*80}")
        # print(f"Frequency:    {results.get('frequency_mhz', 0):.2f} MHz")
        # print(f"Power:        {results.get('power_uw', 0):.3f} µW")
        # print(f"Delay/Stage:  {results.get('delay_per_stage_ps', 0):.2f} ps")
        # print(f"FOM:          {results.get('fom', 0):.4f}")
        # print(f"{'='*80}\n")
            
        return OTAResult(final_results)

    def _simulate_filter_circuit(self, fmt, params, variables, filter_type):
        """Wrapper for folded cascode OTA with HPF/LPF/BPF filter"""
        import math

        # Map YAML names (W_in_base) to sim function names (W_in)
        W_in = float(fmt.get('W_in_base', fmt.get('W_in', 1.0)))
        W_fold = float(fmt.get('W_fold_base', fmt.get('W_fold', 1.0)))
        W_sink = float(fmt.get('W_sink_base', fmt.get('W_sink', 1.0)))
        W_mirr = float(fmt.get('W_mirr_base', fmt.get('W_mirr', 1.0)))
        W_casc_n = float(fmt.get('W_casc_n_base', fmt.get('W_casc_n', 1.0)))
        W_casc_p = float(fmt.get('W_casc_p_base', fmt.get('W_casc_p', 1.0)))
        L = float(fmt.get('L', params.get('L', 0.15)))
        R1 = float(fmt.get('R1', 1000))
        R2 = float(fmt.get('R2', 1000))
        C1 = float(fmt.get('C1', 10e-12))
        C2 = float(fmt.get('C2', 10e-12))

        pdk_lib_path = self.config['pdk_lib_path']
        vdd = float(params.get('vdd', 1.8))
        vbn = float(params.get('vbn', 0.7))
        vbp = float(params.get('vbp', 1.0))
        vcm = float(params.get('vcm', 0.9))
        cload = float(params.get('cload', 1e-12))
        itail = float(params.get('itail', 10e-6))

        sim_kwargs = dict(
            pdk_lib_path=pdk_lib_path,
            W_in=W_in, W_fold=W_fold, W_sink=W_sink,
            W_mirr=W_mirr, W_casc_n=W_casc_n, W_casc_p=W_casc_p,
            L=L, R1=R1, R2=R2, C1=C1, C2=C2,
            vdd=vdd, vbn=vbn, vbp=vbp, cload=cload, vcm=vcm, itail=itail,
            results_dir=str(self.results_dir),
        )

        if filter_type == 'hpf':
            result = simulate_folder_cascode_ota_with_hpf(**sim_kwargs)
        elif filter_type == 'lpf':
            result = simulate_folder_cascode_ota_with_lpf(**sim_kwargs)
        elif filter_type == 'bpf':
            result = simulate_folder_cascode_ota_with_bpf(**sim_kwargs)
        else:
            print(f"Unknown filter type: {filter_type}")
            return OTAResult({k: 0.0 for k in variables})

        if result is None:
            print(f"  ✗ {filter_type.upper()} simulation failed")
            results = {}
        else:
            results = vars(result) if hasattr(result, '__dict__') else result

        # Post-process with metric_post
        metric_post = self.config.get("metric_post", {})
        safe_globals = {"__builtins__": {}, "math": math, "pow": pow, "abs": abs, "max": max}

        processed = {}
        for name in self.config.get('metrics', []):
            spec = metric_post.get(name, {})
            raw_val = results.get(name, None)
            scale = float(spec.get("scale", 1))
            expr = spec.get("expr")

            val = None
            try:
                if expr:
                    val = eval(expr, safe_globals, processed)
                elif raw_val is not None:
                    val = float(raw_val) * scale
                if val is not None:
                    val = float(val)
            except Exception as e:
                print(f"  Warning: Could not calculate {name}: {e}")
                val = None
            processed[name] = val

        final_results = {}
        for k in variables.keys():
            final_results[k] = fmt[k]
        for k, v in processed.items():
            if v is not None:
                final_results[k] = v

        if 'fom' in final_results:
            print(f"  FOM = {final_results['fom']:.4f}")

        return OTAResult(final_results)

    def generate_search_points(self, n_samples, W_values, method='random',
                     previous_best=None, search_radius=None, algorithm_params=None):
        
       
        target_metric_key = self.target_metric['metric_key']
        targets = [target_metric_key]
        
        # Extract weights for weighted formulations
        if (self.target_metric.get('formulation_type') == 'weighted_difference' and 
            'weights' in target_metric):
            weights = self.target_metric['weights']
        else:
            weights = None
        
        return enhanced_generate_search_points(
            config=self.config,
            optimizer=self,
            n_samples=n_samples,
            W_values=W_values,
            method=method,
            previous_best=previous_best,
            search_radius=search_radius,
            all_previous_results=self.all_searched_designs,
            targets=targets,
            weights=weights,
            algorithm_params=algorithm_params,
            optimization_config=self.optimization_config
        )


    # def _check_user_constraints(self, design: OTAResult, fom_only_check: bool = True)) -> bool:
        
    #     """Ask LLM if design meets user specifications"""
    #     if not self.user_specs or not self.llm_agent:
    #         return True
        
    #     # Get all design metrics as a flat dictionary
    #     metrics = design.to_dict()
        
    #     # Build readable metric lines, skipping transistor widths (W_*)
    #     metric_lines = []
    #     for key, value in metrics.items():
    #         if key.startswith("W_") or value is None:
    #             continue
    #         label = key.replace("_", " ").title()
    #         if isinstance(value, (int, float, np.floating)):
    #             metric_lines.append(f"    - {label}: {value:.2f}")
    #         else:
    #             metric_lines.append(f"    - {label}: {value}")
        
    #     metric_text = "\n".join(metric_lines)
        
    #     # Compose the final prompt
    #     prompt = f"""Given user specifications:
    #     "{self.user_specs_metric}"
        
    #     Does this design meet the requirements?
    #     {metric_text}
        
    #     Respond ONLY JSON: {{"meets_specs": true, "reason": "why"}}
    #     """
        
    #     try:
    #         response = self.llm_agent.model.generate_content(prompt)
    #         json_text = response.text.strip()
    #         if "```json" in json_text:
    #             json_text = json_text.split("```json")[1].split("```")[0].strip()
    #         result = json.loads(json_text)
    #         if not result['meets_specs']:
    #             print(f"    Reason: {result['reason']}")
    #         return result['meets_specs']
    #     except:
    #         return True

    def _check_user_constraints(self, design: OTAResult, fom_only_check: bool = True) -> bool:
    
        """Check if design meets user specifications
        
        Args:
            design: OTAResult to check
            fom_only_check: If True, only check FOM directly. If False, use LLM for full specs.
        """
        if not self.user_specs:
            return True
        
        # Get FOM value
        fom = design.fom
        
        if fom is None:
            return False  # Failed simulation doesn't meet specs
        
        # FOM-only direct check (default)
        if fom_only_check:
            try:
                # Parse ONLY the FOM line from user_specs_metric
                spec_lines = self.user_specs_metric.strip().split('\n')
                
                fom_spec = None
                for line in spec_lines:
                    line_stripped = line.strip()
                    line_lower = line_stripped.lower()
                    if line_lower.startswith('fom'):
                        fom_spec = line_stripped  # Use the original line, not lowercase
                        break
                
                if not fom_spec:
                    print(f"    Warning: No FOM specification found. Accepting design.")
                    return True
                
                # Parse the FOM requirement from THIS LINE ONLY
                fom_spec_lower = fom_spec.lower()
                
                # Extract just the number after the operator
                if '>=' in fom_spec_lower:
                    # Split and take only the part after >=, then split by space to get first token
                    threshold_str = fom_spec_lower.split('>=')[1].strip().split()[0]
                    threshold = float(threshold_str)
                    meets_spec = fom >= threshold
                elif '>' in fom_spec_lower:
                    threshold_str = fom_spec_lower.split('>')[1].strip().split()[0]
                    threshold = float(threshold_str)
                    meets_spec = fom > threshold
                elif '<=' in fom_spec_lower:
                    threshold_str = fom_spec_lower.split('<=')[1].strip().split()[0]
                    threshold = float(threshold_str)
                    meets_spec = fom <= threshold
                elif '<' in fom_spec_lower:
                    threshold_str = fom_spec_lower.split('<')[1].strip().split()[0]
                    threshold = float(threshold_str)
                    meets_spec = fom < threshold
                else:
                    print(f"    Warning: Cannot parse FOM spec '{fom_spec}'. Accepting design.")
                    return True
                
                if not meets_spec:
                    print(f"    FOM {fom:.4f} does not meet requirement: {fom_spec}")
                
                return meets_spec
                
            except Exception as e:
                print(f"    Warning: Error parsing FOM spec: {e}. Accepting design.")
                import traceback
                traceback.print_exc()
                return True
        
    
    def search_designs(self, n_samples, method='random', previous_best=None, search_radius=None, algorithm_params=None):
        """Search N design points"""
        self.log(f"Searching {n_samples} design points using {method} method", "INFO")
        
        # FIXED: Support all variable types (not just W_values)
        # Get all variable names and their value ranges
        var_values = {}
        for var_name in self.var_names:
            # Check for variable-specific values first
            specific_key = f"{var_name}_values"
            if specific_key in self.config:
                var_values[var_name] = self.config[specific_key]
            # Fall back to generic W_values or L_values
            elif var_name.startswith('W_') and 'W_values' in self.config:
                var_values[var_name] = self.config['W_values']
            elif (var_name.startswith('L_') or var_name == 'L') and 'L_values' in self.config:
                var_values[var_name] = self.config['L_values']
            elif var_name.startswith('R_') and 'R_values' in self.config:
                var_values[var_name] = self.config['R_values']
            elif var_name.startswith('m_') and 'm_values' in self.config:
                var_values[var_name] = self.config['m_values']
            # Fall back to generic nfin_values for FinFET (ASAP7, etc.)
            elif var_name.startswith('nfin_') and 'nfin_values' in self.config:
                var_values[var_name] = self.config['nfin_values']
            else:
                raise ValueError(f"No value range found for variable {var_name}")
        
        search_points = self.generate_search_points(
            n_samples, var_values, method, previous_best, search_radius, algorithm_params
        )
        
        # FIXED: Use FOM directly
        target_metric_name = 'fom'
        
        results = []
        for i, trial_values in enumerate(search_points, 1):
            result = self.simulate_ota_config(trial_values=trial_values)
            
            if result:
                results.append(result)
                self.all_searched_designs.append(result)
                if i % 5 == 0:
                    # Get FOM value
                    fom_value = result.results.get('fom', 0) if hasattr(result, 'results') else getattr(result, 'fom', 0)
                    fom_str = f"{fom_value:.4f}" if fom_value is not None else "FAILED"
                    print(f"  Progress: {i}/{n_samples} - Latest FOM: {fom_str}")
        
        self.log(f"Search complete: {len(results)}/{n_samples} successful", "INFO")
        
        if results:
            # Sort by FOM (always maximize)
            # results_sorted = sorted(
            #     results,
            #     key=lambda x: x.results.get('fom', 0) if hasattr(x, 'results') else getattr(x, 'fom', 0),
            #     reverse=True  # Always maximize FOM
            # )


            # Filter out designs with None or missing FOM
            valid_results = [r for r in results if getattr(r, 'fom', None) is not None]
            invalid_results = [r for r in results if getattr(r, 'fom', None) is None]
            
            if invalid_results:
                print(f"⚠️  Warning: {len(invalid_results)}/{len(results)} designs failed to produce valid FOM")
            
            # Sort only valid results
            if valid_results:
                results_sorted = self.sort_designs(valid_results)
                print(f"✅ Successfully sorted {len(valid_results)} valid designs")
            else:
                print("❌ Error: All designs failed to produce valid FOM!")
                results_sorted = []
            
            # Check if best design meets user target
            best_design = results_sorted[0]
            meets_target = self._check_user_constraints(best_design)
            metrics_config = self.config.get("metric_post", {})
            
            self.log(f"Top 3 designs (by {self.ranking_method.upper() if hasattr(self, 'ranking_method') else 'FOM'}):", "INFO")
            for i, r in enumerate(results_sorted[:3], 1):
                values = r.results if hasattr(r, "results") else r.__dict__
                
                # Design parameters
                params_list = []
                for var in self.var_names:
                    val = values.get(var, 'NA')
                    # Debug: print what we got
                    if var == 'L' and val == 'NA':
                        print(f"  DEBUG: L not found in values. Available keys: {list(values.keys())[:10]}")
                    if isinstance(val, (int, float)):
                        params_list.append(f"{var}={val:.2f}")
                    else:
                        # Try to convert string to float
                        if isinstance(val, str) and val != 'NA':
                            try:
                                float_val = float(val)
                                params_list.append(f"{var}={float_val:.2e}")
                            except ValueError:
                                params_list.append(f"{var}=NA")
                        else:
                            params_list.append(f"{var}=NA")
                params = " ".join(params_list)
            
                # Performance metrics (use YAML config for precision & units)
                perf_parts = []
                for m in self.config.get('metrics', []):
                    if m in values and values[m] is not None:
                        metric_spec = metrics_config.get(m, {})
                        decimals = metric_spec.get('decimals', 2)
                        unit = metric_spec.get('unit', '')
                        try:
                            val = float(values[m])
                            perf_parts.append(f"{m}={val:.{decimals}f}{unit}")
                        except (ValueError, TypeError):
                            perf_parts.append(f"{m}=NA")
                
                perf = " ".join(perf_parts)
                
                # Calculate and add ranking tuple
                if hasattr(self, 'ranking_method') and self.ranking_method == 'hybrid':
                    ranking_tuple = multi_objective_sort_key_hybrid(r, self.user_specs_metric)
                    # Format tuple nicely
                    if isinstance(ranking_tuple, tuple) and len(ranking_tuple) == 3:
                        rank_str = f"rank=({ranking_tuple[0]}, {ranking_tuple[1]:.3f}, {ranking_tuple[2]:.3f})"
                    else:
                        rank_str = f"rank={ranking_tuple}"
                else:
                    fom_value = single_objective_sort_key_fom(r)
                    rank_str = f"rank={fom_value:.3f}"
                
                print(f"  {i}. {params} | {perf} {rank_str}")
            
            if meets_target:
                self.log(f"✓ Best design MEETS user specifications!", "INFO")
            else:
                self.log(f"⚠ Best design does NOT yet meet all user specifications", "WARNING")
            
            return results_sorted
        
        return []


    def save_spice_for_align(self, best_design, spice_file: Path, length_param_name: str = "L"):
        """
        Save ALIGN-ready SPICE:
          - Replace W={W_*} using (value*10)e-7 format
          - Replace L={L} if provided in params
          - Add nf from width_scales if missing
          - Treat None/missing as 0
          - Leave other params unchanged
        """
        import re
    
        tmpl   = self.config["ota_subckt_template"]
        scales = {str(k).lower(): v for k, v in self.config.get("width_scales", {}).items()}
        params = {str(k).lower(): v for k, v in self.config.get("params", {}).items()}
    
        def nm_to_e7(x):  # interpret nm and show as (value*10)e-7
            return f"{float(x) * 10:.6g}e-7"
    
        def get_base(name):
            return best_design[name] if isinstance(best_design, dict) else getattr(best_design, name, None)
    
        w_re   = re.compile(r'(\bW\s*=\s*)\{(W_[A-Za-z0-9_]+)\}', re.I)
        l_re   = re.compile(r'(\bL\s*=\s*)\{(' + re.escape(length_param_name) + r')\}', re.I)
        tx_re  = re.compile(r'^\s*x[mn]\w*\b', re.I)
        nf_re  = re.compile(r'\bnf\s*=\s*\d+', re.I)
    
        out = []
        for raw in tmpl.splitlines():
            line = raw
            nf_val = 1
    
            # --- Handle W={W_xxx}
            def sub_w(m):
                prefix, key = m.group(1), m.group(2).lower()
                width_nm = 0.0
                if key in params and params[key] is not None:
                    width_nm = float(params[key])
                elif key in scales:
                    base, factor = scales[key]
                    base_val = get_base(base)
                    width_nm = float(base_val or 0.0)
                    nonlocal_nf[0] = int(factor)
                return f"{prefix}{nm_to_e7(width_nm)}"
    
            nonlocal_nf = [1]
            line = w_re.sub(sub_w, line)
            nf_val = nonlocal_nf[0]
    
            # --- Handle L={L}
            if length_param_name.lower() in params:
                line = l_re.sub(lambda m: f"{m.group(1)}{nm_to_e7(float(params[length_param_name.lower()] or 0))}", line)
    
            # --- Add nf if transistor and missing
            if tx_re.match(line) and not nf_re.search(line) and nf_val > 1:
                line = line.rstrip() + f" nf={nf_val}"
    
            out.append(line)
    
        text = "\n".join(out) + "\n"
        text = re.sub(r'e-0(\d)', r'e-\1', text)  # cosmetic
    
        leftovers = re.findall(r'\{[^}]+\}', text)
        if leftovers:
            raise ValueError("Unresolved placeholders: " + ", ".join(sorted(set(leftovers))))
    
        Path(spice_file).write_text(text)
        self.log(f"Saved SPICE netlist: {spice_file}")

    
    
    # def save_spice_for_align(self, best_design, spice_file: Path, length_param_name: str = "L"):
    #     """
    #     Create an ALIGN-friendly subckt:
    #       - Replace W={W_*} with numeric widths displayed as (value*10)e-7
    #       - Convert nm to meters display style, e.g. 0.15nm -> 1.5e-7, 1.05nm -> 10.5e-7
    #       - Add nf=<scale> for each transistor
    #     """

    
    #     subckt_tmpl: str = self.config["ota_subckt_template"]
    #     scales: Mapping = self.config["width_scales"]
    #     variable: Mapping = self.config.get("variable", {})
    #     params: Mapping = self.config.get("params", {})
    #     var_names = list(self.config['variable'].keys())
    
    #     #base_vals = {k: best_design.results[k] for k in var_names if k in best_design.results}
    #     base_vals = best_design
    #     lut = {fn: (bn, factor) for fn, (bn, factor) in scales.items()}
    
    #     L_param = params.get(length_param_name, None)
    #     L_numeric = float(L_param) if L_param is not None else None
    
    #     w_pat = re.compile(r'(\bW\s*=\s*)\{(W_[A-Za-z0-9_]+)\}', flags=re.IGNORECASE)
    #     l_pat = re.compile(r'(\bL\s*=\s*)\{(' + re.escape(length_param_name) + r')\}', flags=re.IGNORECASE)
    #     tx_line_pat = re.compile(r'^\s*x[mn]\w*\b', flags=re.IGNORECASE)
    #     nf_present_pat = re.compile(r'\bnf\s*=\s*\d+', flags=re.IGNORECASE)
    
    #     def nm_to_e7(value_nm: float) -> str:
    #         """Convert nm to '(value*10)e-7' string form."""
    #         return f"{value_nm * 10:.6g}e-7"
    
    #     out_lines = []
    #     for line in subckt_tmpl.splitlines():
    #         new_line = line
    
    #         # Replace W={W_xxx}
    #         def _sub_w(m):
    #             prefix, final_w = m.group(1), m.group(2)
    #             if final_w in lut:
    #                 base_name, factor = lut[final_w]
    #                 if base_name not in base_vals:
    #                     raise KeyError(f"Missing base value for '{base_name}'")
    #                 val_nm = base_vals[base_name]  # still in nm
    #                 return f"{prefix}{nm_to_e7(val_nm)}"
    #             return m.group(0)
    
    #         new_line = w_pat.sub(_sub_w, new_line)
    
    #         # Replace L={L}
    #         def _sub_l(m):
    #             prefix = m.group(1)
    #             if L_numeric is None:
    #                 return m.group(0)
    #             return f"{prefix}{nm_to_e7(L_numeric)}"
    
    #         new_line = l_pat.sub(_sub_l, new_line)
    
    #         # Add nf=<scale> if missing
    #         if tx_line_pat.match(new_line) and not nf_present_pat.search(new_line):
    #             nf = 1
    #             for final_name, (base_name, factor) in lut.items():
    #                 if f"{{{final_name}}}" in line:
    #                     nf = factor
    #                     break
    #             new_line = new_line.rstrip() + f" nf={nf}"
    
    #         out_lines.append(new_line)
    
    #     align_text = "\n".join(out_lines) + "\n"
    
    #     # Normalize exponent: ensure e-07 -> e-7
    #     align_text = re.sub(r'e-0(\d)', r'e-\1', align_text)
    
    #     Path(spice_file).write_text(align_text)
    #     self.log(f"Saved SPICE netlist: {spice_file}")
    
    def run_align(self, spice_file: Path, iteration: int):
        

        """Run ALIGN to generate GDS layout"""
        print(f"RUNNING ALIGN FOR ITERATION {iteration}", "SECTION")
        
        pdk_path = self.config['align_pdk_path']
        results_dir =  Path(self.config["results_dir"])
        output_dir = results_dir / f"iteration_{iteration:02d}_align"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        design_dir = output_dir / "design"
        design_dir.mkdir(exist_ok=True)
        
        # Copy netlist
        temp_netlist = design_dir / f"{self.config.get('subckt_name')}.sp"
        shutil.copy2(spice_file, temp_netlist)
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir(output_dir)
            design_dir_relative = design_dir.relative_to(output_dir)
            align_cmd = f"schematic2layout.py {design_dir_relative} -p {pdk_path}"
            
            self.log(f"Command: {align_cmd}")
            
            result = subprocess.run(
                align_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            os.chdir(original_cwd)
            
            # Save log
            log_file = output_dir / 'align.log'
            with open(log_file, 'w') as f:
                f.write(f"Command: {align_cmd}\n")
                f.write("="*80 + "\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write(result.stderr)
            
            # Find GDS files (.python.gds)
            gds_files = list(output_dir.rglob("*.python.gds"))
            
            if result.returncode == 0 and gds_files:
                self.log(f"✓ ALIGN successful! Generated: {gds_files[0].name}", "INFO")
                return gds_files[0]
            else:
                self.log("✗ ALIGN failed or no GDS generated", "ERROR")
                self.log(f"Check log: {log_file}", "ERROR")
                return None
                
        except Exception as e:
            os.chdir(original_cwd)
            self.log(f"✗ ALIGN error: {e}", "ERROR")
            return None
    
    def run_pex(self, gds_path: Path, iteration: int):
    
        """Run Magic PEX for parasitic extraction"""
        self.log(f"RUNNING PEX FOR ITERATION {iteration}", "SECTION")
        
        pex_script = self.config['pex_script_path']
        results_dir = Path(self.config['results_dir'])
        output_dir = results_dir / f"iteration_{iteration:02d}_pex/"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pex_netlist = output_dir / "pex_output.spice"
        
        # Convert all paths to absolute paths
        gds_path_abs = gds_path.absolute()
        output_dir_abs = output_dir.absolute()
        pex_netlist_abs = pex_netlist.absolute()
        
        cmd = [
            'python3', pex_script,
            '-g', str(gds_path_abs),
            '-t', self.config.get("subckt_name"),
            '-o', str(pex_netlist_abs),
            '-d', str(output_dir_abs)+"/"
        ]
        
        try:
            self.log(f"GDS: {gds_path_abs}")
            self.log(f"Output: {pex_netlist_abs}")
            self.log(f"Command: {' '.join(cmd)}")
            
            # Run from the directory containing the PEX script
            pex_script_dir = Path(pex_script).parent
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(pex_script_dir)  # Run from PEX script directory
            )
            
            # Save log
            log_file = output_dir / 'pex.log'
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Working directory: {pex_script_dir}\n")
                f.write("="*80 + "\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write(result.stderr)
            
            # Check if output file was created
            if pex_netlist_abs.exists():
                file_size = pex_netlist_abs.stat().st_size
                self.log(f"✓ PEX successful! Generated: {pex_netlist.name} ({file_size} bytes)", "INFO")
                return pex_netlist
            else:
                print("✗ PEX failed - output file not created", "ERROR")
                print(f"Check log: {log_file.absolute()}", "ERROR")
                
                # Print relevant error messages
                if "Error" in result.stdout or "Error" in result.stderr:
                    self.log("Key errors found:", "ERROR")
                    for line in (result.stdout + result.stderr).split('\n'):
                        if 'Error' in line or 'FAILED' in line:
                            print(f"  {line}")
                
                return None
                
        except Exception as e:
            self.log(f"✗ PEX error: {e}", "ERROR")
            return None


    def calculate_degradation(self, pre: OTAResult, post: OTAResult):
        """
        Calculate performance degradation for metrics defined in config['metrics'].
        Returns {metric_name_percent: value}.
        """

        if post is None:
            return {}
        
        degradations = {}
        metrics = self.config.get("metrics", [])  # from YAML
    
        pre_results = pre.results
        post_results = post.results
    
        for metric in metrics:
            if metric in pre_results and metric in post_results:
                pre_val = pre_results[metric]
                post_val = post_results[metric]
    
                # Only handle valid numeric entries
                if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)) and pre_val != 0:
                    degradations[f"{metric}_percent"] = ((post_val - pre_val) / pre_val) * 100
    
        return degradations
     
    def extract_area_from_gds(self, gds_path):
        """Extract area from GDS layout using gdspy or klayout"""
        import gdspy
        
        try:
            # Read GDS file
            gdsii = gdspy.GdsLibrary()
            gdsii.read_gds(str(gds_path))
            print(str(gds_path))
            
            # Get top cell
            top_cell = gdsii.top_level()[0]
            
            # Get bounding box
            bbox = top_cell.get_bounding_box()
            
            if bbox is not None:
                width = bbox[1][0] - bbox[0][0]  # µm
                height = bbox[1][1] - bbox[0][1]  # µm
                area = width * height  # µm²
                
                return {
                    'width': width,
                    'height': height,
                    'area': area
                }
        except Exception as e:
            print(f"Error extracting area: {e}")
            return None
        
    
    def run_iteration(self, iteration: int, n_samples: int, trial_index: int,
                      search_method='random', previous_best=None, algorithm_params: dict = None,
                      pre_layout_only: bool = False):
        """
        Run one complete iteration
        
        Parameters:
        -----------
        iteration: int
            Current iteration number
        n_samples: int
            Number of designs to search
        search_method: str
            Search algorithm to use
        previous_best: OTAResult, optional
            Best design from previous iteration
        algorithm_params: dict, optional
            Parameters for the search algorithm
        pre_layout_only: bool
            If True, skip ALIGN layout and post-PEX verification (faster)
        """

        if self.start_time is None:
            self.start_time = time.time()
        
        iter_start = time.time()

        
        self.log(f"ITERATION {iteration}", "SECTION")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        var_names = list(self.config['variable'].keys())
    
        if algorithm_params:
            print(f"\n🔧 Algorithm parameters for iteration {iteration}:")
            for param_name, param_value in algorithm_params.items():
                print(f"   - {param_name}: {param_value}")
        else:
            print(f"\n⚠️ No custom parameters (using defaults)")
        
        # Show mode
        if pre_layout_only:
            print(f" Mode: PRE-LAYOUT ONLY (skipping ALIGN and post-PEX)")
        else:
            print(f" Mode: FULL FLOW (including ALIGN and post-PEX)")
        
        # Step 1: Search N designs (ALWAYS RUN)
        self.log(f"STEP 1: SEARCHING {n_samples} DESIGNS", "SECTION")
        if algorithm_params is None:
            algorithm_params = {}
        
        results = self.search_designs(
            n_samples=n_samples,
            method=search_method,
            previous_best=previous_best,
            algorithm_params=algorithm_params
        )
        
        if not results:
            self.log("No valid designs found!", "ERROR")
            return None
        
        best_design = results[0]
        best_design_var = {k: best_design.results[k] for k in var_names if k in best_design.results}
        
        self.log(f"\nBest design found: {best_design}", "INFO")
        
        # ========================================================================
        # CONDITIONAL: LAYOUT AND POST-PEX (based on pre_layout_only flag)
        # ========================================================================
        if pre_layout_only:
            # Skip ALIGN and post-PEX - return early with just pre-layout
            self.log(f"\n⚡ PRE-LAYOUT ONLY MODE: Skipping ALIGN and post-PEX", "INFO")
            
            # Create iteration result with only pre-layout data
            iter_result = IterationResult(
                iteration=iteration,
                pre_layout=best_design,
                post_pex=None,  # No post-PEX
                gds_path=None,  # No layout
                pex_netlist_path=None,  # No PEX
                degradation_percent={},  # No degradation
                num_designs_searched=n_samples,
                timestamp=timestamp,
                method=search_method
            )
            
            self.iteration_history.append(iter_result)
            self.save_summary(trial_index=trial_index)

            iter_end = time.time()
            iter_elapsed = iter_end - iter_start
            self.iteration_times.append(iter_elapsed)
            
            # Add time to iter_result
            iter_result.elapsed_time = iter_elapsed
            iter_result.cumulative_time = time.time() - self.start_time
        
            
            return iter_result
        
        # ========================================================================
        # FULL FLOW: ALIGN + POST-PEX
        # ========================================================================
        
        # Step 2: Save and run ALIGN
        self.log(f"STEP 2: GENERATING LAYOUT WITH ALIGN", "SECTION")
        spice_file = self.results_dir / f"iteration_{iteration:02d}_ota.sp"
        self.save_spice_for_align(best_design_var, spice_file)
        
        gds_path = self.run_align(spice_file, iteration)
        if not gds_path:
            self.log("ALIGN failed - stopping iteration", "ERROR")
            # Return with just pre-layout
            iter_result = IterationResult(
                iteration=iteration,
                pre_layout=best_design,
                post_pex=None,
                gds_path=None,
                pex_netlist_path=None,
                degradation_percent={},
                num_designs_searched=n_samples,
                timestamp=timestamp,
                method=search_method
            )
            self.iteration_history.append(iter_result)
            return iter_result
        
        gds_path = gds_path.absolute()
        
        # Extract area from GDS
        area_info = self.extract_area_from_gds(gds_path)
        if area_info:
            best_design.area = round(area_info['area'] * 1e12, 1) 
            best_design.calculate_fom_per_area()
            
            self.log(f"Layout Area: {area_info['area']:.2e} × 1e12 = {area_info['area']*1e12:.1f} µm²", "INFO")
            self.log(f"  Width: {area_info['width']:.2e} × 1e6 = {area_info['width']*1e6:.1f} µm", "INFO")
            self.log(f"  Height: {area_info['height']:.2e} × 1e6 = {area_info['height']*1e6:.1f} µm", "INFO")
            self.log(f"FOM per Area: {(best_design.fom_per_area or 0):.4f}", "INFO")
        else:
            self.log("Could not extract area from GDS - continuing without area metrics", "WARNING")
        
        # Step 3: Run PEX
        self.log(f"STEP 3: EXTRACTING PARASITICS WITH PEX", "SECTION")
        pex_netlist = self.run_pex(gds_path, iteration)
        if not pex_netlist:
            self.log("PEX failed - stopping iteration", "ERROR")
            # Return with pre-layout and layout but no post-PEX
            iter_result = IterationResult(
                iteration=iteration,
                pre_layout=best_design,
                post_pex=None,
                gds_path=gds_path,
                pex_netlist_path=None,
                degradation_percent={},
                num_designs_searched=n_samples,
                timestamp=timestamp,
                method=search_method
            )
            self.iteration_history.append(iter_result)
            return iter_result
        
        # Step 4: Re-simulate with PEX
        self.log(f"STEP 4: RE-SIMULATING WITH PARASITICS", "SECTION")
        best_trial_values = [best_design.results[v] for v in var_names]
        post_pex_result = self.simulate_ota_config(
            trial_values=best_trial_values,
            netlist_path=pex_netlist
        )
        
        if not post_pex_result:
            self.log("✗ Post-PEX simulation failed", "ERROR")
            degradation = {}
        else:
            # Copy area info to post-PEX result
            if best_design.area:
                post_pex_result.area = best_design.area
                post_pex_result.calculate_fom_per_area()
            
            degradation = self.calculate_degradation(best_design, post_pex_result)
            self.log(f"✓ Post-PEX performance:", "INFO")
            print(f"     {post_pex_result}")
            
            self.log(f"\nDegradation due to parasitics:", "INFO")
            # Automatically print all degradation metrics
            for key, value in degradation.items():
                # Format key for readability
                label = key.replace("_percent", "").replace("_", " ").title()
                print(f"     {label}: {value:+.1f}%")
            
            # Highlight target metric from user query
            target_metric = self.llm_agent.target_metric

            
            if target_metric['is_composite']:
                # Compute composite metric values
                pre_value = self.llm_agent._compute_composite_metric(best_design.to_dict(), target_metric)
                post_value = self.llm_agent._compute_composite_metric(post_pex_result.to_dict(), target_metric)
                degradation_pct = ((post_value - pre_value) / pre_value * 100) if pre_value != 0 else 0
                
                self.log(f"\n⭐ PRIMARY METRIC ({target_metric['metric_name']}): {degradation_pct:+.1f}% change", "INFO")
                print(f"     Pre-layout: {pre_value:.3f}")
                print(f"     Post-PEX:   {post_value:.3f}")
            else:
                degradation_key = target_metric['degradation_key']
                if degradation_key in degradation:
                    deg_value = degradation[degradation_key]
                    self.log(f"\n⭐ PRIMARY METRIC ({target_metric['metric_name']}): {deg_value:+.1f}% degradation", "INFO")
                    
                    metric_key = target_metric['metric_key']
                    pre_value = getattr(best_design, metric_key, 'N/A')
                    post_value = getattr(post_pex_result, metric_key, 'N/A')
                    unit = target_metric['metric_unit']
                    fmt = target_metric['metric_format']
                    
                    print(f"     Pre-layout: {pre_value:{fmt}} {unit}")
                    print(f"     Post-PEX:   {post_value:{fmt}} {unit}")
            
            # Check if meets user specs
            meets_specs = self._check_user_constraints(post_pex_result)
            if meets_specs:
                self.log(f"✓ Post-PEX design MEETS user specifications!", "INFO")
        
        # Store results with full flow data
        iter_result = IterationResult(
            iteration=iteration,
            pre_layout=best_design,
            post_pex=post_pex_result,
            gds_path=gds_path,
            pex_netlist_path=pex_netlist,
            degradation_percent=degradation,
            num_designs_searched=n_samples,
            timestamp=timestamp,
            method=search_method
        )
        
        self.iteration_history.append(iter_result)
        self.save_summary(trial_index=trial_index)

        iter_end = time.time()
        iter_elapsed = iter_end - iter_start
        self.iteration_times.append(iter_elapsed)
            
        # Add time to iter_result
        iter_result.elapsed_time = iter_elapsed
        iter_result.cumulative_time = time.time() - self.start_time
        
        
        return iter_result

    
    def save_summary(self, trial_index: int = 0):
        """
        Save optimization summary with all designs from each iteration
        
        Parameters:
        -----------
        trial_index: int
            Index of the current trial/run (for multi-trial experiments)
        """
        # Save individual trial summary
        summary_file = self.results_dir / f"optimization_summary_trial_{trial_index}.json"
        
        # Get target metric info
        target_metric = self.llm_agent.target_metric

        # Try to load trial summary for total time
        trial_summary_file = self.results_dir / f'trial_{trial_index}_summary.json'
        if trial_summary_file.exists():
            with open(trial_summary_file, 'r') as f:
                trial_summary = json.load(f)
                total_time = trial_summary['total_time_seconds']
        else:
            # Fallback: calculate from iteration times
            total_time = sum(getattr(iter_result, 'elapsed_time', 0) 
                            for iter_result in self.iteration_history)
        
        # Calculate metrics for this trial
        best_fom = None
        evals_to_best = 0
        total_time = 0
        success = False
        
        # Track cumulative evaluations and find best
        cumulative_evals = 0
        start_time = None
        time_to_best = 0
        
        for iter_result in self.iteration_history:
            cumulative_evals += iter_result.num_designs_searched
            
            # Get iteration timestamp
            if start_time is None:
                start_time = datetime.strptime(iter_result.timestamp, "%Y%m%d_%H%M%S")
            
            current_time = datetime.strptime(iter_result.timestamp, "%Y%m%d_%H%M%S")
            elapsed = (current_time - start_time).total_seconds()
            
            # Get FOM from post-PEX if available, otherwise pre-layout
            if iter_result.post_pex:
                current_fom = iter_result.post_pex.fom
            else:
                current_fom = iter_result.pre_layout.fom
            
            # Track best
            if best_fom is None or current_fom > best_fom:
                best_fom = current_fom
                evals_to_best = cumulative_evals
                time_to_best = iter_result.cumulative_time
            
        
        # Check success (did we meet user specs with best design?)
        final_result = self.iteration_history[-1]
        if final_result.post_pex:
            success = self._check_user_constraints(final_result.post_pex)
        else:
            success = self._check_user_constraints(final_result.pre_layout)
        
        summary = {
            'trial_index': trial_index,
            'user_specs': self.user_specs,
            'target_metric': {
                'name': target_metric['metric_name'],
                'key': target_metric['metric_key'],
                'unit': target_metric['metric_unit'],
                'direction': target_metric['direction'],
                'is_composite': target_metric['is_composite']
            },
            'config': self.config,
            'total_designs_searched': len(self.all_searched_designs),
            
            # Summary metrics for this trial
            'metrics': {
                'best_fom': best_fom,
                'evals_to_best': evals_to_best,
                'time_to_best_seconds': time_to_best,
                'total_time_seconds': total_time,
                'success': success,
                'num_iterations': len(self.iteration_history),
                'total_evaluations': cumulative_evals
            },
            
            'iterations': []
        }
        
        # Track which designs belong to which iteration
        design_idx = 0
        
        for iter_result in self.iteration_history:
            # Get all designs for this iteration
            iteration_designs = []
            for i in range(iter_result.num_designs_searched):
                if design_idx < len(self.all_searched_designs):
                    design = self.all_searched_designs[design_idx]
                    design_dict = {'design_number': i + 1, **design.to_dict()}
                    
                    # Add target metric value
                    if target_metric['is_composite']:
                        design_dict['target_metric_value'] = self.llm_agent._compute_composite_metric(
                            design.to_dict(), target_metric)
                    else:
                        design_dict['target_metric_value'] = design.to_dict().get(target_metric['metric_key'])
                    
                    iteration_designs.append(design_dict)
                    design_idx += 1
            
            iter_data = {
                'iteration': iter_result.iteration,
                'timestamp': iter_result.timestamp,
                'method': iter_result.method,
                'num_designs_searched': iter_result.num_designs_searched,
                'all_designs': iteration_designs,
                'best_design_pre_layout': iter_result.pre_layout.to_dict(),
                'gds_path': str(iter_result.gds_path) if iter_result.gds_path else None,
                'pex_netlist_path': str(iter_result.pex_netlist_path) if iter_result.pex_netlist_path else None,
            }
            
            # Add target metric values for best designs
            if target_metric['is_composite']:
                iter_data['best_pre_target_metric'] = self.llm_agent._compute_composite_metric(
                    iter_result.pre_layout.to_dict(), target_metric)
            else:
                iter_data['best_pre_target_metric'] = iter_result.pre_layout.to_dict().get(target_metric['metric_key'])
            
            if iter_result.post_pex:
                iter_data['best_design_post_pex'] = iter_result.post_pex.to_dict()
                iter_data['degradation'] = iter_result.degradation_percent
                
                if target_metric['is_composite']:
                    iter_data['best_post_target_metric'] = self.llm_agent._compute_composite_metric(
                        iter_result.post_pex.to_dict(), target_metric)
                else:
                    iter_data['best_post_target_metric'] = iter_result.post_pex.to_dict().get(target_metric['metric_key'])
            
            summary['iterations'].append(iter_data)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Summary saved to: {summary_file}", "INFO")
        
        return summary['metrics']  # Return metrics for aggregation
    
    def print_summary(self):
        """Print summary of all iterations"""
        print("\n" + "="*100)
        print("OPTIMIZATION SUMMARY")
        print("="*100)
        print(f"Total designs searched across all iterations: {len(self.all_searched_designs)}")
        print(f"Number of iterations completed: {len(self.iteration_history)}")
        
        # Extract target metric from user specs
        target_metric = self.llm_agent.target_metric
        
        for iter_result in self.iteration_history:
            print(f"\n{'─'*100}")
            print(f"Iteration {iter_result.iteration} ({iter_result.num_designs_searched} designs searched)")
            print(f"{'─'*100}")
            print(f"  Pre-layout:  {iter_result.pre_layout}")
            
            if iter_result.post_pex:
                print(f"  Post-PEX:    {iter_result.post_pex}")
                
                # Show target metric degradation
                if target_metric['is_composite']:
                    pre_value = self.llm_agent._compute_composite_metric(
                        iter_result.pre_layout.to_dict(), target_metric)
                    post_value = self.llm_agent._compute_composite_metric(
                        iter_result.post_pex.to_dict(), target_metric)
                    deg_pct = ((post_value - pre_value) / pre_value * 100) if pre_value != 0 else 0
                    print(f"  Target Metric ({target_metric['metric_name']}): {deg_pct:+.1f}%")
                else:
                    deg_key = target_metric['degradation_key']
                    if deg_key in iter_result.degradation_percent:
                        print(f"  Target Metric ({target_metric['metric_name']}): "
                              f"{iter_result.degradation_percent[deg_key]:+.1f}%")
            
            if iter_result.gds_path:
                print(f"  GDS: {iter_result.gds_path}")
        
        print("="*100)