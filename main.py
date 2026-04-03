import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from llm_guided_ota_optimization_test import run_llm_guided_optimization
from iterative_ota_optimization_test import aggregate_trial_metrics
import copy
import numpy as np

# ========================================================================
# AGGREGATION FUNCTION
# ========================================================================
def modify_yaml_results_dir(input_yaml_path, output_yaml_path, new_results_dir):
    """
    Modify only the results_dir in YAML file, preserving all formatting and structure
    """
    with open(input_yaml_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        # Check if this line contains results_dir
        if line.strip().startswith('results_dir:'):
            # Replace only the results_dir value
            indent = len(line) - len(line.lstrip())
            modified_lines.append(' ' * indent + f'results_dir: "{new_results_dir}"\n')
        else:
            modified_lines.append(line)
    
    # Write to output file
    with open(output_yaml_path, 'w') as f:
        f.writelines(modified_lines)
        
def aggregate_trial_metrics_across_dirs(base_results_dir, n_trials):
    """Aggregate metrics from separate trial directories"""
    all_metrics = []
    
    for trial_idx in range(n_trials):
        # Each trial has its own directory
        trial_dir = f"{base_results_dir}_trial_{trial_idx}"
        
        # Read from trial_X_summary.json (created by run_llm_guided_optimization)
        trial_summary_file = os.path.join(trial_dir, f"trial_{trial_idx}_summary.json")
        
        print(f"  Looking for: {trial_summary_file}")
        
        if os.path.exists(trial_summary_file):
            with open(trial_summary_file, 'r') as f:
                trial_data = json.load(f)
                
                # Extract metrics directly from trial summary
                token_usage = trial_data.get('token_usage', {})
                metrics = {
                    'best_fom': trial_data.get('best_fom'),
                    'evals_to_best': trial_data.get('evals_to_best'),  # ← From trial summary
                    'total_designs_searched': trial_data.get('total_designs_searched'),
                    'time_to_best_seconds': trial_data.get('total_time_seconds'),
                    'total_time_seconds': trial_data.get('total_time_seconds'),
                    'success': trial_data.get('specs_met', False),
                    'num_iterations': trial_data.get('num_iterations', 0),
                    'convergence_reason': trial_data.get('convergence_reason', 'unknown'),
                    # Token usage
                    'total_input_tokens': token_usage.get('total_input_tokens'),
                    'total_output_tokens': token_usage.get('total_output_tokens'),
                    'total_tokens': token_usage.get('total_tokens')
                }

                # Extract individual circuit metrics dynamically from best_design_metrics
                best_design_metrics = trial_data.get('best_design_metrics', {})
                for metric_name, metric_value in best_design_metrics.items():
                    metrics[metric_name] = metric_value

                all_metrics.append(metrics)
                print(f"       Found trial {trial_idx} metrics")
                print(f"      - Best FOM: {metrics['best_fom']}")
                print(f"      - Evals to best: {metrics['evals_to_best']}/{metrics['total_designs_searched']}")
                print(f"      - Time: {metrics['total_time_seconds']:.1f}s")
                print(f"      - Success: {metrics['success']}")
        else:
            print(f"    ✗ Not found: {trial_summary_file}")
    
    if not all_metrics:
        print(f"  No metrics found")
        return None
    
    print(f"  ✓ Successfully loaded {len(all_metrics)}/{n_trials} trials\n")

    # Extract data for statistics
    best_foms = [m['best_fom'] for m in all_metrics if m['best_fom'] is not None]
    evals_to_best = [m['evals_to_best'] for m in all_metrics if m['evals_to_best'] is not None]
    times = [m['total_time_seconds'] for m in all_metrics if m['total_time_seconds'] is not None]
    successes = [m['success'] for m in all_metrics]

    if not best_foms:
        print(f" No valid FOM values found")
        return None

    # Build aggregation results
    aggregated = {
        'avg_best_fom': float(np.mean(best_foms)),
        'std_best_fom': float(np.std(best_foms)),
        'avg_evals_to_best': float(np.mean(evals_to_best)),
        'std_evals_to_best': float(np.std(evals_to_best)),
        'avg_time_seconds': float(np.mean(times)),
        'std_time_seconds': float(np.std(times)),
        'success_rate_percent': (sum(successes) / len(successes)) * 100,
        'n_trials': len(all_metrics),
        'individual_trials': all_metrics  # Include raw data for reference
    }

    # Calculate token usage statistics
    input_tokens = [m['total_input_tokens'] for m in all_metrics if m.get('total_input_tokens') is not None]
    output_tokens = [m['total_output_tokens'] for m in all_metrics if m.get('total_output_tokens') is not None]
    total_tokens = [m['total_tokens'] for m in all_metrics if m.get('total_tokens') is not None]

    if input_tokens:
        aggregated['avg_input_tokens'] = float(np.mean(input_tokens))
        aggregated['std_input_tokens'] = float(np.std(input_tokens))
    if output_tokens:
        aggregated['avg_output_tokens'] = float(np.mean(output_tokens))
        aggregated['std_output_tokens'] = float(np.std(output_tokens))
    if total_tokens:
        aggregated['avg_total_tokens'] = float(np.mean(total_tokens))
        aggregated['std_total_tokens'] = float(np.std(total_tokens))

    # Dynamically add mean/std for all individual metrics found in best_design_metrics
    # Collect all unique metric names (excluding already-handled keys)
    excluded_keys = {'best_fom', 'evals_to_best', 'total_designs_searched', 'time_to_best_seconds',
                     'total_time_seconds', 'success', 'num_iterations', 'convergence_reason',
                     'total_input_tokens', 'total_output_tokens', 'total_tokens'}
    all_metric_names = set()
    for m in all_metrics:
        all_metric_names.update(k for k in m.keys() if k not in excluded_keys)

    # Calculate mean/std for each individual metric
    for metric_name in sorted(all_metric_names):
        values = [m[metric_name] for m in all_metrics if metric_name in m and m[metric_name] is not None]
        if values:
            aggregated[f'avg_{metric_name}'] = float(np.mean(values))
            aggregated[f'std_{metric_name}'] = float(np.std(values))

    return aggregated


def save_global_summary(all_circuit_results, summary_file="./all_circuits_llm_agent_summary.json"):
    """Save or update global summary file"""
    with open(summary_file, 'w') as f:
        json.dump(all_circuit_results, f, indent=2)
    print(f" Global summary updated: {summary_file}")
    
def main():
    """Run LLM-guided optimization for all circuits with multiple trials"""

    # ========================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ========================================================================
    parser = argparse.ArgumentParser(description='AutoSizer: LLM-guided analog circuit optimization')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model name (e.g. gemini-2.5-flash, gpt-4o, Llama-3.3-70B-Instruct)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for summary JSON (e.g. ./results/gemini-2.5-flash)')
    parser.add_argument('--circuits', type=str, nargs='+', required=True,
                        help='Circuits as name:yaml_path pairs (e.g. five_trans_ota:./AMS-SizingBench/five_trans_ota.yaml)')
    args = parser.parse_args()

    # Build CIRCUIT_REGISTRY from --circuits argument
    CIRCUIT_REGISTRY = {}
    for entry in args.circuits:
        name, path = entry.split(':', 1)
        CIRCUIT_REGISTRY[name] = {"config_path": path}

    # ========================================================================
    # GLOBAL CONFIGURATION
    # ========================================================================
    n_trials = 3
    max_total_designs = 100
    num_variables_to_optimize = 3
    max_regeneration_cycles = 3
    plateau_patience = 2
    pre_layout_only = True

    # LLM Model Selection
    # Options: "gemini-2.5-flash", "claude-sonnet-4-5", "gpt-4o", "Llama-3.3-70B-Instruct"
    model_name = args.model if args.model else "gemini-2.5-flash"

    # Output directory for results
    output_dir = args.output_dir if args.output_dir else "."
    os.makedirs(output_dir, exist_ok=True)
    global_summary_file = os.path.join(output_dir, "all_circuits_llm_agent_summary.json")
    
    # Load existing results if file exists (for resuming)
    if os.path.exists(global_summary_file):
        print(f"Found existing summary file, loading previous results...")
        with open(global_summary_file, 'r') as f:
            all_circuit_results = json.load(f)
    else:
        all_circuit_results = {}
    
    print(f"\n{'='*80}")
    print(f"MULTI-CIRCUIT MULTI-TRIAL LLM-GUIDED OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Total circuits: {len(CIRCUIT_REGISTRY)}")
    print(f"Trials per circuit: {n_trials}")
    print(f"LLM Model: {model_name}")
    print(f"max_regeneration_cycles: {max_regeneration_cycles}")
    print(f"Initial num_variables_to_optimize: {num_variables_to_optimize}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # LOOP THROUGH ALL CIRCUITS
    # ========================================================================
    for circuit_idx, (circuit_name, circuit_info) in enumerate(CIRCUIT_REGISTRY.items()):
        print(f"\n{'#'*80}")
        print(f"# CIRCUIT {circuit_idx + 1}/{len(CIRCUIT_REGISTRY)}: {circuit_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Skip if already completed
        if circuit_name in all_circuit_results and all_circuit_results[circuit_name].get('status') == 'SUCCESS':
            print(f"⏭️  {circuit_name} already completed, skipping...\n")
            continue
        
        config_path = circuit_info['config_path']
        
        # Check if config exists
        if not os.path.exists(config_path):
            print(f" Config file not found: {config_path}")
            all_circuit_results[circuit_name] = {
                "status": "FAILED",
                "error": "Config file not found",
                "timestamp": datetime.now().isoformat()
            }
            save_global_summary(all_circuit_results, global_summary_file)
            continue
        
        # Load base config to get original results_dir
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        original_results_dir = os.path.join(output_dir, circuit_name, "test")
        
        print(f"Config loaded from: {config_path}")
        print(f"Base results dir: {original_results_dir}\n")
        
        # ====================================================================
        # RUN MULTIPLE TRIALS FOR THIS CIRCUIT
        # ====================================================================
        trial_results = []
        
        for trial_idx in range(n_trials):
            print(f"\n{'-'*80}")
            print(f"  CIRCUIT: {circuit_name} | TRIAL {trial_idx + 1}/{n_trials}")
            print(f"{'-'*80}\n")
            
            # Create trial-specific results directory
            trial_results_dir = f"{original_results_dir}_trial_{trial_idx}"
            os.makedirs(trial_results_dir, exist_ok=True)
            
            # Create modified YAML file (only results_dir changed)
            trial_config_path = os.path.join(trial_results_dir, f"{circuit_name}_trial_{trial_idx}.yaml")
            modify_yaml_results_dir(config_path, trial_config_path, trial_results_dir)
            
            print(f"  Trial config saved → {trial_config_path}")
            print(f"  Results directory → {trial_results_dir}\n")
            
            # Load the modified config for this trial
            with open(trial_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize trial result entry
            trial_result = {
                "trial": trial_idx,
                "results_dir": trial_results_dir,
                "config_path": trial_config_path
            }
            
            try:
                # Run optimization
                optimizer = run_llm_guided_optimization(
                                    config=trial_config_path,
                                    max_total_designs=max_total_designs,
                                    num_variables_to_optimize=num_variables_to_optimize,
                                    pre_layout_only=True,
                                    max_regeneration_cycles=max_regeneration_cycles,
                                    plateau_patience=plateau_patience,
                                    trial_index=trial_idx,
                                    model=model_name
                                )
                
                # Mark as success
                trial_result["status"] = "SUCCESS"
                trial_result["timestamp"] = datetime.now().isoformat()
                
                print(f"\n  {circuit_name} Trial {trial_idx + 1}/{n_trials} completed")
                
            except KeyboardInterrupt:
                print(f"\n  Interrupted by user")
                
                # Mark trial as interrupted
                trial_result["status"] = "INTERRUPTED"
                trial_result["timestamp"] = datetime.now().isoformat()
                trial_results.append(trial_result)
                
                # Save partial results
                all_circuit_results[circuit_name] = {
                    "status": "INTERRUPTED",
                    "trials": trial_results,
                    "timestamp": datetime.now().isoformat()
                }
                save_global_summary(all_circuit_results, global_summary_file)
                raise
                
            except Exception as e:
                print(f"\n  {circuit_name} Trial {trial_idx + 1}/{n_trials} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Mark trial as failed
                trial_result["status"] = "FAILED"
                trial_result["error"] = str(e)
                trial_result["timestamp"] = datetime.now().isoformat()
            
            # Append result after try-except (always executed unless KeyboardInterrupt)
            trial_results.append(trial_result)

        
        # ====================================================================
        # AGGREGATE RESULTS FOR THIS CIRCUIT
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"AGGREGATING RESULTS FOR {circuit_name.upper()}")
        print(f"{'='*80}\n")
        
        aggregated = aggregate_trial_metrics_across_dirs(
            base_results_dir=original_results_dir,
            n_trials=n_trials
        )
        
        if aggregated:
            print(f"\n{'='*80}")
            print(f"{circuit_name.upper()} - AGGREGATED METRICS (n={n_trials})")
            print(f"{'='*80}")
            print(f"FOM:    {aggregated['avg_best_fom']:.4f}±{aggregated['std_best_fom']:.4f}")
            print(f"Evals:  {aggregated['avg_evals_to_best']:.1f}±{aggregated['std_evals_to_best']:.1f}")
            print(f"Time:   {aggregated['avg_time_seconds']:.1f}±{aggregated['std_time_seconds']:.1f}s")
            print(f"SR%:    {aggregated['success_rate_percent']:.1f}%")
            print(f"{'='*80}\n")
            
            # Save per-circuit aggregated results
            aggregated_file = f"{original_results_dir}_aggregated_metrics.json"
            with open(aggregated_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            
            print(f"{circuit_name} aggregated results saved to: {aggregated_file}\n")
            
            all_circuit_results[circuit_name] = {
                "status": "SUCCESS",
                "metrics": aggregated,
                "trials": trial_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f" No valid results to aggregate for {circuit_name}\n")
            all_circuit_results[circuit_name] = {
                "status": "FAILED",
                "error": "No valid metrics",
                "trials": trial_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Save global summary after each circuit
        save_global_summary(all_circuit_results, global_summary_file)
        print(f"{circuit_name} results saved to global summary\n")
    
    # ========================================================================
    # FINAL SUMMARY TABLE (DYNAMIC - reads all metrics from YAML)
    # ========================================================================

    # Collect all unique metric names across all circuits
    all_metric_keys = set()
    for result in all_circuit_results.values():
        if result['status'] == 'SUCCESS' and 'metrics' in result:
            m = result['metrics']
            # Find all avg_ metrics (excluding standard ones)
            for key in m.keys():
                if key.startswith('avg_') and key not in ['avg_best_fom', 'avg_evals_to_best', 'avg_time_seconds',
                                                            'avg_input_tokens', 'avg_output_tokens', 'avg_total_tokens']:
                    all_metric_keys.add(key[4:])  # Remove 'avg_' prefix

    # Build dynamic header
    header_parts = [f"{'Circuit':<25}", f"{'FOM':<18}"]

    # Add individual metric columns
    for metric_name in sorted(all_metric_keys):
        header_parts.append(f"{metric_name:<18}")

    # Add token usage
    header_parts.extend([f"{'Tokens(in/out)':<20}", f"{'Evals':<12}", f"{'SR%':<10}"])

    table_width = sum(len(h) for h in header_parts)

    print(f"\n{'='*table_width}")
    print(f"SUMMARY TABLE")
    print(f"{'='*table_width}")
    print(" ".join(header_parts))
    print(f"{'-'*table_width}")

    for circuit_name, result in all_circuit_results.items():
        if result['status'] == 'SUCCESS' and 'metrics' in result:
            m = result['metrics']

            row_parts = [
                f"{circuit_name:<25}",
                f"{m['avg_best_fom']:.4f}±{m['std_best_fom']:.4f}      "
            ]

            # Add individual metric values dynamically
            for metric_name in sorted(all_metric_keys):
                avg_key = f'avg_{metric_name}'
                std_key = f'std_{metric_name}'
                if avg_key in m and std_key in m:
                    row_parts.append(f"{m[avg_key]:.2e}±{m[std_key]:.2e} ")
                else:
                    row_parts.append(f"{'N/A':<18}")

            # Add token usage
            if 'avg_input_tokens' in m and 'avg_output_tokens' in m:
                tokens_str = f"{m['avg_input_tokens']/1000:.1f}k/{m['avg_output_tokens']/1000:.1f}k"
            else:
                tokens_str = "N/A"

            row_parts.extend([
                f"{tokens_str:<20}",
                f"{m['avg_evals_to_best']:.1f}±{m['std_evals_to_best']:.1f}  ",
                f"{m['success_rate_percent']:.1f}"
            ])

            print("".join(row_parts))
        else:
            status = result.get('status', 'UNKNOWN')
            print(f"{circuit_name:<25} {status}")

    print(f"{'='*table_width}\n")
    print(f" ALL CIRCUITS COMPLETED\n")

if __name__ == "__main__":
    main()