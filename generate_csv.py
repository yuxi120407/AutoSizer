#!/usr/bin/env python3
"""Generate CSV from existing results JSON"""
import json
import csv
import os
import sys

def classify_circuit(name):
    """Classify circuit by type for grouping"""
    name_lower = name.lower()
    if 'ota' in name_lower or 'amplifier' in name_lower or 'amp' in name_lower:
        return 'Amplifier'
    elif 'bandgap' in name_lower or 'reference' in name_lower:
        return 'Reference'
    elif 'ldo' in name_lower or 'regulator' in name_lower:
        return 'Regulator'
    elif 'osc' in name_lower or 'vco' in name_lower:
        return 'Oscillator'
    elif 'gate' in name_lower or 'inverter' in name_lower or 'buffer' in name_lower:
        return 'Digital/Logic'
    elif 'capacitor' in name_lower or 'filter' in name_lower:
        return 'Analog Block'
    else:
        return 'Other'

# Load results
results_dir = './results/gemini-2.5-flash'
json_file = os.path.join(results_dir, 'all_circuits_llm_agent_summary.json')

with open(json_file, 'r') as f:
    all_circuit_results = json.load(f)

# Collect all unique metric names
all_metric_keys = set()
for result in all_circuit_results.values():
    if result['status'] == 'SUCCESS' and 'metrics' in result:
        m = result['metrics']
        for key in m.keys():
            if key.startswith('avg_') and key not in ['avg_best_fom', 'avg_evals_to_best', 'avg_time_seconds',
                                                        'avg_input_tokens', 'avg_output_tokens', 'avg_total_tokens']:
                all_metric_keys.add(key[4:])

csv_file = os.path.join(results_dir, "benchmark_results.csv")

# Sort circuits by type then name
sorted_circuits = sorted(
    [(name, result) for name, result in all_circuit_results.items()],
    key=lambda x: (classify_circuit(x[0]), x[0])
)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # Build header
    header = ['Circuit', 'Type', 'FOM_avg', 'FOM_std']

    # Add all metric columns (avg and std pairs)
    for metric_name in sorted(all_metric_keys):
        header.extend([f'{metric_name}_avg', f'{metric_name}_std'])

    header.extend(['Input_Tokens', 'Output_Tokens', 'Total_Tokens',
                   'Evals_avg', 'Evals_std', 'Time_avg_s', 'Time_std_s', 'Success_Rate_%'])

    writer.writerow(header)

    # Write data rows
    current_type = None
    for circuit_name, result in sorted_circuits:
        circuit_type = classify_circuit(circuit_name)

        # Add blank row when type changes
        if current_type is not None and circuit_type != current_type:
            writer.writerow([])
        current_type = circuit_type

        if result['status'] == 'SUCCESS' and 'metrics' in result:
            m = result['metrics']

            row = [
                circuit_name,
                circuit_type,
                f"{m['avg_best_fom']:.4f}",
                f"{m['std_best_fom']:.4f}"
            ]

            # Add all metric values
            for metric_name in sorted(all_metric_keys):
                avg_key = f'avg_{metric_name}'
                std_key = f'std_{metric_name}'
                if avg_key in m and std_key in m:
                    row.extend([f"{m[avg_key]:.6e}", f"{m[std_key]:.6e}"])
                else:
                    row.extend(['N/A', 'N/A'])

            # Add token usage
            row.extend([
                f"{m.get('avg_input_tokens', 0):.0f}",
                f"{m.get('avg_output_tokens', 0):.0f}",
                f"{m.get('avg_total_tokens', 0):.0f}",
                f"{m['avg_evals_to_best']:.1f}",
                f"{m['std_evals_to_best']:.1f}",
                f"{m['avg_time_seconds']:.1f}",
                f"{m['std_time_seconds']:.1f}",
                f"{m['success_rate_percent']:.1f}"
            ])

            writer.writerow(row)
        else:
            # Failed circuit
            status = result.get('status', 'UNKNOWN')
            row = [circuit_name, circuit_type, 'FAILED', status]
            writer.writerow(row)

print(f"✓ CSV exported to: {csv_file}")
