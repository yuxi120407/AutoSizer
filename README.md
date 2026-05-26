# AutoSizer

**Automatic Sizing of Analog and Mixed-Signal Circuits via LLM Agents**

AutoSizer is a reflective, LLM-driven meta-optimization framework for automated transistor sizing of analog and mixed-signal (AMS) circuits. It combines circuit understanding, adaptive search-space construction, and multi-algorithm optimization orchestration into a closed-loop two-loop architecture powered by Google Gemini. The framework is benchmarked on 24 diverse AMS circuits in the open-source SKY130 CMOS technology (AMS-SizingBench).
> 🚧 This repository is still under construction. We welcome any suggestions or contributions for AMS-SizingBench!
---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
  - [Outer Loop — Search Space Management](#outer-loop--search-space-management)
  - [Inner Loop — Iterative Optimization](#inner-loop--iterative-optimization)
  - [Simulation & Layout Flow](#simulation--layout-flow)
- [Key Components](#key-components)
  - [LLMOptimizationAgent](#llmoptimizationagent)
  - [ControlledOTAOptimizer](#controlledotaoptimizer)
  - [AdvancedSearchMethods](#advancedsearchmethods)
- [Circuit YAML Configuration](#circuit-yaml-configuration)
- [Supported Circuits](#supported-circuits)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running a Single Circuit](#running-a-single-circuit)
  - [Running the Full Benchmark Suite](#running-the-full-benchmark-suite)
  - [Configuration Parameters](#configuration-parameters)
- [Optimization Algorithms](#optimization-algorithms)
- [FoM Definition](#fom-definition)

---


## Repository Structure

```
AutoSizer/
│
├── main.py                                  # Entry point: multi-circuit, multi-trial runner
├── llm_guided_ota_optimization_test.py      # LLM agent + outer-loop orchestration
├── iterative_ota_optimization_test.py       # Optimizer core: simulation, layout, search dispatch
├── advanced_search_methods_test.py          # All optimization algorithms
│
├── prompts.py                               # Algorithm-orchestration prompt templates
├── problem_agent_test.py                    # Circuit-understanding & search-space prompts
├── test_magic_pex.py                        # Magic PEX parasitic extraction script
├── generate_csv.py                          # Generate benchmark CSV from results
├── run_all_gemini_2_5.sh                    # Run full benchmark suite
├── requirements.txt                         # Python dependencies
│
├── utils/
│   └── feedback_extraction.py               # Parses optimization history into LLM feedback
│
├── circuit_sim/                             # Per-circuit simulation helpers
│   ├── bandgap_reference_sim.py
│   ├── diode_load_amp_sim.py
│   ├── ldo_regulator_sim.py
│   ├── nand_gate_sim.py
│   ├── resistive_load_amp_sim.py
│   ├── ring_osc_sim.py
│   ├── switched_capacitor_sim.py
│   ├── vco_characterization.py
│   ├── xor_gate_sim.py
│   ├── sim_lpf.py
│   ├── sim_hpf.py
│   └── sim_bandpf.py
│
├── AMS-SizingBench/                         # YAML configs for all 24 benchmark circuits
│   ├── telescopic_ota.yaml
│   ├── folded_cascode_ota.yaml
│   ├── five_trans_ota.yaml
│   ├── current_mirror_ota.yaml
│   ├── folded_cascode_ota_lp.yaml
│   ├── folded_cascode_ota_hp.yaml
│   ├── folded_cascode_ota_bandp.yaml
│   ├── azc.yaml
│   ├── dfcfc.yaml
│   ├── fdgb.yaml
│   ├── iac.yaml
│   ├── nmcnr.yaml
│   ├── smc.yaml
│   ├── ldo_regulator.yaml
│   ├── bandgap_reference.yaml
│   ├── voltage_controlled_osc.yaml
│   ├── switched_capacitor.yaml
│   ├── 3_stage_ring_osc.yaml
│   ├── inverter.yaml
│   ├── buffer.yaml
│   ├── nand_gate.yaml
│   ├── xor_gate.yaml
│   ├── resistive_load_amp.yaml
│   └── diode_load_amp.yaml
│
└── results/                                 # Generated at runtime
```

---

## How It Works

### Outer Loop — Search Space Management

The outer loop runs up to `max_regeneration_cycles` times. Each cycle consists of four steps.

**Step 1 — Circuit Understanding (cycle 0 only).** The LLM receives the full SPICE netlist, testbench, fixed parameters, optimization variables, and target metrics. It produces a structured JSON that identifies the circuit topology, maps each variable to its transistor(s), estimates per-metric sensitivity, and extracts key trade-offs. This output feeds directly into Step 2.

**Step 2 — Search Space Decision / Regeneration.** On the first cycle the LLM ranks all variables by impact on the target FoM, selects the top-*k* to actively optimize (the rest are fixed at recommended values), and assigns discrete search ranges for each. On subsequent cycles the LLM receives a full feedback report — convergence history, boundary-clustering analysis, variable-impact statistics, and detected issues — and decides whether to expand ranges, unfix variables, change focus, or declare convergence. The result is a compact `optimization_config` dict that every downstream component consumes.

**Step 3 — Optimizer Initialization.** A `ControlledOTAOptimizer` is instantiated with the current `optimization_config`. The optimizer sets up per-variable search spaces so that all algorithms automatically respect the LLM's variable prioritization.

**Step 4 — Inner Loop.** See below.

### Inner Loop — Iterative Optimization

Inside each outer-loop cycle, AutoSizer runs an iterative loop until one of three conditions is met: user specifications are satisfied, the FoM plateaus for `plateau_patience` consecutive iterations, or the design budget is exhausted.

Each iteration works as follows:

1. The current optimization state (all previous designs, iteration history, convergence trends) is serialized and sent to the LLM.
2. The LLM returns a structured decision: which algorithm to use, how many samples to draw, and algorithm-specific hyperparameters (mutation rate, acquisition function, temperature, etc.).
3. `run_iteration()` executes the chosen algorithm, simulates every candidate design through ngspice, and (optionally) runs the full ALIGN → PEX layout flow on the best design.
4. Plateau detection and spec-checking logic decide whether to continue, trigger an outer-loop regeneration, or stop.

### Simulation & Layout Flow

Each candidate design is evaluated by `simulate_ota_config()`, a fully generic, template-based ngspice runner. The YAML config supplies the subcircuit template, testbench template, parameter values, and scaling rules; the simulator fills in placeholders, writes a `.spice` file, invokes ngspice, and parses the output into an `OTAResult`.

When running in full-flow mode (`pre_layout_only=False`), the best pre-layout design is additionally processed through:

- **ALIGN** (`run_align()`) — generates a GDS layout from the sized netlist using `schematic2layout.py`.
- **Magic PEX** (`run_pex()`) — performs parasitic extraction on the GDS, producing a post-layout netlist.
- A second simulation run on the post-layout netlist, with pre-vs-post degradation computed for every metric.

---

## Key Components

### LLMOptimizationAgent

Defined in `llm_guided_ota_optimization_test.py`. This class owns every interaction with the Gemini API.

| Method | Purpose |
|---|---|
| `circuits_understanding()` | Sends netlist + specs to LLM; returns structured topology/sensitivity analysis |
| `search_space_generating_new()` | Builds the initial variable-ranking and range-selection prompt; returns `optimization_config` |
| `decide_next_iteration(state)` | Given full optimization state, returns `{action, method, n_samples, parameters}` |
| `llm_regenerating_searchspace(feedback)` | Outer-loop regeneration: feeds back convergence/boundary data, gets revised search space |
| `_auto_generate_fom_expression()` | Parses `user_specs_metric` string into a normalized FoM expression |

### ControlledOTAOptimizer

Defined in `iterative_ota_optimization_test.py`. Manages one complete optimization run for a single circuit.

| Method | Purpose |
|---|---|
| `simulate_ota_config(trial_values)` | Generic ngspice runner — works for any circuit whose YAML provides templates |
| `search_designs(n_samples, method, ...)` | Generates candidate points via the algorithm pool, simulates all of them, sorts by FoM |
| `run_iteration(iteration, n_samples, ...)` | One full iteration: search → (optional) ALIGN → PEX → degradation |
| `run_align(spice_file)` | Calls ALIGN `schematic2layout.py`, returns GDS path |
| `run_pex(gds_path)` | Calls Magic PEX script, returns post-layout netlist path |
| `save_summary(trial_index)` | Writes a complete per-trial JSON with every design evaluated |

**Data classes:**

- `OTAResult` — generic container for any simulation result. Stores a `results` dict and exposes keys as attributes. Computes FoM/Area if layout area is available.
- `IterationResult` — bundles the best pre-layout and post-PEX results, GDS path, degradation percentages, and timing for one iteration.

### AdvancedSearchMethods

Defined in `advanced_search_methods_test.py`. All algorithms operate natively on the discrete, per-variable search space produced by the LLM. Fixed variables are automatically injected; only the optimizable subset is varied.

| Algorithm | Key Use Case |
|---|---|
| `latin_hypercube_sampling` | Early exploration when the space is poorly understood |
| `genetic_algorithm` | Robust global search with evolutionary pressure |
| `true_bayesian_optimization` | Sample-efficient exploitation once 25+ evaluations exist; supports EI, UCB, LCB, PI acquisition |
| `optuna_bayesian_optimization` | Alternative GP-based BO via scikit-optimize |
| `simulated_annealing` | Escaping local optima / plateaus |
| `adaptive_search` | Balanced exploration–exploitation blend |
| `multi_start_local_search` | Verifying convergence from diverse starting points |

The top-level dispatcher `enhanced_generate_search_points()` routes the LLM's method choice to the correct class method and passes through all algorithm-specific hyperparameters.

---

## Circuit YAML Configuration

Every circuit in the benchmark is fully described by a single YAML file. The file supplies everything the framework needs without any hard-coded circuit logic in the Python layer.

```yaml
# AMS-SizingBench/telescopic_ota.yaml  (abbreviated)

pdk_lib_path:   ".../sky130.lib.spice"
align_pdk_path: ".../ALIGN-pdk-sky130/SKY130_PDK/"
pex_script_path: ".../test_magic_pex.py"
results_dir:    "./telescopic_ota_test_folder"

# What the user wants
user_specs: "Maximize DC gain and UGBW while minimizing power for a telescopic OTA with 1pF load."
user_specs_metric: "fom > 0.100 AND dc_gain_db > 55 AND ugbw > 10 AND power_dc < 50"

# Fixed design parameters (never optimized)
params:
  L:      0.15
  vdd:    1.8
  vcm:    0.7
  cload:  1e-12
  ibias:  10e-6

# Optimization variables (values the LLM will rank and search over)
variable:
  W_tail_base:  null
  W_diff_base:  null
  W_casc_base:  null
  W_load_base:  null
W_values: [0.84, 1.05, 1.26, 1.47, 1.68, 1.89, 2.10, 2.31, 2.52]

# How base variables map to actual transistor widths
width_scales:
  W_tail: [W_tail_base, 2]
  W_diff: [W_diff_base, 4]
  W_casc: [W_casc_base, 2]
  W_load: [W_load_base, 2]

# Subcircuit and testbench as SPICE templates (placeholders filled at runtime)
subckt_name: TELESCOPIC_OTA
ota_subckt_template: |
  .subckt TELESCOPIC_OTA ...
  xm1 ... w={W_tail} l={L}
  ...
  .ends TELESCOPIC_OTA

testbench_template: |
  .lib {pdk_lib_path} tt
  ...
  .end

# Metrics extracted from simulation output
metrics: [dc_gain_db, ugbw, power_dc, fom]
```

Key fields:

| Field | Description |
|---|---|
| `user_specs` | Natural-language goal shown to the LLM during circuit understanding |
| `user_specs_metric` | Machine-parseable constraint string; also used to auto-generate the normalized FoM |
| `variable` | Dictionary of optimization variable names (values are `null`; actual values come from `W_values` / scaling) |
| `width_scales` / `length_scales` | Maps base variables to physical device widths/lengths via `[base_var, multiplier]` pairs |
| `ota_subckt_template` | SPICE subcircuit with `{placeholder}` syntax for every parameter and variable |
| `testbench_template` | SPICE testbench; analysis commands and output extraction are embedded here |
| `metrics` | List of metric keys the simulator should parse from ngspice output |

---

## Supported Circuits

AutoSizer ships with 24 benchmark circuits across six categories, matching AMS-SizingBench from the paper.

| Category | Circuits |
|---|---|
| **Logic** | Inverter, Buffer, NAND gate, XOR gate |
| **Amplifiers (simple)** | Resistive-load amp, Diode-load amp, Five-transistor OTA, Telescopic OTA, Current-mirror OTA, Folded-cascode OTA |
| **Amplifiers (advanced)** | AZC, DFCFC, FDGB, IAC, NMCNR, SMC |
| **Oscillators** | 3-stage ring oscillator, VCO |
| **Switched-capacitor** | Switched-capacitor integrator |
| **Reference / Power / Filter** | Bandgap reference, LDO regulator, Folded-cascode OTA LPF, Folded-cascode OTA HPF, Folded-cascode OTA BPF |

---

## Prerequisites

**Software**

| Tool | Role | Required? |
|---|---|---|
| Python 3.9+ | Runtime | Yes |
| ngspice | SPICE circuit simulation | Yes |
| SKY130 PDK | Device models and layout rules | Yes |
| ALIGN | Automatic analog layout generation | Optional |
| Magic + PEX script | Parasitic extraction from GDS | Optional |

> **Note:** ALIGN and Magic PEX are only needed when running in full layout flow mode (`pre_layout_only=False`). The default mode is pre-layout only, which requires only ngspice and the SKY130 PDK. ALIGN layout support is currently available for a subset of circuits (simple OTAs and amplifiers), not all 24 benchmark circuits.

**Python packages**

```
google-generativeai
numpy
scipy
scikit-optimize   (skopt)
pyyaml
```

**API key**

A Google Gemini API key is required. Set it in the environment or add a `GOOGLE_API_KEY` field to your circuit YAML.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yuxi120407/AutoSizer.git
cd AutoSizer

# 2. Create and activate a conda environment (recommended)
conda create -n autosizer python=3.10 -y
conda activate autosizer

# 3. Install Python dependencies
pip install google-generativeai numpy scipy scikit-optimize pyyaml

# 4. Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-here"

# 5. Point paths in AMS-SizingBench/*.yaml to your local PDK and tool installations
#    (pdk_lib_path, align_pdk_path, pex_script_path)
```

---

## Usage

### Running a Single Circuit

```bash
python main.py \
    --model gemini-2.5-flash \
    --output-dir ./results/my_test \
    --circuits telescopic_ota:AMS-SizingBench/telescopic_ota.yaml
```

### Running the Full Benchmark Suite

```bash
python main.py \
    --model gemini-2.5-flash \
    --output-dir ./results/gemini-2.5-flash \
    --circuits all
```

AutoSizer will run all 24 circuits sequentially, writing a rolling `all_circuits_llm_agent_summary.json` so that interrupted runs can be resumed (completed circuits are skipped automatically).

### Configuration Parameters

These are set at the top of `main()` in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_trials` | 3 | Number of independent trials per circuit (results are averaged for the paper table) |
| `max_total_designs` | 100 | Total simulation budget per trial (shared across all outer-loop cycles) |
| `num_variables_to_optimize` | 3-6 | How many variables the LLM is allowed to actively search; the rest are fixed |
| `max_regeneration_cycles` | 3 | Maximum outer-loop iterations (search-space regenerations) |
| `plateau_patience` | 2 | Number of consecutive inner-loop iterations with < 0.1 % FoM improvement before triggering regeneration or stopping |
| `pre_layout_only` | `True` | When `False`, skips ALIGN layout and post-PEX simulation (much faster; used for the main benchmark) |

---

## Optimization Algorithms

The LLM selects among the following algorithms each iteration based on the current optimization state. All algorithms operate on the discrete search space defined by the outer loop.

| Algorithm | When the LLM Picks It | Key Hyperparameters |
|---|---|---|
| **LHS** (Latin Hypercube Sampling) | Early exploration; < 10 previous designs | `seed` |
| **Genetic** | Diverse global search; 10–50 previous designs | `mutation_rate`, `crossover_rate`, `tournament_size` |
| **Bayesian** (Gaussian Process) | Sample-efficient exploitation; 25+ previous designs | `acquisition_function` (EI / UCB / LCB / PI), `exploration_weight` |
| **Simulated Annealing** | Stuck in a plateau; suspect local optimum | `initial_temperature`, `cooling_rate` |
| **Adaptive** | Unsure whether to explore or exploit | `explore_weight`, `exploit_weight`, `random_weight` |
| **Multi-Start Local** | Late stage; verifying convergence | `n_starts`, `search_radius` |

The LLM also decides when to **stop** the inner loop (< 2 % improvement over two iterations with diverse methods, or 3+ iteration plateau).

---

## FoM Definition

The figure of merit normalizes all performance metrics against their specification targets so that circuits with different performance scales are directly comparable:

```
FoM = ∏(yᵢ / yᵢ_spec) for maximize metrics
      ─────────────────────────────────────
      ∏(yⱼ / yⱼ_spec) for minimize metrics
```

where `yᵢ_spec` is the target threshold from `user_specs_metric`. A design is **feasible** when every constraint is satisfied.

AutoSizer auto-generates this expression from the `user_specs_metric` field. For example:

```
user_specs_metric: "fom > 0.100 AND dc_gain_db > 55 AND ugbw > 10 AND power_dc < 50"
```

produces: `FoM = (dc_gain_db / 55) × (ugbw / 10) / (power_dc / 50)`.

---

*AutoSizer is designed to augment, not replace, human circuit designers. All generated designs should be validated through standard verification flows before fabrication.*
