import os
import subprocess
import re
import math
import numpy as np
import itertools
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json
from datetime import datetime

# Result dataclass
@dataclass
class VCOResult:
    # Size variables
    W_inv_p: float
    W_inv_n: float
    L_inv_p: float
    L_inv_n: float

    # Performance metrics
    freq_hz: float
    power_uw: float
    v_ctrl: float

    # Optional metrics (populated if sweep performed)
    freq_at_min_v: float = None
    freq_at_max_v: float = None
    tuning_range_percent: float = None
    vco_gain_MHz_per_V: float = None


def _simulate_vco_single_point(pdk_lib_path, W_inv_p, W_inv_n, L_inv_p, L_inv_n,
                                v_ctrl, vdd, temp):
    """Helper: simulate VCO at single control voltage point"""

    num_stages = 5
    tran_file = f'vco_tran_{v_ctrl:.3f}.txt'

    # Build netlist
    netlist_stages = ""
    for i in range(num_stages):
        current_node = f"N{i}"
        next_node = f"N{(i+1) % num_stages}"

        # Stage 0 slightly weaker to break symmetry
        if i == 0:
            w_p = W_inv_p * 0.9
            w_n = W_inv_n
        else:
            w_p = W_inv_p
            w_n = W_inv_n

        netlist_stages += f"""
*** Stage {i}
XMPINV{i} {next_node} {current_node} VDD VDD sky130_fd_pr__pfet_01v8 l={L_inv_p} w={w_p} m=1
XMNINV{i} {next_node} {current_node} GND GND sky130_fd_pr__nfet_01v8 l={L_inv_n} w={w_n} m=1
"""

    # Variable capacitance based on v_ctrl
    cap_base = 5e-15
    cap_variable = (v_ctrl / vdd) * 50e-15
    cap_total = cap_base + cap_variable

    netlist = f"""* VCO Single Point
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp}

{netlist_stages}

*** Load capacitors
C0 N0 GND {cap_total}
C1 N1 GND {cap_total}
C2 N2 GND {cap_total}
C3 N3 GND {cap_total}
C4 N4 GND {cap_total}

*** Power supply
VSUP VDD GND PWL(0 0 10n {vdd})

*** Symmetry breaking
RBREAK N0 GND 1G

.control
tran 1n 20u
print v(N3) > {tran_file}

* Measure average current through power supply
meas tran i_avg AVG i(VSUP) from=15u to=20u

quit
.endc

.end
"""

    netlist_path = f'vco_sim_{v_ctrl:.3f}.spice'
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"    ✗ ngspice error at V_ctrl={v_ctrl:.3f}V")
            return None

        # Parse frequency from transient output
        freq = None
        if os.path.exists(tran_file):
            time_vals = []
            volt_vals = []

            with open(tran_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or 'Index' in line or 'v(n3)' in line or '---' in line or line.startswith('*') or 'Transient' in line or 'Analysis' in line:
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        t = float(parts[1])
                        v = float(parts[2])
                        if t >= 10e-6:
                            time_vals.append(t)
                            volt_vals.append(v)
                    except (ValueError, IndexError):
                        continue

            if len(volt_vals) > 20:
                v_arr = np.array(volt_vals)
                t_arr = np.array(time_vals)

                v_range = v_arr.max() - v_arr.min()
                if v_range < vdd * 0.3:
                    print(f"✗ Not osc (range={v_range:.3f}V)", end="")
                    return None

                v_mid = vdd / 2
                crossings = np.where(np.diff(np.sign(v_arr - v_mid)) > 0)[0]

                if len(crossings) >= 3:
                    periods = np.diff(t_arr[crossings])
                    avg_period = np.mean(periods)
                    freq = 1.0 / avg_period
                    print(f"✓ {freq/1e6:.3f} MHz", end="")
                else:
                    print(f"✗ {len(crossings)} crossings", end="")
                    return None
            else:
                print(f"✗ {len(volt_vals)} pts", end="")
                return None
        else:
            print(f"✗ No file", end="")
            return None

        # Parse power - look for i_avg measurement
        power_uw = 0.0
        for line in result.stdout.split('\n'):
            if 'i_avg' in line.lower():
                try:
                    if '=' in line:
                        parts = line.split('=')
                        current_str = parts[-1].strip().split()[0]
                        current_a = float(current_str)
                        power_w = abs(current_a) * vdd
                        power_uw = power_w * 1e6
                        print(f", {power_uw:.3f} uW", end="")
                        break
                except:
                    pass

        if power_uw == 0.0:
            print(", power N/A", end="")

        if freq is None:
            return None

        return VCOResult(
            W_inv_p=W_inv_p, W_inv_n=W_inv_n,
            L_inv_p=L_inv_p, L_inv_n=L_inv_n,
            freq_hz=freq, power_uw=power_uw, v_ctrl=v_ctrl
        )

    except Exception as e:
        print(f"✗ Exception: {e}")
        return None


def simulate_vco(pdk_lib_path,
                 W_inv_p_base=1.0, W_inv_n_base=0.5,
                 L_inv_p_base=0.15, L_inv_n_base=0.15,
                 v_ctrl_min=0.0, v_ctrl_max=1.8,
                 num_v_ctrl_points=5,
                 vdd=1.8, temp=27,
                 results_dir='./results',
                 **kwargs):
    """
    Simulate VCO with characterization sweep.
    Returns dict with tuning range, gain, power, etc.
    """

    W_inv_p = float(W_inv_p_base)
    W_inv_n = float(W_inv_n_base)
    L_inv_p = float(L_inv_p_base)
    L_inv_n = float(L_inv_n_base)

    v_ctrl_vals = np.linspace(v_ctrl_min, v_ctrl_max, num_v_ctrl_points)

    frequencies = []
    powers = []
    valid_v_ctrl = []

    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(sim_dir)

    print(f"  Characterizing VCO with {num_v_ctrl_points} control voltage points...")

    try:
        for i, v_ctrl in enumerate(v_ctrl_vals):
            print(f"    [{i+1}/{num_v_ctrl_points}] V_ctrl={v_ctrl:.3f}V ", end="")

            result_single = _simulate_vco_single_point(
                pdk_lib_path, W_inv_p, W_inv_n, L_inv_p, L_inv_n,
                v_ctrl, vdd, temp
            )
            print()  # newline after each point

            if result_single and result_single.freq_hz > 0:
                frequencies.append(result_single.freq_hz)
                powers.append(result_single.power_uw)
                valid_v_ctrl.append(v_ctrl)

        if len(frequencies) < 2:
            print(f"  ✗ Not enough valid points ({len(frequencies)}/2 minimum)")
            return None

        # Calculate characterization metrics
        frequencies = np.array(frequencies)
        powers = np.array(powers)
        valid_v_ctrl = np.array(valid_v_ctrl)

        freq_at_min = frequencies[0]
        freq_at_max = frequencies[-1]
        f_max = max(frequencies)
        f_min = min(frequencies)
        f_center = (f_max + f_min) / 2
        tuning_range_percent = abs((f_max - f_min) / f_center * 100)

        mid_idx = len(powers) // 2
        power_uw = powers[mid_idx]

        local_gains = np.diff(frequencies) / np.diff(valid_v_ctrl) / 1e6
        vco_gain_MHz_per_V = abs(np.mean(local_gains))

        # Print summary
        print(f"  VCO Summary: freq={f_center/1e6:.1f}MHz, tuning={tuning_range_percent:.1f}%, "
              f"power={power_uw:.1f}uW, Kvco={vco_gain_MHz_per_V:.1f}MHz/V")

        return {
            'W_inv_p': W_inv_p,
            'W_inv_n': W_inv_n,
            'L_inv_p': L_inv_p,
            'L_inv_n': L_inv_n,
            'freq_at_min': freq_at_min,
            'freq_at_max': freq_at_max,
            'frequency_mhz': f_center / 1e6,
            'tuning_range_percent': tuning_range_percent,
            'power_uw': power_uw,
            'vco_gain_MHz_per_V': vco_gain_MHz_per_V,
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pdk_lib = './pdk/sky130.lib.spice'

    results = simulate_vco(
        pdk_lib_path=pdk_lib,
        W_inv_p_base=1.0,
        W_inv_n_base=0.5,
        L_inv_p_base=0.15,
        L_inv_n_base=0.15,
        vdd=1.8,
        temp=27,
        v_ctrl_min=0.0,
        v_ctrl_max=1.8,
        num_v_ctrl_points=10,
        results_dir='./vco_test'
    )

    if results:
        print("\n✓ Characterization complete")
        for k, v in results.items():
            print(f"  {k}: {v}")
