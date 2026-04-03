import os
import subprocess
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class RingOscResult:
    # Design parameters
    W_pmos0: float
    W_nmos0: float
    W_pmos1: float
    W_nmos1: float
    W_pmos2: float
    W_nmos2: float
    L_inv: float
    n_stages: int
    
    # Performance metrics
    frequency_mhz: Optional[float]
    period_ns: Optional[float]
    power_uw: Optional[float]
    delay_per_stage_ps: Optional[float]

def simulate_ring_oscillator(pdk_lib_path,
                             L_inv,
                             W_pmos0, W_nmos0,
                             W_pmos1, W_nmos1,
                             W_pmos2, W_nmos2,
                             vdd=1.8,
                             temp=27,
                             c_load=10e-15,
                             results_dir='./results',
                             **kwargs):  # Catch any extra params
    """
    Simulate 3-stage ring oscillator with individual transistor sizing
    
    Parameters:
    -----------
    pdk_lib_path: str
        Path to PDK library
    L_inv: float
        Channel length for all transistors (shared)
    W_pmos0, W_nmos0: float
        PMOS and NMOS widths for stage 0
    W_pmos1, W_nmos1: float
        PMOS and NMOS widths for stage 1
    W_pmos2, W_nmos2: float
        PMOS and NMOS widths for stage 2
    vdd: float
        Supply voltage
    temp: float
        Temperature in Celsius
    c_load: float
        Load capacitance in Farads
    results_dir: str
        Directory for simulation results
    """
    
    n_stages = 3
    
    # Create simulation directory
    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        results_file = 'ring_osc_results.txt'
        
        netlist = f"""* 3-Stage Ring Oscillator with Individual Transistor Sizing
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp}

* Stage 0: N0 → N1
XMP0 N1 N0 VDD VDD sky130_fd_pr__pfet_01v8 l={L_inv} w={W_pmos0}
XMN0 N1 N0 GND GND sky130_fd_pr__nfet_01v8 l={L_inv} w={W_nmos0}

* Stage 1: N1 → N2
XMP1 N2 N1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_inv} w={W_pmos1}
XMN1 N2 N1 GND GND sky130_fd_pr__nfet_01v8 l={L_inv} w={W_nmos1}

* Stage 2: N2 → N0 (closes the ring)
XMP2 N0 N2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_inv} w={W_pmos2}
XMN2 N0 N2 GND GND sky130_fd_pr__nfet_01v8 l={L_inv} w={W_nmos2}

* Load capacitor
CL N0 GND {c_load}

* Supply
VDD VDD GND DC {vdd}

* Initial condition to help start oscillation
.ic v(N0)={vdd}

.control
set noaskquit
set wr_singlescale
set wr_vecnames
option numdgt=7

* Transient to observe oscillation
tran 1p 50n

* Measure oscillation frequency
meas tran v_first WHEN v(N0)={vdd/2} RISE=1
meas tran v_second WHEN v(N0)={vdd/2} RISE=2
let period = v_second - v_first

echo "PERIOD:" > {results_file}
print period >> {results_file}

* Measure average power
let i_avg = mean(-i(VDD))

echo "I_AVG:" >> {results_file}
print i_avg >> {results_file}

* Save waveform for verification
wrdata {results_file}_waveform.txt time v(N0) v(N1) v(N2)

quit
.endc

.end
"""

        netlist_path = 'ring_osc_sim.spice'
        with open(netlist_path, 'w') as f:
            f.write(netlist)

        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print("NGSPICE ERRORS:")
            print(result.stderr)
            return None

        # Parse results
        period = None
        i_avg = None
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if 'PERIOD:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        period = float(match.group(1))
                
                elif 'I_AVG:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        i_avg = abs(float(match.group(1)))
        
        if period is None or period <= 0:
            return None
        
        # Calculate metrics
        frequency_hz = 1.0 / period
        frequency_mhz = frequency_hz / 1e6
        period_ns = period * 1e9
        power_uw = i_avg * vdd * 1e6 if i_avg else None
        delay_per_stage_ps = (period / (2 * n_stages)) * 1e12
        
        return RingOscResult(
            W_pmos0=W_pmos0,
            W_nmos0=W_nmos0,
            W_pmos1=W_pmos1,
            W_nmos1=W_nmos1,
            W_pmos2=W_pmos2,
            W_nmos2=W_nmos2,
            L_inv=L_inv,
            n_stages=n_stages,
            frequency_mhz=frequency_mhz,
            period_ns=period_ns,
            power_uw=power_uw,
            delay_per_stage_ps=delay_per_stage_ps
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        os.chdir(original_dir)