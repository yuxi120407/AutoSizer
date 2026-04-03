import os
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class DiodeLoadAmpResult:
    W_input: float
    W_load: float
    L_input: float
    L_load: float
    # Performance metrics
    gain_db: Optional[float]  # Voltage gain (dB)
    bandwidth_mhz: Optional[float]  # 3dB bandwidth (MHz)
    power_uw: Optional[float]  # DC power consumption (µW)
    output_swing_v: Optional[float]  # Output voltage swing (V)

def simulate_diode_load_amp(pdk_lib_path,
                            W_input=5.0,
                            W_load=2.0,
                            L_input=0.5,
                            L_load=0.5,
                            vdd=1.8,
                            vbias=0.6,
                            temp=27,
                            c_load=10e-15,
                            results_dir='./results',
                            **kwargs):
    """
    Simulate diode-connected load amplifier (common-source with diode load)
    
    Circuit:
    - NMOS input transistor (common-source)
    - PMOS diode-connected load (gate tied to drain)
    - Measures gain, bandwidth, power
    """
    
    # Create simulation directory
    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        results_file = 'diode_load_amp_results.txt'
        
        netlist = f"""* Diode Load Amplifier
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp}

* Input transistor (NMOS common-source)
XM_INPUT VOUT VIN GND GND sky130_fd_pr__nfet_01v8 l={L_input} w={W_input}

* Diode-connected load (PMOS)
XM_LOAD VOUT VOUT VDD VDD sky130_fd_pr__pfet_01v8 l={L_load} w={W_load}

* Load capacitor
CL VOUT GND {c_load}

* DC bias + AC signal (combined in one source)
VIN VIN GND DC {vbias} AC 0.001

* Supply
VDD VDD GND DC {vdd}

.control
set noaskquit
set wr_singlescale
set wr_vecnames
option numdgt=7

* Operating point
op

echo "I_DD:" > {results_file}
print -i(VDD) >> {results_file}

echo "VOUT_DC:" >> {results_file}
print v(VOUT) >> {results_file}

echo "VIN_DC:" >> {results_file}
print v(VIN) >> {results_file}

* AC analysis
ac dec 100 1 10G

* Save AC data
wrdata {results_file}_ac.txt frequency db(v(VOUT))

quit
.endc

.end
"""

        netlist_path = 'diode_load_amp_sim.spice'
        with open(netlist_path, 'w') as f:
            f.write(netlist)

        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print("NGSPICE ERRORS:")
            print(result.stderr)
            return None

        # Parse results from file
        i_dd = None
        vout_dc = None
        vin_dc = None
        gain_db = None
        bw_hz = None
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            # Parse line by line
            for i, line in enumerate(lines):
                line = line.strip()
                
                if 'I_DD:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        i_dd = abs(float(match.group(1)))
                
                elif 'VOUT_DC:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        vout_dc = float(match.group(1))
                
                elif 'VIN_DC:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        vin_dc = float(match.group(1))

        # Parse AC data file
        ac_file = f"{results_file}_ac.txt"
        
        if os.path.exists(ac_file):
            with open(ac_file, 'r') as f:
                ac_lines = f.readlines()
            
            # Parse AC data
            frequencies = []
            gains = []
            
            for line in ac_lines:
                line = line.strip()
                if not line or 'frequency' in line.lower():
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        freq = float(parts[0])
                        gain = float(parts[3])  # Column 3 is db(v(VOUT))
                        frequencies.append(freq)
                        gains.append(gain)
                    except ValueError:
                        continue
            
            if len(gains) > 0:
                # DC gain (first point)
                gain_db_raw = gains[0]
                
                # Correct for 1mV input: add 60dB
                gain_db = gain_db_raw + 60
                
                # Find 3dB bandwidth
                gain_3db = gain_db - 3
                
                for i in range(len(gains)-1):
                    gain_corrected_i = gains[i] + 60
                    gain_corrected_i1 = gains[i+1] + 60
                    
                    if gain_corrected_i > gain_3db and gain_corrected_i1 <= gain_3db:
                        # Linear interpolation
                        f1, g1 = frequencies[i], gain_corrected_i
                        f2, g2 = frequencies[i+1], gain_corrected_i1
                        bw_hz = f1 + (gain_3db - g1) * (f2 - f1) / (g2 - g1)
                        break
        
        # Calculate metrics
        power_uw = i_dd * vdd * 1e6 if i_dd else None
        bandwidth_mhz = bw_hz / 1e6 if bw_hz and bw_hz > 0 else None
        
        # Output swing (estimate from DC bias point)
        output_swing_v = min(vout_dc, vdd - vout_dc) if vout_dc else None
        
        # print(f"\n{'='*70}")
        # print(f"DIODE LOAD AMPLIFIER SIMULATION RESULTS")
        # print(f"{'='*70}")
        # print(f"  Gain:         {gain_db:.2f} dB" if gain_db else "  Gain:         NOT MEASURED")
        # print(f"  Bandwidth:    {bandwidth_mhz:.2f} MHz" if bandwidth_mhz else "  Bandwidth:    NOT MEASURED")
        # print(f"  Power:        {power_uw:.3f} µW" if power_uw else "  Power:        NOT MEASURED")
        # print(f"  Out Swing:    {output_swing_v:.4f} V" if output_swing_v else "  Out Swing:    NOT MEASURED")
        # print(f"{'='*70}")
        
        if gain_db is not None:
            return DiodeLoadAmpResult(
                W_input=W_input,
                W_load=W_load,
                L_input=L_input,
                L_load=L_load,
                gain_db=gain_db,
                bandwidth_mhz=bandwidth_mhz,
                power_uw=power_uw,
                output_swing_v=output_swing_v
            )
        
        return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        os.chdir(original_dir)