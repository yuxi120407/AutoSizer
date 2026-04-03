import os
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ResistiveLoadAmpResult:
    W_input: float
    L_input: float
    R_load: float
    # Performance metrics
    gain_db: Optional[float]  # Voltage gain (dB)
    bandwidth_mhz: Optional[float]  # 3dB bandwidth (MHz)
    power_uw: Optional[float]  # DC power consumption (µW)
    output_swing_v: Optional[float]  # Output voltage swing (V)
    input_referred_noise_nv: Optional[float]  # Input-referred noise (nV/√Hz)

def simulate_resistive_load_amp(pdk_lib_path,
                                W_input=5.0,
                                L_input=0.5,
                                R_load=50e3,
                                vdd=1.8,
                                vbias=0.6,
                                temp=27,
                                c_load=10e-15,
                                results_dir='./results',
                                **kwargs):
    """
    Simulate resistive load amplifier (common-source with R_load)
    """
    
    # Create simulation directory
    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        results_file = 'rload_amp_results.txt'
        
        netlist = f"""* Resistive Load Amplifier
        .lib {pdk_lib_path} tt
        .global VDD GND
        .temp {temp}
        
        * Input transistor (NMOS common-source)
        XM1 VOUT VIN GND GND sky130_fd_pr__nfet_01v8 l={L_input} w={W_input}
        
        * Load resistor
        RL VDD VOUT {R_load}
        
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
        set hcopydevtype=postscript
        wrdata {results_file}_ac.txt frequency db(v(VOUT))
        
        quit
        .endc
        
        .end
        """

        
        netlist_path = 'rload_amp_sim.spice'
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
        
        # print("\n=== Files created ===")
        # for f in os.listdir('.'):
        #     if 'rload' in f:
        #         print(f"  {f}")
        # print("=====================\n")
        
        # Parse results from file
        i_dd = None
        vout_dc = None
        vin_dc = None
        gain_db = None
        bw_hz = None
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            # print("\n=== DEBUG: Parsing results ===")
            # for i, line in enumerate(lines):
            #     print(f"Line {i}: {line.strip()}")
            # print("==============================\n")
            
            # Parse line by line
            for i, line in enumerate(lines):
                line = line.strip()
                
                if 'I_DD:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        i_dd = abs(float(match.group(1)))
                        #print(f"✓ Parsed i_dd = {i_dd}")
                
                elif 'VOUT_DC:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        vout_dc = float(match.group(1))
                        #print(f"✓ Parsed vout_dc = {vout_dc}")
                
                elif 'VIN_DC:' in line and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                    if match:
                        vin_dc = float(match.group(1))
                        #print(f"✓ Parsed vin_dc = {vin_dc}")
        
        # Try to parse AC data file directly
        ac_file = f"{results_file}_ac.txt"
        #print(f"\n=== Looking for AC file: {ac_file} ===")
        
        if os.path.exists(ac_file):
            #print(f"✓ Found AC data file")
            with open(ac_file, 'r') as f:
                ac_lines = f.readlines()
            
            #print(f"  Total lines: {len(ac_lines)}")
            #print(f"  First 10 lines:")
            # for i, line in enumerate(ac_lines[:10]):
            #     print(f"    {i}: {line.strip()}")
            
            # Parse AC data
            frequencies = []
            gains = []
            
            for line in ac_lines:
                line = line.strip()
                # Skip headers and empty lines
                if not line or 'frequency' in line.lower() or 'index' in line.lower():
                    continue
                
                parts = line.split()
                if len(parts) >= 4:  # ← FIXED: Need 4 columns
                    try:
                        freq = float(parts[0])
                        gain = float(parts[3])  # ← FIXED: Column 3 is db(v(VOUT))
                        frequencies.append(freq)
                        gains.append(gain)
                    except ValueError:
                        continue
            
            if len(gains) > 0:
                # DC gain (first point)
                gain_db = gains[0]
                
                # The gain is relative to 1mV AC input, so add 60dB (20*log10(1000))
                gain_db_corrected = gain_db + 60  # Convert from absolute to relative
                
                #print(f"✓ Raw Gain = {gain_db:.2f} dB (absolute)")
                #print(f"✓ DC Gain = {gain_db_corrected:.2f} dB (relative to 1mV input) @ {frequencies[0]:.1f} Hz")
                
                gain_db = gain_db_corrected  # Use corrected gain
                
                # Find 3dB bandwidth
                gain_3db = gain_db - 3
                bw_hz = None
                
                for i in range(len(gains)-1):
                    gain_corrected_i = gains[i] + 60
                    gain_corrected_i1 = gains[i+1] + 60
                    
                    if gain_corrected_i > gain_3db and gain_corrected_i1 <= gain_3db:
                        # Linear interpolation
                        f1, g1 = frequencies[i], gain_corrected_i
                        f2, g2 = frequencies[i+1], gain_corrected_i1
                        bw_hz = f1 + (gain_3db - g1) * (f2 - f1) / (g2 - g1)
                        print(f"✓ 3dB BW = {bw_hz/1e6:.2f} MHz")
                        break
                
                if bw_hz is None:
                    print(f"⚠️ 3dB point not found (gain doesn't fall by 3dB)")
            else:
                print("❌ No valid AC data found")
        else:
            print(f"❌ AC file not found: {ac_file}")
        
        # Calculate metrics
        power_uw = i_dd * vdd * 1e6 if i_dd else None
        bandwidth_mhz = bw_hz / 1e6 if bw_hz and bw_hz > 0 else None
        
        # Output swing
        output_swing_v = min(vout_dc, vdd - vout_dc) if vout_dc else None
        
        # GBW
        gain_linear = 10**(gain_db/20) if gain_db else None
        gbw_mhz = (gain_linear * bandwidth_mhz) if (gain_linear and bandwidth_mhz) else None
        
        # print(f"\n{'='*70}")
        # print(f"RESISTIVE LOAD AMPLIFIER SIMULATION RESULTS")
        # print(f"{'='*70}")
        # print(f"  Gain:         {gain_db:.2f} dB" if gain_db else "  Gain:         NOT MEASURED")
        # print(f"  Bandwidth:    {bandwidth_mhz:.2f} MHz" if bandwidth_mhz else "  Bandwidth:    NOT MEASURED")
        # print(f"  GBW:          {gbw_mhz:.2f} MHz" if gbw_mhz else "  GBW:          NOT MEASURED")
        # print(f"  Power:        {power_uw:.3f} µW" if power_uw else "  Power:        NOT MEASURED")
        # print(f"  Vin DC:       {vin_dc:.4f} V" if vin_dc else "  Vin DC:       NOT MEASURED")
        # print(f"  Vout DC:      {vout_dc:.4f} V" if vout_dc else "  Vout DC:      NOT MEASURED")
        # print(f"  Out Swing:    {output_swing_v:.4f} V" if output_swing_v else "  Out Swing:    NOT MEASURED")
        # print(f"{'='*70}")
        
        if gain_db is not None:
            return ResistiveLoadAmpResult(
                W_input=W_input,
                L_input=L_input,
                R_load=R_load,
                gain_db=gain_db,
                bandwidth_mhz=bandwidth_mhz,
                power_uw=power_uw,
                output_swing_v=output_swing_v,
                input_referred_noise_nv=None
            )
        
        return None


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        os.chdir(original_dir)