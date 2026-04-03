import os
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class NANDResult:
    W_pmos: float
    W_nmos: float
    L_pmos: float
    L_nmos: float
    tphl_ps: Optional[float]  # Propagation delay high-to-low (ps)
    tplh_ps: Optional[float]  # Propagation delay low-to-high (ps)
    avg_delay_ps: Optional[float]  # Average propagation delay (ps)
    power_static_uw: Optional[float]  # Static power (µW)
    power_dynamic_uw: Optional[float]  # Dynamic power (µW)
    power_total_uw: Optional[float]  # Total power (µW)
    energy_per_transition_fj: Optional[float]  # Energy per switching (fJ)
    voh: Optional[float]  # Output high voltage (V)
    vol: Optional[float]  # Output low voltage (V)
    noise_margin_high: Optional[float]  # NMH (V)
    noise_margin_low: Optional[float]  # NML (V)

def simulate_nand_gate(pdk_lib_path,
                       W_pmos=1.0, W_nmos=0.5,
                       L_pmos=0.15, L_nmos=0.15,
                       vdd=1.8, temp=27,
                       freq=100e6,
                       c_load=10e-15,
                       results_dir='./results',
                       **kwargs):
    """
    Simulate 2-input NAND gate
    """


    W_pmos = float(W_pmos)
    W_nmos = float(W_nmos)
    L_pmos = float(L_pmos)
    L_nmos = float(L_nmos)
    vdd = float(vdd)
    temp = float(temp)
    freq = float(freq)  # ← Add this conversion
    c_load = float(c_load)
    
    # Create simulation directory
    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        period = 1.0 / freq  # 10ns @ 100MHz
        t_rise = 100e-12     # 100ps rise time
        t_fall = 100e-12     # 100ps fall time
        
        results_file = 'nand_results.txt'
        
        netlist = f"""* 2-Input NAND Gate
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp}

* NAND Gate: Y = NOT(A AND B)
* PMOS pull-up (parallel)
XMP1 Y A VDD VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={W_pmos}
XMP2 Y B VDD VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={W_pmos}

* NMOS pull-down (series)
XMN1 Y A N1 GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={W_nmos}
XMN2 N1 B GND GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={W_nmos}

* Load
CL Y GND {c_load}

* Supply
VDD VDD GND DC {vdd}

* Test pattern for delay measurement
* Cycle 1: A=1, B=0->1 (test tPHL when both inputs go high)
* Cycle 2: A=1, B=1->0 (test tPLH when one input goes low)
VA A GND PWL(0 0 1n {vdd} 20n {vdd})
VB B GND PWL(0 0 5n 0 6n {vdd} 15n {vdd} 16n 0 25n 0)

.control
set noaskquit

* Operating point for static power
op
let i_static = abs(i(VDD))

echo "I_STATIC:" > {results_file}
print i_static >> {results_file}

* Transient simulation
tran 10p 30n

* Save currents for power calculation
let i_vdd = -i(VDD)

* Measure tPHL (B rising at 6ns, Y should fall)
meas tran v_b_50_rise WHEN v(B)={vdd/2} CROSS=1
meas tran v_y_50_fall WHEN v(Y)={vdd/2} CROSS=1
let tphl_val = (v_y_50_fall - v_b_50_rise) * 1e12

echo "TPHL_PS:" >> {results_file}
print tphl_val >> {results_file}

* Measure tPLH (B falling at 16ns, Y should rise)
meas tran v_b_50_fall WHEN v(B)={vdd/2} CROSS=2
meas tran v_y_50_rise WHEN v(Y)={vdd/2} CROSS=2
let tplh_val = (v_y_50_rise - v_b_50_fall) * 1e12

echo "TPLH_PS:" >> {results_file}
print tplh_val >> {results_file}

* Average delay
let avg_delay_val = (tphl_val + tplh_val) / 2

echo "AVG_DELAY_PS:" >> {results_file}
print avg_delay_val >> {results_file}

* Dynamic power (average over simulation)
let i_avg = mean(abs(i_vdd))
let power_dynamic = i_avg * {vdd} * 1e6

echo "I_DYNAMIC:" >> {results_file}
print i_avg >> {results_file}

* Output voltage levels (measure actual VOH and VOL)
let voh_val = vecmax(v(Y))
let vol_val = vecmin(v(Y))

echo "VOH:" >> {results_file}
print voh_val >> {results_file}

echo "VOL:" >> {results_file}
print vol_val >> {results_file}

quit
.endc

.end
"""

        netlist_path = 'nand_sim.spice'
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
        tphl_ps = None
        tplh_ps = None
        avg_delay_ps = None
        i_static = None
        i_dynamic = None
        voh = None
        vol = None
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                content = f.read()
            
            # More robust parsing using regex on entire content
            match = re.search(r'i_static\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                i_static = abs(float(match.group(1)))
            
            match = re.search(r'tphl_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                tphl_ps = abs(float(match.group(1)))
            
            match = re.search(r'tplh_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                tplh_ps = abs(float(match.group(1)))
            
            match = re.search(r'avg_delay_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                avg_delay_ps = abs(float(match.group(1)))
            
            match = re.search(r'i_avg\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                i_dynamic = abs(float(match.group(1)))
            
            match = re.search(r'voh_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                voh = float(match.group(1))
            
            match = re.search(r'vol_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
            if match:
                vol = float(match.group(1))
        
        # Calculate metrics
        power_static_uw = i_static * vdd * 1e6 if i_static else None
        power_dynamic_uw = i_dynamic * vdd * 1e6 if i_dynamic else None
        power_total_uw = None
        
        if power_static_uw and power_dynamic_uw:
            power_total_uw = power_static_uw + power_dynamic_uw
        elif power_dynamic_uw:
            power_total_uw = power_dynamic_uw
        
        # FIXED: Energy per transition calculation
        energy_per_transition_fj = None
        if i_dynamic and freq > 0:
            # Energy = Power × Time = (I × V) × (1/freq) 
            # Convert to fJ: × 1e15
            energy_per_transition_fj = (i_dynamic * vdd / freq) * 1e15
        
        # Noise margins
        vil = vdd * 0.3
        vih = vdd * 0.7
        noise_margin_high = (voh - vih) if voh else None
        noise_margin_low = (vil - vol) if vol else None
        
        # print(f"\n{'='*70}")
        # print(f"NAND GATE SIMULATION RESULTS")
        # print(f"{'='*70}")
        # print(f"  tPHL:         {tphl_ps:.2f} ps" if tphl_ps else "  tPHL:         NOT MEASURED")
        # print(f"  tPLH:         {tplh_ps:.2f} ps" if tplh_ps else "  tPLH:         NOT MEASURED")
        # print(f"  Avg Delay:    {avg_delay_ps:.2f} ps" if avg_delay_ps else "  Avg Delay:    NOT MEASURED")
        # print(f"  Static Power: {power_static_uw:.3f} µW" if power_static_uw else "  Static Power: NOT MEASURED")
        # print(f"  Dynamic Power:{power_dynamic_uw:.3f} µW" if power_dynamic_uw else "  Dynamic Power:NOT MEASURED")
        # print(f"  Total Power:  {power_total_uw:.3f} µW" if power_total_uw else "  Total Power:  NOT MEASURED")
        # print(f"  Energy/Trans: {energy_per_transition_fj:.3f} fJ" if energy_per_transition_fj else "  Energy/Trans: NOT MEASURED")
        # print(f"  VOH:          {voh:.4f} V" if voh else "  VOH:          NOT MEASURED")
        # print(f"  VOL:          {vol:.4f} V" if vol else "  VOL:          NOT MEASURED")
        # print(f"  NMH:          {noise_margin_high:.4f} V" if noise_margin_high else "  NMH:          NOT MEASURED")
        # print(f"  NML:          {noise_margin_low:.4f} V" if noise_margin_low else "  NML:          NOT MEASURED")
        # print(f"{'='*70}")
        
        if avg_delay_ps is not None:
            return NANDResult(
                W_pmos=W_pmos, W_nmos=W_nmos,
                L_pmos=L_pmos, L_nmos=L_nmos,
                tphl_ps=tphl_ps, tplh_ps=tplh_ps,
                avg_delay_ps=avg_delay_ps,
                power_static_uw=power_static_uw,
                power_dynamic_uw=power_dynamic_uw,
                power_total_uw=power_total_uw,
                energy_per_transition_fj=energy_per_transition_fj,
                voh=voh, vol=vol,
                noise_margin_high=noise_margin_high,
                noise_margin_low=noise_margin_low
            )
        
        return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        os.chdir(original_dir)