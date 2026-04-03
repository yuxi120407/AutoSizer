import os
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class XORResult:
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

def simulate_xor_gate(pdk_lib_path,
                      W_pmos=1.0, W_nmos=0.5,
                      L_pmos=0.15, L_nmos=0.15,
                      vdd=1.8, temp=27,
                      freq=100e6,
                      c_load=10e-15,
                      results_dir='./results',
                      **kwargs):
    """
    Simulate 2-input XOR gate using standard CMOS logic
    XOR = A⊕B = (A+B)·(A'·B') = (A·B') + (A'·B)
    """
    
    # Create simulation directory
    sim_dir = os.path.join(results_dir, 'ngspice_sim')
    os.makedirs(sim_dir, exist_ok=True)

    freq = float(freq)
    
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        period = 1.0 / freq
        t_rise = 100e-12
        t_fall = 100e-12
        
        results_file = 'xor_results.txt'
        
        # XOR using CMOS logic gates
        # XOR = (A + B) · (A' + B') using De Morgan's laws
        netlist = f"""* 2-Input XOR Gate (Transmission Gate - FIXED)
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp}

* Inverters for A and B
XMP_INVA A_N A VDD VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={W_pmos}
XMN_INVA A_N A GND GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={W_nmos}

XMP_INVB B_N B VDD VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={W_pmos}
XMN_INVB B_N B GND GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={W_nmos}

* Transmission gate 1: passes A when B=0
XMP_TG1 N_TG B_N A VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={{{W_pmos}*2}}
XMN_TG1 N_TG B A GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={{{W_nmos}*2}}

* Transmission gate 2: passes A' when B=1
XMP_TG2 N_TG B A_N VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={{{W_pmos}*2}}
XMN_TG2 N_TG B_N A_N GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={{{W_nmos}*2}}

* Output buffer (reduces load on transmission gates)
XMP_BUF Y N_TG VDD VDD sky130_fd_pr__pfet_01v8 l={L_pmos} w={{{W_pmos}*3}}
XMN_BUF Y N_TG GND GND sky130_fd_pr__nfet_01v8 l={L_nmos} w={{{W_nmos}*3}}

* Load capacitor
CL Y GND {c_load}

* Input signals
VA A GND PWL(0 0 1n {vdd} 30n {vdd})
VB B GND PWL(0 0 6n 0 7n {vdd} 16n {vdd} 17n 0 26n 0)

* Supply
VDD VDD GND DC {vdd}

.control
set noaskquit

op
let i_static = abs(i(VDD))

echo "I_STATIC:" > {results_file}
print i_static >> {results_file}

tran 10p 30n

let i_vdd = -i(VDD)

meas tran v_b_50_rise WHEN v(B)={vdd/2} CROSS=1
meas tran v_y_50_fall WHEN v(Y)={vdd/2} CROSS=1
let tphl_val = abs(v_y_50_fall - v_b_50_rise) * 1e12

echo "TPHL_PS:" >> {results_file}
print tphl_val >> {results_file}

meas tran v_b_50_fall WHEN v(B)={vdd/2} CROSS=2
meas tran v_y_50_rise WHEN v(Y)={vdd/2} CROSS=2
let tplh_val = abs(v_y_50_rise - v_b_50_fall) * 1e12

echo "TPLH_PS:" >> {results_file}
print tplh_val >> {results_file}

let avg_delay_val = (tphl_val + tplh_val) / 2

echo "AVG_DELAY_PS:" >> {results_file}
print avg_delay_val >> {results_file}

let i_avg = mean(abs(i_vdd))

echo "I_DYNAMIC:" >> {results_file}
print i_avg >> {results_file}

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

        netlist_path = 'xor_sim.spice'
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

        # Parse results
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
        
        # Energy per transition
        energy_per_transition_fj = None
        if i_dynamic and freq > 0:
            energy_per_transition_fj = (i_dynamic * vdd / freq) * 1e15
        
        # Noise margins
        vil = vdd * 0.3
        vih = vdd * 0.7
        noise_margin_high = (voh - vih) if voh else None
        noise_margin_low = (vil - vol) if vol else None
        
        # print(f"\n{'='*70}")
        # print(f"XOR GATE SIMULATION RESULTS")
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
            return XORResult(
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