import os
import subprocess
import re
import math
from dataclasses import dataclass
from typing import Optional


# @dataclass
# class SwitchedCapResult:
#     W_op1: float
#     W_op2: float
#     W_sw: float
#     L_op: float
#     L_sw: float
#     m_op: int
#     m_sw: int
#     C_samp: float
#     C_hold: float
#     C_load: float
#     vout: float
#     gain_db: Optional[float]
#     gain_error_db: Optional[float]
#     settling_time_ns: Optional[float]
#     charge_injection_mv: Optional[float]
#     power_uw: Optional[float]
#     thd_db: Optional[float]
#     output_swing_v: Optional[float]
#     phase_margin_deg: Optional[float]
#     ugbw_mhz: Optional[float]
#     dc_gain_db: Optional[float]


# def simulate_switched_capacitor(
#     pdk_lib_path,
#     W_op1=10,
#     W_op2=10,
#     W_sw=5,
#     L_op=0.5,
#     L_sw=0.15,
#     m_op=4,
#     m_sw=2,
#     C_samp=1e-12,
#     C_hold=1e-12,
#     C_load=2e-12,
#     vdd=1.8,
#     vin_ac=0.25,
#     ibias=10e-6,
#     temp_nom=27,
#     results_dir="./results",
# ):
#     """
#     Simulate switched capacitor integrator stage - SIZING OPTIMIZATION
#     """


#     W_op1 = float(W_op1)
#     W_op2 = float(W_op2)
#     W_sw = float(W_sw)
#     L_op = float(L_op)
#     L_sw = float(L_sw)
#     m_op = int(m_op)
#     m_sw = int(m_sw)
#     C_samp = float(C_samp)
#     C_hold = float(C_hold)
#     C_load = float(C_load)
#     vdd = float(vdd)
#     vin_ac = float(vin_ac)
#     ibias = float(ibias)
#     temp_nom = float(temp_nom)
    
#     # CREATE SIMULATION DIRECTORY
#     sim_dir = os.path.join(results_dir, "ngspice_sim")
#     os.makedirs(sim_dir, exist_ok=True)

#     # Change to simulation directory
#     original_dir = os.getcwd()
#     os.chdir(sim_dir)

#     try:
#         # ------------------------------------------------------------------
#         # Parameters
#         # ------------------------------------------------------------------
#         f_clk = 1e6
#         t_period = 1.0 / f_clk
#         t_rise = t_period * 0.01
#         t_high = t_period * 0.48
#         t_fall = t_period * 0.01

#         R_bias = vdd / (ibias * 10)
#         gain_target_db = 0.0

#         vcm = vdd / 2
#         vin_dc = vcm + 0.1

#         results_file = "sc_results.txt"

#         # ------------------------------------------------------------------
#         # MAIN SC NETLIST
#         # ------------------------------------------------------------------
#         netlist = f"""* Switched Capacitor Integrator Stage
# .lib {pdk_lib_path} tt
# .global VDD GND
# .temp {temp_nom}

# .subckt SWMOD in out ctrl
# XN in ctrl out GND sky130_fd_pr__nfet_01v8 l={L_sw} w={W_sw} m={m_sw}
# XP in ctrl out VDD sky130_fd_pr__pfet_01v8 l={L_sw} w={W_sw} m={m_sw}
# .ends

# XM1 VD1 VIN_P VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
# XM2 VD2 VIN_N VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
# XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
# XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
# XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
# XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
# XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
# CC VD2 VOUT 2p

# XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
# RBIAS VDD VBIAS {R_bias}

# XSW1 VIN VSAMP CLK1 SWMOD
# XSW2 VSAMP VIN_N CLK2 SWMOD
# XSW3 VIN_P VOUT CLK2 SWMOD

# CSAMP VSAMP GND {C_samp}
# CHOLD VIN_N VOUT {C_hold}
# CLOAD VOUT GND {C_load}

# VIN VIN GND DC {vin_dc} AC {vin_ac}
# VREF VIN_P GND DC {vcm}
# VDD VDD GND DC {vdd}

# VCLK1 CLK1 GND PULSE(0 {vdd} 0 {t_rise} {t_fall} {t_high} {t_period})
# VCLK2 CLK2 GND PULSE(0 {vdd} {t_period/2} {t_rise} {t_fall} {t_high} {t_period})

# .control
# set noaskquit
# op
# tran {t_period/100} {t_period*50} {t_period*10}
# write {results_file}
# quit
# .endc
# .end
# """

#         # ------------------------------------------------------------------
#         # OTA AC NETLIST
#         # ------------------------------------------------------------------
#         ota_netlist = f"""* OTA Open-Loop AC Analysis
# .lib {pdk_lib_path} tt
# .global VDD GND
# .temp {temp_nom}

# XM1 VD1 VINP VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
# XM2 VD2 VINN VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
# XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
# XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
# XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
# XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
# XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
# CC VD2 VOUT 2p
# CL VOUT GND {C_load}

# XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
# RBIAS VDD VBIAS {R_bias}

# VINP VINP GND DC {vcm} AC 0.5
# VINN VINN GND DC {vcm} AC -0.5
# VDD VDD GND DC {vdd}

# .control
# op
# ac dec 50 1 1G
# write {results_file}
# quit
# .endc
# .end
# """

#         # ------------------------------------------------------------------
#         # Write netlists
#         # ------------------------------------------------------------------
#         with open("sc_sim.spice", "w") as f:
#             f.write(netlist)

#         with open("ota_ac.spice", "w") as f:
#             f.write(ota_netlist)

#         # ------------------------------------------------------------------
#         # Run simulations
#         # ------------------------------------------------------------------
#         print("Running switched capacitor simulation...")
#         r1 = subprocess.run(
#             ["ngspice", "-b", "sc_sim.spice"],
#             capture_output=True,
#             text=True,
#             timeout=180,
#         )

#         if r1.returncode != 0:
#             print(r1.stderr)
#             return None

#         print("Running OTA AC simulation...")
#         r2 = subprocess.run(
#             ["ngspice", "-b", "ota_ac.spice"],
#             capture_output=True,
#             text=True,
#             timeout=60,
#         )

#         if r2.returncode != 0:
#             print(r2.stderr)

#         # ------------------------------------------------------------------
#         # RESULT PARSING (unchanged logic)
#         # ------------------------------------------------------------------
#         if not os.path.exists(results_file):
#             return None

#         with open(results_file, "r") as f:
#             content = f.read()

#         def find_value(tag):
#             m = re.search(rf"{tag}.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content, re.S)
#             return float(m.group(1)) if m else None

#         vout = find_value("VOUT_FINAL")
#         gain_db = find_value("GAIN_DB")
#         dc_gain_db = find_value("DC_GAIN_DB")
#         settling_time_ns = find_value("SETTLING_TIME_NS")
#         charge_injection_mv = find_value("CHARGE_INJ_MV")
#         power_uw = find_value("POWER_UW")
#         thd_db = find_value("THD_DB")
#         output_swing = find_value("OUTPUT_SWING")
#         ugbw_mhz = find_value("UGBW_MHZ")
#         phase_margin = find_value("PHASE_MARGIN")

#         gain_error_db = abs(gain_db - gain_target_db) if gain_db is not None else None

#         if vout is None:
#             return None

#         return SwitchedCapResult(
#             W_op1=W_op1,
#             W_op2=W_op2,
#             W_sw=W_sw,
#             L_op=L_op,
#             L_sw=L_sw,
#             m_op=m_op,
#             m_sw=m_sw,
#             C_samp=C_samp,
#             C_hold=C_hold,
#             C_load=C_load,
#             vout=vout,
#             gain_db=gain_db,
#             gain_error_db=gain_error_db,
#             settling_time_ns=settling_time_ns,
#             charge_injection_mv=charge_injection_mv,
#             power_uw=power_uw,
#             thd_db=thd_db,
#             output_swing_v=output_swing,
#             phase_margin_deg=phase_margin,
#             ugbw_mhz=ugbw_mhz,
#             dc_gain_db=dc_gain_db,
#         )

#     except Exception as e:
#         print(f"Simulation error: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

#     finally:
#         # ALWAYS restore directory
#         os.chdir(original_dir)


import os
import subprocess
import re
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class SwitchedCapResult:
    W_op1: float
    W_op2: float
    W_sw: float
    L_op: float
    L_sw: float
    m_op: int
    m_sw: int
    C_samp: float
    C_hold: float
    C_load: float
    vout: float
    gain_db: Optional[float]
    gain_error_db: Optional[float]
    settling_time_ns: Optional[float]
    charge_injection_mv: Optional[float]
    power_uw: Optional[float]
    thd_db: Optional[float]
    output_swing_v: Optional[float]
    phase_margin_deg: Optional[float]
    ugbw_mhz: Optional[float]
    dc_gain_db: Optional[float]

def simulate_switched_capacitor(pdk_lib_path,
                                W_op1=10, W_op2=10, W_sw=5,
                                L_op=0.5, L_sw=0.15, 
                                m_op=4, m_sw=2,
                                C_samp=1e-12, C_hold=1e-12, C_load=2e-12,
                                vdd=1.8, vin_ac=0.25,
                                ibias=10e-6, temp_nom=27,
                                results_dir="./results",
                                **kwargs):

    """
    Simulate switched capacitor integrator stage - SIZING OPTIMIZATION
    """


    # CREATE SIMULATION DIRECTORY

    W_op1 = float(W_op1)
    W_op2 = float(W_op2)
    W_sw = float(W_sw)
    L_op = float(L_op)
    L_sw = float(L_sw)
    m_op = int(m_op)
    m_sw = int(m_sw)
    C_samp = float(C_samp)
    C_hold = float(C_hold)
    C_load = float(C_load)
    vdd = float(vdd)
    vin_ac = float(vin_ac)
    ibias = float(ibias)
    temp_nom = float(temp_nom)

    sim_dir = os.path.join(results_dir, "ngspice_sim")
    os.makedirs(sim_dir, exist_ok=True)

    # Change to simulation directory
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    try:
        # FIXED clock frequency at 1 MHz
        f_clk = 1e6
        t_period = 1.0 / f_clk
        t_rise = t_period * 0.01
        t_high = t_period * 0.48
        t_fall = t_period * 0.01
        
        # Bias resistor
        R_bias = vdd / (ibias * 10)
        
        # Target gain in dB
        gain_target_db = 0.0
        
        # DC bias points
        vcm = vdd / 2
        vin_dc = vcm + 0.1
        
        results_file = 'sc_results.txt'
        
        # Main switched capacitor simulation (unchanged)
        netlist = f"""* Switched Capacitor Integrator Stage
    .lib {pdk_lib_path} tt
    .global VDD GND
    .temp {temp_nom}
    
    * Transmission gate switch
    .subckt SWMOD in out ctrl
    XN in ctrl out GND sky130_fd_pr__nfet_01v8 l={L_sw} w={W_sw} m={m_sw}
    XP in ctrl out VDD sky130_fd_pr__pfet_01v8 l={L_sw} w={W_sw} m={m_sw}
    .ends
    
    * Two-stage OTA
    XM1 VD1 VIN_P VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
    XM2 VD2 VIN_N VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
    XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
    XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
    XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
    XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
    XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
    CC VD2 VOUT 2p
    
    * Bias
    XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
    RBIAS VDD VBIAS {R_bias}
    
    * Switched capacitor network
    XSW1 VIN VSAMP CLK1 SWMOD
    XSW2 VSAMP VIN_N CLK2 SWMOD
    XSW3 VIN_P VOUT CLK2 SWMOD
    
    CSAMP VSAMP GND {C_samp}
    CHOLD VIN_N VOUT {C_hold}
    CLOAD VOUT GND {C_load}
    
    VIN VIN GND DC {vin_dc} AC {vin_ac}
    VREF VIN_P GND DC {vcm}
    VDD VDD GND DC {vdd}
    
    VCLK1 CLK1 GND PULSE(0 {vdd} 0 {t_rise} {t_fall} {t_high} {t_period})
    VCLK2 CLK2 GND PULSE(0 {vdd} {t_period/2} {t_rise} {t_fall} {t_high} {t_period})
    
    .ic v(VOUT)={vcm} v(VSAMP)=0 v(VIN_N)={vcm}
    
    .control
    set noaskquit
    set wr_singlescale
    set wr_vecnames
    
    op
    let i_supply_op = abs(i(VDD))
    echo "I_SUPPLY:" > {results_file}
    echo "$&i_supply_op" >> {results_file}
    
    let vout_dc = v(VOUT)
    echo "VOUT_DC:" >> {results_file}
    echo "$&vout_dc" >> {results_file}
    
    let swing_top = {vdd} - vout_dc
    let swing_bot = vout_dc
    let output_swing = swing_top
    if swing_bot < swing_top
      let output_swing = swing_bot
    end
    echo "OUTPUT_SWING:" >> {results_file}
    echo "$&output_swing" >> {results_file}
    
    tran {t_period/100} {t_period*50} {t_period*10}
    
    let vout_final = v(VOUT)[length(v(VOUT))-1]
    echo "VOUT_FINAL:" >> {results_file}
    echo "$&vout_final" >> {results_file}
    
    let gain_linear = (vout_final - {vcm}) / ({vin_dc} - {vcm})
    let gain_db = 20 * log10(abs(gain_linear))
    echo "GAIN_DB:" >> {results_file}
    echo "$&gain_db" >> {results_file}
    
    let idx_start = 20 * {t_period} / {t_period/100}
    let vout_start = v(VOUT)[idx_start]
    let delta_target = 0.01 * abs(vout_final - vout_start)
    let idx = idx_start
    let idx_settled = idx_start
    while idx < length(v(VOUT)) - 1
      if abs(v(VOUT)[idx] - vout_final) < delta_target
        let idx_settled = idx
        break
      end
      let idx = idx + 1
    end
    let settling_time_ns = (time[idx_settled] - time[idx_start]) * 1e9
    echo "SETTLING_TIME_NS:" >> {results_file}
    echo "$&settling_time_ns" >> {results_file}
    
    let idx_before = floor((20 * {t_period} + {t_high}) / {t_period/100})
    let idx_after = idx_before + 10
    let charge_inj_mv = abs(v(VSAMP)[idx_before] - v(VSAMP)[idx_after]) * 1000
    echo "CHARGE_INJ_MV:" >> {results_file}
    echo "$&charge_inj_mv" >> {results_file}
    
    let i_supply_avg = mean(abs(i(VDD)))
    let power_uw = i_supply_avg * {vdd} * 1e6
    echo "POWER_UW:" >> {results_file}
    echo "$&power_uw" >> {results_file}
    
    set specwindow=hann
    linearize v(VOUT)
    let vout_ac = v(VOUT) - mean(v(VOUT))
    fft vout_ac
    let mag_fft = mag(vout_ac)
    let fund_mag = mag_fft[1]
    let harm2_mag = mag_fft[2]
    let harm3_mag = mag_fft[3]
    let harm_sum_sq = harm2_mag*harm2_mag + harm3_mag*harm3_mag
    let thd = sqrt(harm_sum_sq) / fund_mag
    let thd_db = 20 * log10(thd)
    echo "THD_DB:" >> {results_file}
    echo "$&thd_db" >> {results_file}
    
    quit
    .endc
    .end
    """
    
        # OTA AC analysis - FIXED: Just use the phase directly
        ota_netlist = f"""* OTA Open-Loop AC Analysis
    .lib {pdk_lib_path} tt
    .global VDD GND
    .temp {temp_nom}
    
    XM1 VD1 VINP VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
    XM2 VD2 VINN VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
    XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
    XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
    XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
    XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
    XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
    CC VD2 VOUT 2p
    CL VOUT GND {C_load}
    
    XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
    RBIAS VDD VBIAS {R_bias}
    
    VINP VINP GND DC {vcm} AC 0.5
    VINN VINN GND DC {vcm} AC -0.5
    VDD VDD GND DC {vdd}
    
    .control
    set noaskquit
    set wr_singlescale
    set wr_vecnames
    
    op
    ac dec 50 1 1G
    
    let gain_mag = db(v(VOUT))
    let gain_phase = phase(v(VOUT)) * 180 / pi
    let freq_vec = frequency
    
    let dc_gain = gain_mag[0]
    echo "DC_GAIN_DB:" >> {results_file}
    echo "$&dc_gain" >> {results_file}
    
    let ugbw_hz = 0
    let pm = 0
    let found = 0
    
    let i = 0
    while i < length(gain_mag) - 1
      if gain_mag[i] >= 0 & gain_mag[i+1] < 0
        let ugbw_hz = freq_vec[i]
        let phase_ugf = gain_phase[i]
        
        * FIXED: For differential OTA with AC inputs ±0.5, 
        * ngspice reports phase margin directly when phase is positive
        * If phase is 0-90°, it's likely already the phase margin
        * If phase is negative or > 90°, use standard formula
        
        if phase_ugf >= 0 & phase_ugf <= 90
          * Phase appears to already be the phase margin
          let pm = phase_ugf
        else
          * Standard phase margin calculation
          if phase_ugf > 180
            let phase_ugf = phase_ugf - 360
          end
          let pm = 180 + phase_ugf
        end
        
        let found = 1
        break
      end
      let i = i + 1
    end
    
    if found > 0
      let ugbw_mhz = ugbw_hz / 1e6
      echo "UGBW_MHZ:" >> {results_file}
      echo "$&ugbw_mhz" >> {results_file}
      echo "PHASE_MARGIN:" >> {results_file}
      echo "$&pm" >> {results_file}
    else
      echo "UGBW_MHZ:" >> {results_file}
      echo "0" >> {results_file}
      echo "PHASE_MARGIN:" >> {results_file}
      echo "0" >> {results_file}
    end
    
    quit
    .endc
    .end
    """
    
        netlist_path = 'sc_sim.spice'
        ota_netlist_path = 'ota_ac.spice'
        
        with open(netlist_path, 'w') as f:
            f.write(netlist)
        
        with open(ota_netlist_path, 'w') as f:
            f.write(ota_netlist)
    
        try:
            # Run simulations
            print("Running switched capacitor simulation...")
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode != 0:
                print("NGSPICE ERRORS (Main):")
                print(result.stderr)
                return None
    
            print("Running OTA AC analysis...")
            result_ota = subprocess.run(
                ['ngspice', '-b', ota_netlist_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result_ota.returncode != 0:
                print("NGSPICE ERRORS (OTA):")
                print(result_ota.stderr)
    
            # Parse results
            vout = None
            gain_db = None
            settling_time_ns = None
            charge_injection_mv = None
            power_uw = None
            thd_db = None
            output_swing = None
            phase_margin = None
            ugbw_mhz = None
            dc_gain_db = None
            
            print(f"\nLooking for: {results_file}")
            
            content = None
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    content = f.read()
            
            if content:
                lines_list = content.split('\n')
                for i, line in enumerate(lines_list):
                    if 'VOUT_FINAL:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            vout = float(match.group(1))
                    
                    if line.startswith('GAIN_DB:') and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            gain_db = float(match.group(1))
                    
                    if 'DC_GAIN_DB:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            dc_gain_db = float(match.group(1))
                    
                    if 'SETTLING_TIME_NS:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            settling_time_ns = float(match.group(1))
                    
                    if 'CHARGE_INJ_MV:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            charge_injection_mv = abs(float(match.group(1)))
                    
                    if 'POWER_UW:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            power_uw = float(match.group(1))
                    
                    if 'THD_DB:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            thd_db = float(match.group(1))
                    
                    if 'OUTPUT_SWING:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            output_swing = float(match.group(1))
                    
                    if 'UGBW_MHZ:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            ugbw_mhz = float(match.group(1))
                    
                    if 'PHASE_MARGIN:' in line and i+1 < len(lines_list):
                        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
                        if match:
                            phase_margin = float(match.group(1))
            
            gain_error_db = None
            if gain_db is not None:
                gain_error_db = abs(gain_db - gain_target_db)
            
            print(f"\n{'='*70}")
            print(f"SWITCHED CAPACITOR CIRCUIT - READY FOR OPTIMIZATION")
            print(f"{'='*70}")
            print(f"  SC Gain:      {gain_db:.2f} dB (target: 0±0.2 dB)" if gain_db is not None else "  SC Gain:      NOT MEASURED")
            print(f"  THD:          {thd_db:.2f} dB (target: ≤-60 dB)" if thd_db else "  THD:          NOT MEASURED")
            print(f"  Power:        {power_uw:.0f} µW (target: ≤1000 µW)" if power_uw else "  Power:        NOT MEASURED")
            print(f"  Phase Margin: {phase_margin:.1f}° (target: ≥60°)" if phase_margin else "  Phase Margin: NOT MEASURED")
            print(f"  UGBW:         {ugbw_mhz:.1f} MHz (target: ≥10 MHz)" if ugbw_mhz else "  UGBW:         NOT MEASURED")
            print(f"  Settling:     {settling_time_ns:.0f} ns" if settling_time_ns else "  Settling:     NOT MEASURED")
            print(f"{'='*70}")
    
            if vout is not None:
                return SwitchedCapResult(
                    W_op1=W_op1, W_op2=W_op2, W_sw=W_sw,
                    L_op=L_op, L_sw=L_sw,
                    m_op=m_op, m_sw=m_sw,
                    C_samp=C_samp, C_hold=C_hold, C_load=C_load,
                    vout=vout, gain_db=gain_db, 
                    gain_error_db=gain_error_db,
                    settling_time_ns=settling_time_ns,
                    charge_injection_mv=charge_injection_mv,
                    power_uw=power_uw,
                    thd_db=thd_db,
                    output_swing_v=output_swing,
                    phase_margin_deg=phase_margin,
                    ugbw_mhz=ugbw_mhz,
                    dc_gain_db=dc_gain_db
                )
    
            return None
    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        os.chdir(original_dir)
        

# import os
# import subprocess
# import re
# import math
# from dataclasses import dataclass
# from typing import Optional

# @dataclass
# class SwitchedCapResult:
#     W_op1: float
#     W_op2: float
#     W_sw: float
#     L_op: float
#     L_sw: float
#     m_op: int
#     m_sw: int
#     C_samp: float
#     C_hold: float
#     C_load: float
#     vout: float
#     gain_db: Optional[float]
#     gain_error_db: Optional[float]
#     settling_time_ns: Optional[float]
#     charge_injection_mv: Optional[float]
#     power_uw: Optional[float]
#     thd_db: Optional[float]
#     output_swing_v: Optional[float]
#     phase_margin_deg: Optional[float]
#     ugbw_mhz: Optional[float]
#     dc_gain_db: Optional[float]

# def simulate_switched_capacitor(pdk_lib_path,
#                                 W_op1=10, W_op2=10, W_sw=5,
#                                 L_op=0.5, L_sw=0.15, 
#                                 m_op=4, m_sw=2,
#                                 C_samp=1e-12, C_hold=1e-12, C_load=2e-12,
#                                 vdd=1.8, vin_ac=0.25,
#                                 ibias=10e-6, temp_nom=27,
#                                 results_dir='./results'):
#     """
#     Simulate switched capacitor integrator stage - SIZING OPTIMIZATION
#     """

#     # CREATE SIMULATION DIRECTORY using results_dir from YAML
#     sim_dir = os.path.join(results_dir, 'ngspice_sim')
#     os.makedirs(sim_dir, exist_ok=True)
    
#     # Change to simulation directory
#     original_dir = os.getcwd()
#     os.chdir(sim_dir)

#     try:
#         pdk_dir = os.path.dirname(pdk_lib_path)
        
#         # FIXED clock frequency at 1 MHz
#         f_clk = 1e6
#         t_period = 1.0 / f_clk
#         t_rise = t_period * 0.01
#         t_high = t_period * 0.48
#         t_fall = t_period * 0.01
        
#         # Bias resistor
#         R_bias = vdd / (ibias * 10)
        
#         # Target gain in dB
#         gain_target_db = 0.0
        
#         # DC bias points
#         vcm = vdd / 2
#         vin_dc = vcm + 0.1
        
#         results_file = 'sc_results.txt'
        
#         # Main switched capacitor simulation (unchanged)
#         netlist = f"""* Switched Capacitor Integrator Stage
#     .lib {pdk_lib_path} tt
#     .global VDD GND
#     .temp {temp_nom}
    
#     * Transmission gate switch
#     .subckt SWMOD in out ctrl
#     XN in ctrl out GND sky130_fd_pr__nfet_01v8 l={L_sw} w={W_sw} m={m_sw}
#     XP in ctrl out VDD sky130_fd_pr__pfet_01v8 l={L_sw} w={W_sw} m={m_sw}
#     .ends
    
#     * Two-stage OTA
#     XM1 VD1 VIN_P VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
#     XM2 VD2 VIN_N VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
#     XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
#     XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
#     XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
#     XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
#     XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
#     CC VD2 VOUT 2p
    
#     * Bias
#     XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
#     RBIAS VDD VBIAS {R_bias}
    
#     * Switched capacitor network
#     XSW1 VIN VSAMP CLK1 SWMOD
#     XSW2 VSAMP VIN_N CLK2 SWMOD
#     XSW3 VIN_P VOUT CLK2 SWMOD
    
#     CSAMP VSAMP GND {C_samp}
#     CHOLD VIN_N VOUT {C_hold}
#     CLOAD VOUT GND {C_load}
    
#     VIN VIN GND DC {vin_dc} AC {vin_ac}
#     VREF VIN_P GND DC {vcm}
#     VDD VDD GND DC {vdd}
    
#     VCLK1 CLK1 GND PULSE(0 {vdd} 0 {t_rise} {t_fall} {t_high} {t_period})
#     VCLK2 CLK2 GND PULSE(0 {vdd} {t_period/2} {t_rise} {t_fall} {t_high} {t_period})
    
#     .ic v(VOUT)={vcm} v(VSAMP)=0 v(VIN_N)={vcm}
    
#     .control
#     set noaskquit
#     set wr_singlescale
#     set wr_vecnames
    
#     op
#     let i_supply_op = abs(i(VDD))
#     echo "I_SUPPLY:" > {results_file}
#     echo "$&i_supply_op" >> {results_file}
    
#     let vout_dc = v(VOUT)
#     echo "VOUT_DC:" >> {results_file}
#     echo "$&vout_dc" >> {results_file}
    
#     let swing_top = {vdd} - vout_dc
#     let swing_bot = vout_dc
#     let output_swing = swing_top
#     if swing_bot < swing_top
#       let output_swing = swing_bot
#     end
#     echo "OUTPUT_SWING:" >> {results_file}
#     echo "$&output_swing" >> {results_file}
    
#     tran {t_period/100} {t_period*50} {t_period*10}
    
#     let vout_final = v(VOUT)[length(v(VOUT))-1]
#     echo "VOUT_FINAL:" >> {results_file}
#     echo "$&vout_final" >> {results_file}
    
#     let gain_linear = (vout_final - {vcm}) / ({vin_dc} - {vcm})
#     let gain_db = 20 * log10(abs(gain_linear))
#     echo "GAIN_DB:" >> {results_file}
#     echo "$&gain_db" >> {results_file}
    
#     let idx_start = 20 * {t_period} / {t_period/100}
#     let vout_start = v(VOUT)[idx_start]
#     let delta_target = 0.01 * abs(vout_final - vout_start)
#     let idx = idx_start
#     let idx_settled = idx_start
#     while idx < length(v(VOUT)) - 1
#       if abs(v(VOUT)[idx] - vout_final) < delta_target
#         let idx_settled = idx
#         break
#       end
#       let idx = idx + 1
#     end
#     let settling_time_ns = (time[idx_settled] - time[idx_start]) * 1e9
#     echo "SETTLING_TIME_NS:" >> {results_file}
#     echo "$&settling_time_ns" >> {results_file}
    
#     let idx_before = floor((20 * {t_period} + {t_high}) / {t_period/100})
#     let idx_after = idx_before + 10
#     let charge_inj_mv = abs(v(VSAMP)[idx_before] - v(VSAMP)[idx_after]) * 1000
#     echo "CHARGE_INJ_MV:" >> {results_file}
#     echo "$&charge_inj_mv" >> {results_file}
    
#     let i_supply_avg = mean(abs(i(VDD)))
#     let power_uw = i_supply_avg * {vdd} * 1e6
#     echo "POWER_UW:" >> {results_file}
#     echo "$&power_uw" >> {results_file}
    
#     set specwindow=hann
#     linearize v(VOUT)
#     let vout_ac = v(VOUT) - mean(v(VOUT))
#     fft vout_ac
#     let mag_fft = mag(vout_ac)
#     let fund_mag = mag_fft[1]
#     let harm2_mag = mag_fft[2]
#     let harm3_mag = mag_fft[3]
#     let harm_sum_sq = harm2_mag*harm2_mag + harm3_mag*harm3_mag
#     let thd = sqrt(harm_sum_sq) / fund_mag
#     let thd_db = 20 * log10(thd)
#     echo "THD_DB:" >> {results_file}
#     echo "$&thd_db" >> {results_file}
    
#     quit
#     .endc
#     .end
#     """
    
#         # OTA AC analysis - FIXED: Just use the phase directly
#         ota_netlist = f"""* OTA Open-Loop AC Analysis
#     .lib {pdk_lib_path} tt
#     .global VDD GND
#     .temp {temp_nom}
    
#     XM1 VD1 VINP VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
#     XM2 VD2 VINN VTAIL GND sky130_fd_pr__nfet_01v8 l={L_op} w={W_op1} m={m_op}
#     XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1} m={m_op}
#     XM3 VD1 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
#     XM4 VD2 VD1 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2} m={m_op}
#     XM5 VOUT VD2 VDD VDD sky130_fd_pr__pfet_01v8 l={L_op} w={W_op2*4} m={m_op}
#     XM6 VOUT VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*2} w={W_op1*2} m={m_op/2}
#     CC VD2 VOUT 2p
#     CL VOUT GND {C_load}
    
#     XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_op*4} w={W_op1} m=1
#     RBIAS VDD VBIAS {R_bias}
    
#     VINP VINP GND DC {vcm} AC 0.5
#     VINN VINN GND DC {vcm} AC -0.5
#     VDD VDD GND DC {vdd}
    
#     .control
#     set noaskquit
#     set wr_singlescale
#     set wr_vecnames
    
#     op
#     ac dec 50 1 1G
    
#     let gain_mag = db(v(VOUT))
#     let gain_phase = phase(v(VOUT)) * 180 / pi
#     let freq_vec = frequency
    
#     let dc_gain = gain_mag[0]
#     echo "DC_GAIN_DB:" >> {results_file}
#     echo "$&dc_gain" >> {results_file}
    
#     let ugbw_hz = 0
#     let pm = 0
#     let found = 0
    
#     let i = 0
#     while i < length(gain_mag) - 1
#       if gain_mag[i] >= 0 & gain_mag[i+1] < 0
#         let ugbw_hz = freq_vec[i]
#         let phase_ugf = gain_phase[i]
        
#         * FIXED: For differential OTA with AC inputs ±0.5, 
#         * ngspice reports phase margin directly when phase is positive
#         * If phase is 0-90°, it's likely already the phase margin
#         * If phase is negative or > 90°, use standard formula
        
#         if phase_ugf >= 0 & phase_ugf <= 90
#           * Phase appears to already be the phase margin
#           let pm = phase_ugf
#         else
#           * Standard phase margin calculation
#           if phase_ugf > 180
#             let phase_ugf = phase_ugf - 360
#           end
#           let pm = 180 + phase_ugf
#         end
        
#         let found = 1
#         break
#       end
#       let i = i + 1
#     end
    
#     if found > 0
#       let ugbw_mhz = ugbw_hz / 1e6
#       echo "UGBW_MHZ:" >> {results_file}
#       echo "$&ugbw_mhz" >> {results_file}
#       echo "PHASE_MARGIN:" >> {results_file}
#       echo "$&pm" >> {results_file}
#     else
#       echo "UGBW_MHZ:" >> {results_file}
#       echo "0" >> {results_file}
#       echo "PHASE_MARGIN:" >> {results_file}
#       echo "0" >> {results_file}
#     end
    
#     quit
#     .endc
#     .end
#     """
    
#         netlist_path = 'sc_sim.spice'
#         ota_netlist_path = 'ota_ac.spice'
        
#         with open(netlist_path, 'w') as f:
#             f.write(netlist)
        
#         with open(ota_netlist_path, 'w') as f:
#             f.write(ota_netlist)
    
#         try:
#             # Run simulations
#             print("Running switched capacitor simulation...")
#             result = subprocess.run(
#                 ['ngspice', '-b', netlist_path],
#                 capture_output=True,
#                 text=True,
#                 timeout=180
#             )
            
#             if result.returncode != 0:
#                 print("NGSPICE ERRORS (Main):")
#                 print(result.stderr)
#                 return None
    
#             print("Running OTA AC analysis...")
#             result_ota = subprocess.run(
#                 ['ngspice', '-b', ota_netlist_path],
#                 capture_output=True,
#                 text=True,
#                 timeout=60
#             )
            
#             if result_ota.returncode != 0:
#                 print("NGSPICE ERRORS (OTA):")
#                 print(result_ota.stderr)
    
#             # Parse results
#             vout = None
#             gain_db = None
#             settling_time_ns = None
#             charge_injection_mv = None
#             power_uw = None
#             thd_db = None
#             output_swing = None
#             phase_margin = None
#             ugbw_mhz = None
#             dc_gain_db = None
            
#             print(f"\nLooking for: {results_file}")
            
#             content = None
#             if os.path.exists(results_file):
#                 with open(results_file, 'r') as f:
#                     content = f.read()
            
#             if content:
#                 lines_list = content.split('\n')
#                 for i, line in enumerate(lines_list):
#                     if 'VOUT_FINAL:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             vout = float(match.group(1))
                    
#                     if line.startswith('GAIN_DB:') and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             gain_db = float(match.group(1))
                    
#                     if 'DC_GAIN_DB:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             dc_gain_db = float(match.group(1))
                    
#                     if 'SETTLING_TIME_NS:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             settling_time_ns = float(match.group(1))
                    
#                     if 'CHARGE_INJ_MV:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             charge_injection_mv = abs(float(match.group(1)))
                    
#                     if 'POWER_UW:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             power_uw = float(match.group(1))
                    
#                     if 'THD_DB:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             thd_db = float(match.group(1))
                    
#                     if 'OUTPUT_SWING:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             output_swing = float(match.group(1))
                    
#                     if 'UGBW_MHZ:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             ugbw_mhz = float(match.group(1))
                    
#                     if 'PHASE_MARGIN:' in line and i+1 < len(lines_list):
#                         match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines_list[i+1])
#                         if match:
#                             phase_margin = float(match.group(1))
            
#             gain_error_db = None
#             if gain_db is not None:
#                 gain_error_db = abs(gain_db - gain_target_db)
            
#             print(f"\n{'='*70}")
#             print(f"SWITCHED CAPACITOR CIRCUIT - READY FOR OPTIMIZATION")
#             print(f"{'='*70}")
#             print(f"  SC Gain:      {gain_db:.2f} dB (target: 0±0.2 dB)" if gain_db is not None else "  SC Gain:      NOT MEASURED")
#             print(f"  THD:          {thd_db:.2f} dB (target: ≤-60 dB)" if thd_db else "  THD:          NOT MEASURED")
#             print(f"  Power:        {power_uw:.0f} µW (target: ≤1000 µW)" if power_uw else "  Power:        NOT MEASURED")
#             print(f"  Phase Margin: {phase_margin:.1f}° (target: ≥60°)" if phase_margin else "  Phase Margin: NOT MEASURED")
#             print(f"  UGBW:         {ugbw_mhz:.1f} MHz (target: ≥10 MHz)" if ugbw_mhz else "  UGBW:         NOT MEASURED")
#             print(f"  Settling:     {settling_time_ns:.0f} ns" if settling_time_ns else "  Settling:     NOT MEASURED")
#             print(f"{'='*70}")
    
#             if vout is not None:
#                 return SwitchedCapResult(
#                     W_op1=W_op1, W_op2=W_op2, W_sw=W_sw,
#                     L_op=L_op, L_sw=L_sw,
#                     m_op=m_op, m_sw=m_sw,
#                     C_samp=C_samp, C_hold=C_hold, C_load=C_load,
#                     vout=vout, gain_db=gain_db, 
#                     gain_error_db=gain_error_db,
#                     settling_time_ns=settling_time_ns,
#                     charge_injection_mv=charge_injection_mv,
#                     power_uw=power_uw,
#                     thd_db=thd_db,
#                     output_swing_v=output_swing,
#                     phase_margin_deg=phase_margin,
#                     ugbw_mhz=ugbw_mhz,
#                     dc_gain_db=dc_gain_db
#                 )
    
#             return None
    
#         except Exception as e:
#             print(f"Error: {e}")
#             import traceback
#             traceback.print_exc()
#             return None

#         finally:
#             # ALWAYS return to original directory
#             os.chdir(original_dir)


        