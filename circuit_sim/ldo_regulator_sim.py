import os
import subprocess
import re
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LDOResult:
    W_pass: float
    L_pass: float
    m_pass: int
    W_diff: float
    W_load: float
    W_bias: float
    L_amp: float
    m_diff: int
    m_load: int
    L_r1: float
    L_r2: float
    vref: float
    vout: float
    iload_max_ma: float
    dropout_mv: Optional[float]
    line_reg: Optional[float]
    load_reg: Optional[float]
    psrr_db: Optional[float]
    power_uw: Optional[float]
    vout_min: Optional[float]
    vout_max: Optional[float]


def simulate_ldo_regulator(
    pdk_lib_path,
    W_pass=100,
    L_pass=0.5,
    m_pass=10,
    W_diff=10,
    W_load=20,
    W_bias=5,
    L_amp=1,
    m_diff=4,
    m_load=4,
    L_r1=156,
    L_r2=156,
    vref=0.6,
    vdd=1.8,
    vout_target=1.2,
    iload=10e-3,
    ibias=10e-6,
    temp_nom=27,
    temp_range=(-40, 125),
    results_dir="./results",
    **kwargs,
):
    """
    Simulate LDO voltage regulator - ALL values measured
    """

    # ------------------------------------------------------------------
    # Simulation directory
    # ------------------------------------------------------------------
    sim_dir = os.path.join(results_dir, "ngspice_sim")
    os.makedirs(sim_dir, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(sim_dir)

    try:
        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        R_bias = vdd / (ibias * 10)
        iload_min = iload * 0.1
        iload_max = iload * 2.0

        results_file = "ldo_results.txt"

        # ------------------------------------------------------------------
        # Netlist
        # ------------------------------------------------------------------
        netlist = f"""* LDO Voltage Regulator
.lib {pdk_lib_path} tt
.global VDD GND
.temp {temp_nom}

XMPASS VOUT VGATE VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_pass} w={W_pass} m={m_pass}

XM1 VD1 VREF VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}
XM2 VD2 VFB VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}

XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias} m=4

XM3 VD1 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}
XM4 VD2 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}

XM5 VGATE VD2 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load*4} m={m_load}
XM6 VGATE VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias*2} m=2

CC VGATE VOUT 5p

XR1 VOUT VFB VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r1}
XR2 VFB GND VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r2}

XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*4} w={W_bias} m=1
RBIAS VDD_INTERNAL VBIAS {R_bias}

ILOAD VOUT GND DC {iload}
CLOAD VOUT GND 100p

VREF VREF GND DC {vref}
VSUP_DC VDD GND DC {vdd}
VSUP_AC VDD_INTERNAL VDD DC 0 AC 1

.ic v(VOUT)={vout_target} v(VGATE)={vdd-0.7} v(VFB)={vref}

.control
set noaskquit
set wr_singlescale
set wr_vecnames

op

echo "VOUT_NOM:" > {results_file}
print v(VOUT) >> {results_file}

echo "I_SUPPLY:" >> {results_file}
print i(VSUP_DC) >> {results_file}

let i_load_val = {iload}
echo "I_LOAD:" >> {results_file}
print i_load_val >> {results_file}

let vout_nom = v(VOUT)
let dropout_mv = ({vdd} - vout_nom) * 1000
print dropout_mv

dc VSUP_DC {vdd*0.9} {vdd*1.1} 0.01
let vout_line = v(VOUT)
let line_reg = (vecmax(vout_line) - vecmin(vout_line)) * 1000 / {vdd*0.2}
print line_reg

dc ILOAD {iload_min} {iload_max} {(iload_max-iload_min)/20}
let vout_load = v(VOUT)
let load_reg = (vout_load[0] - vout_load[length(vout_load)-1]) * 1000 / (({iload_max}-{iload_min})*1000)
print load_reg

dc temp {temp_range[0]} {temp_range[1]} 10
let vout_temp = v(VOUT)
let vout_temp_min = vecmin(vout_temp)
let vout_temp_max = vecmax(vout_temp)

echo "TEMP_MIN:" >> {results_file}
print vout_temp_min >> {results_file}

echo "TEMP_MAX:" >> {results_file}
print vout_temp_max >> {results_file}

destroy all
op
ac dec 20 1 1Meg

set curplot=ac1
let vout_ac_mag = mag(v(VOUT)[10])
echo "AC_VOUT_MAG:" >> {results_file}
print vout_ac_mag >> {results_file}

quit
.endc
.end
"""

        with open("ldo_sim.spice", "w") as f:
            f.write(netlist)

        # ------------------------------------------------------------------
        # Run ngspice
        # ------------------------------------------------------------------
        result = subprocess.run(
            ["ngspice", "-b", "ldo_sim.spice"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(result.stderr)
            return None

        # ------------------------------------------------------------------
        # Parse stdout
        # ------------------------------------------------------------------
        vout = None
        line_reg = None
        load_reg = None
        dropout_mv = None

        for line in result.stdout.splitlines():
            if "dropout_mv" in line and "=" in line:
                dropout_mv = float(line.split("=")[-1])
            elif "line_reg" in line and "=" in line:
                line_reg = float(line.split("=")[-1])
            elif "load_reg" in line and "=" in line:
                load_reg = abs(float(line.split("=")[-1]))

        # ------------------------------------------------------------------
        # Parse results file
        # ------------------------------------------------------------------
        if not os.path.exists(results_file):
            return None

        with open(results_file, "r") as f:
            content = f.read()

        def grab(name):
            m = re.search(
                rf"{name}.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                content,
                re.IGNORECASE | re.S,
            )
            return float(m.group(1)) if m else None

        vout = grab("v\\(vout\\)")
        i_supply = abs(grab("i\\(vsup_dc\\)") or 0.0)
        i_load = abs(grab("i_load_val") or 0.0)
        vout_min = grab("vout_temp_min")
        vout_max = grab("vout_temp_max")
        vout_ac_mag = grab("vout_ac_mag")

        power_uw = i_supply * vdd * 1e6 if i_supply else None
        psrr_db = (
            20 * math.log10(1.0 / vout_ac_mag)
            if vout_ac_mag and vout_ac_mag > 1e-12
            else None
        )

        if vout is None:
            return None

        return LDOResult(
            W_pass=W_pass,
            L_pass=L_pass,
            m_pass=m_pass,
            W_diff=W_diff,
            W_load=W_load,
            W_bias=W_bias,
            L_amp=L_amp,
            m_diff=m_diff,
            m_load=m_load,
            L_r1=L_r1,
            L_r2=L_r2,
            vref=vref,
            vout=vout,
            iload_max_ma=iload_max * 1000,
            dropout_mv=dropout_mv,
            line_reg=line_reg,
            load_reg=load_reg,
            psrr_db=psrr_db,
            power_uw=power_uw,
            vout_min=vout_min,
            vout_max=vout_max,
        )

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # ALWAYS restore working directory
        os.chdir(original_dir)


# import os
# import subprocess
# import re
# import math
# from dataclasses import dataclass
# from typing import Optional


# @dataclass
# class LDOResult:
#     W_pass: float
#     L_pass: float
#     m_pass: int
#     W_diff: float
#     W_load: float
#     W_bias: float
#     L_amp: float
#     m_diff: int
#     m_load: int
#     L_r1: float
#     L_r2: float
#     vref: float
#     vout: float
#     iload_max_ma: float
#     dropout_mv: Optional[float]
#     line_reg: Optional[float]
#     load_reg: Optional[float]
#     psrr_db: Optional[float]
#     power_uw: Optional[float]
#     vout_min: Optional[float]
#     vout_max: Optional[float]


# def simulate_ldo_regulator(
#     pdk_lib_path,
#     W_pass=100,
#     L_pass=0.5,
#     m_pass=10,
#     W_diff=10,
#     W_load=20,
#     W_bias=5,
#     L_amp=1,
#     m_diff=4,
#     m_load=4,
#     L_r1=156,
#     L_r2=156,
#     vref=0.6,
#     vdd=1.8,
#     vout_target=1.2,
#     iload=10e-3,
#     ibias=10e-6,
#     temp_nom=27,
#     temp_range=(-40, 125),
#     results_dir="./results",
# ):
#     """
#     Simulate LDO voltage regulator - ALL values measured
#     """

#     # ------------------------------------------------------------------
#     # Simulation directory
#     # ------------------------------------------------------------------
#     sim_dir = os.path.join(results_dir, "ngspice_sim")
#     os.makedirs(sim_dir, exist_ok=True)

#     original_dir = os.getcwd()
#     os.chdir(sim_dir)

#     try:
#         # ------------------------------------------------------------------
#         # Parameters
#         # ------------------------------------------------------------------
#         R_bias = vdd / (ibias * 10)
#         iload_min = iload * 0.1
#         iload_max = iload * 2.0

#         results_file = "ldo_results.txt"

#         # ------------------------------------------------------------------
#         # Netlist
#         # ------------------------------------------------------------------
#         netlist = f"""* LDO Voltage Regulator
# .lib {pdk_lib_path} tt
# .global VDD GND
# .temp {temp_nom}

# XMPASS VOUT VGATE VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_pass} w={W_pass} m={m_pass}

# XM1 VD1 VREF VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}
# XM2 VD2 VFB VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}

# XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias} m=4

# XM3 VD1 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}
# XM4 VD2 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}

# XM5 VGATE VD2 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load*4} m={m_load}
# XM6 VGATE VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias*2} m=2

# CC VGATE VOUT 5p

# XR1 VOUT VFB VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r1}
# XR2 VFB GND VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r2}

# XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*4} w={W_bias} m=1
# RBIAS VDD_INTERNAL VBIAS {R_bias}

# ILOAD VOUT GND DC {iload}
# CLOAD VOUT GND 100p

# VREF VREF GND DC {vref}
# VSUP_DC VDD GND DC {vdd}
# VSUP_AC VDD_INTERNAL VDD DC 0 AC 1

# .ic v(VOUT)={vout_target} v(VGATE)={vdd-0.7} v(VFB)={vref}

# .control
# set noaskquit
# set wr_singlescale
# set wr_vecnames

# op

# echo "VOUT_NOM:" > {results_file}
# print v(VOUT) >> {results_file}

# echo "I_SUPPLY:" >> {results_file}
# print i(VSUP_DC) >> {results_file}

# let i_load_val = {iload}
# echo "I_LOAD:" >> {results_file}
# print i_load_val >> {results_file}

# let vout_nom = v(VOUT)
# let dropout_mv = ({vdd} - vout_nom) * 1000
# print dropout_mv

# dc VSUP_DC {vdd*0.9} {vdd*1.1} 0.01
# let vout_line = v(VOUT)
# let line_reg = (vecmax(vout_line) - vecmin(vout_line)) * 1000 / {vdd*0.2}
# print line_reg

# dc ILOAD {iload_min} {iload_max} {(iload_max-iload_min)/20}
# let vout_load = v(VOUT)
# let load_reg = (vout_load[0] - vout_load[length(vout_load)-1]) * 1000 / (({iload_max}-{iload_min})*1000)
# print load_reg

# dc temp {temp_range[0]} {temp_range[1]} 10
# let vout_temp = v(VOUT)
# let vout_temp_min = vecmin(vout_temp)
# let vout_temp_max = vecmax(vout_temp)

# echo "TEMP_MIN:" >> {results_file}
# print vout_temp_min >> {results_file}

# echo "TEMP_MAX:" >> {results_file}
# print vout_temp_max >> {results_file}

# destroy all
# op
# ac dec 20 1 1Meg

# set curplot=ac1
# let vout_ac_mag = mag(v(VOUT)[10])
# echo "AC_VOUT_MAG:" >> {results_file}
# print vout_ac_mag >> {results_file}

# quit
# .endc
# .end
# """

#         with open("ldo_sim.spice", "w") as f:
#             f.write(netlist)

#         # ------------------------------------------------------------------
#         # Run ngspice
#         # ------------------------------------------------------------------
#         result = subprocess.run(
#             ["ngspice", "-b", "ldo_sim.spice"],
#             capture_output=True,
#             text=True,
#             timeout=120,
#         )

#         if result.returncode != 0:
#             print(result.stderr)
#             return None

#         # ------------------------------------------------------------------
#         # Parse stdout
#         # ------------------------------------------------------------------
#         vout = None
#         line_reg = None
#         load_reg = None
#         dropout_mv = None

#         for line in result.stdout.splitlines():
#             if "dropout_mv" in line and "=" in line:
#                 dropout_mv = float(line.split("=")[-1])
#             elif "line_reg" in line and "=" in line:
#                 line_reg = float(line.split("=")[-1])
#             elif "load_reg" in line and "=" in line:
#                 load_reg = abs(float(line.split("=")[-1]))

#         # ------------------------------------------------------------------
#         # Parse results file
#         # ------------------------------------------------------------------
#         if not os.path.exists(results_file):
#             return None

#         with open(results_file, "r") as f:
#             content = f.read()

#         def grab(name):
#             m = re.search(
#                 rf"{name}.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
#                 content,
#                 re.IGNORECASE | re.S,
#             )
#             return float(m.group(1)) if m else None

#         vout = grab("v\\(vout\\)")
#         i_supply = abs(grab("i\\(vsup_dc\\)") or 0.0)
#         i_load = abs(grab("i_load_val") or 0.0)
#         vout_min = grab("vout_temp_min")
#         vout_max = grab("vout_temp_max")
#         vout_ac_mag = grab("vout_ac_mag")

#         power_uw = i_supply * vdd * 1e6 if i_supply else None
#         psrr_db = (
#             20 * math.log10(1.0 / vout_ac_mag)
#             if vout_ac_mag and vout_ac_mag > 1e-12
#             else None
#         )

#         if vout is None:
#             return None

#         return LDOResult(
#             W_pass=W_pass,
#             L_pass=L_pass,
#             m_pass=m_pass,
#             W_diff=W_diff,
#             W_load=W_load,
#             W_bias=W_bias,
#             L_amp=L_amp,
#             m_diff=m_diff,
#             m_load=m_load,
#             L_r1=L_r1,
#             L_r2=L_r2,
#             vref=vref,
#             vout=vout,
#             iload_max_ma=iload_max * 1000,
#             dropout_mv=dropout_mv,
#             line_reg=line_reg,
#             load_reg=load_reg,
#             psrr_db=psrr_db,
#             power_uw=power_uw,
#             vout_min=vout_min,
#             vout_max=vout_max,
#         )

#     except Exception as e:
#         print(f"❌ Simulation failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

#     finally:
#         # ALWAYS restore working directory
#         os.chdir(original_dir)


# # import os
# # import subprocess
# # import re
# # import math
# # from dataclasses import dataclass
# # from typing import Optional

# # @dataclass
# # class LDOResult:
# #     W_pass: float
# #     L_pass: float
# #     m_pass: int
# #     W_diff: float
# #     W_load: float
# #     W_bias: float
# #     L_amp: float
# #     m_diff: int
# #     m_load: int
# #     L_r1: float
# #     L_r2: float
# #     vref: float
# #     vout: float
# #     iload_max_ma: float
# #     dropout_mv: Optional[float]
# #     line_reg: Optional[float]
# #     load_reg: Optional[float]
# #     psrr_db: Optional[float]
# #     power_uw: Optional[float]
# #     vout_min: Optional[float]
# #     vout_max: Optional[float]

# # def simulate_ldo_regulator(pdk_lib_path,
# #                            W_pass=100, L_pass=0.5, m_pass=10,
# #                            W_diff=10, W_load=20, W_bias=5,
# #                            L_amp=1, m_diff=4, m_load=4,
# #                            L_r1=156, L_r2=156,
# #                            vref=0.6, vdd=1.8, vout_target=1.2,
# #                            iload=10e-3, ibias=10e-6,
# #                            temp_nom=27, temp_range=(-40, 125),
# #                            results_dir='./results'):
# #     """
# #     Simulate LDO voltage regulator - ALL values measured
# #     """

# #     # CREATE SIMULATION DIRECTORY using results_dir from YAML
# #     sim_dir = os.path.join(results_dir, 'ngspice_sim')
# #     os.makedirs(sim_dir, exist_ok=True)
    
# #     # Change to simulation directory
# #     original_dir = os.getcwd()
# #     os.chdir(sim_dir)

# #     try:
# #         pdk_dir = os.path.dirname(pdk_lib_path)
        
# #         R_bias = vdd / (ibias * 10)
# #         iload_min = iload * 0.1
# #         iload_max = iload * 2.0
        
# #         results_file = 'ldo_results.txt'
        
# #         netlist = f"""* LDO Voltage Regulator
# #     .lib {pdk_lib_path} tt
# #     .global VDD GND
# #     .temp {temp_nom}
    
# #     * Pass Transistor (Main Power Device)
# #     XMPASS VOUT VGATE VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_pass} w={W_pass} m={m_pass}
    
# #     * Error Amplifier - Differential Input Pair
# #     XM1 VD1 VREF VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}
# #     XM2 VD2 VFB VTAIL GND sky130_fd_pr__nfet_01v8 l={L_amp} w={W_diff} m={m_diff}
    
# #     * Error Amplifier - Tail Current Source
# #     XMTAIL VTAIL VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias} m=4
    
# #     * Error Amplifier - Active Load (Current Mirror)
# #     XM3 VD1 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}
# #     XM4 VD2 VD1 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load} m={m_load}
    
# #     * Output Stage Driver
# #     XM5 VGATE VD2 VDD_INTERNAL VDD sky130_fd_pr__pfet_01v8 l={L_amp} w={W_load*4} m={m_load}
# #     XM6 VGATE VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*2} w={W_bias*2} m=2
    
# #     * Compensation Capacitor
# #     CC VGATE VOUT 5p
    
# #     * Feedback Resistor Divider
# #     XR1 VOUT VFB VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r1}
# #     XR2 VFB GND VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_r2}
    
# #     * Bias Circuit
# #     XMBIAS VBIAS VBIAS GND GND sky130_fd_pr__nfet_01v8 l={L_amp*4} w={W_bias} m=1
# #     RBIAS VDD_INTERNAL VBIAS {R_bias}
    
# #     * Load and Output Capacitor
# #     ILOAD VOUT GND DC {iload}
# #     CLOAD VOUT GND 100p
    
# #     * Supply and Reference
# #     VREF VREF GND DC {vref}
# #     VSUP_DC VDD GND DC {vdd}
# #     VSUP_AC VDD_INTERNAL VDD DC 0 AC 1
    
# #     .ic v(VOUT)={vout_target} v(VGATE)={vdd-0.7} v(VFB)={vref}
    
# #     .op
# #     .dc VSUP_DC {vdd*0.9} {vdd*1.1} 0.01
# #     .dc ILOAD {iload_min} {iload_max} {(iload_max-iload_min)/20}
# #     .dc temp {temp_range[0]} {temp_range[1]} 10
# #     .ac dec 20 1 1Meg
    
# #     .control
# #     set noaskquit
# #     set wr_singlescale
# #     set wr_vecnames
    
# #     op
    
# #     echo "VOUT_NOM:" > {results_file}
# #     print v(VOUT) >> {results_file}
    
# #     echo "I_SUPPLY:" >> {results_file}
# #     print i(VSUP_DC) >> {results_file}
    
# #     let i_load_val = {iload}
# #     echo "I_LOAD:" >> {results_file}
# #     print i_load_val >> {results_file}
    
# #     let vout_nom = v(VOUT)
# #     let dropout_mv = ({vdd} - vout_nom) * 1000
# #     print dropout_mv
    
# #     dc VSUP_DC {vdd*0.9} {vdd*1.1} 0.01
# #     let vout_line = v(VOUT)
# #     let line_reg = (vecmax(vout_line) - vecmin(vout_line)) * 1000 / {vdd*0.2}
# #     print line_reg
    
# #     dc ILOAD {iload_min} {iload_max} {(iload_max-iload_min)/20}
# #     let vout_load = v(VOUT)
# #     let load_reg = (vout_load[0] - vout_load[length(vout_load)-1]) * 1000 / (({iload_max} - {iload_min}) * 1000)
# #     print load_reg
    
# #     dc temp {temp_range[0]} {temp_range[1]} 10
# #     let vout_temp = v(VOUT)
# #     let vout_temp_min = vecmin(vout_temp)
# #     let vout_temp_max = vecmax(vout_temp)
    
# #     echo "TEMP_MIN:" >> {results_file}
# #     print vout_temp_min >> {results_file}
    
# #     echo "TEMP_MAX:" >> {results_file}
# #     print vout_temp_max >> {results_file}
    
# #     destroy all
# #     op
# #     ac dec 20 1 1Meg
    
# #     set curplot = ac1
# #     let vout_ac_mag = mag(v(VOUT)[10])
# #     echo "AC_VOUT_MAG:" >> {results_file}
# #     print vout_ac_mag >> {results_file}
    
# #     quit
# #     .endc
# #     .end
# #     """
    
# #         netlist_path = 'ldo_sim.spice'
# #         with open(netlist_path, 'w') as f:
# #             f.write(netlist)
    
# #         try:
# #             result = subprocess.run(
# #                 ['ngspice', '-b', netlist_path],
# #                 capture_output=True,
# #                 text=True,
# #                 timeout=120
# #             )
            
# #             if result.returncode != 0:
# #                 print("NGSPICE ERRORS:")
# #                 print(result.stderr)
# #                 return None
    
# #             lines = result.stdout.split('\n')
            
# #             vout = None
# #             line_reg = None
# #             load_reg = None
# #             dropout_mv = None
# #             power_uw = None
# #             vout_min = None
# #             vout_max = None
# #             psrr_db = None
            
# #             i_supply = None
# #             i_load = None
# #             vout_ac_mag = None
    
# #             # Parse from stdout
# #             for line in lines:
# #                 line = line.strip()
                
# #                 if 'dropout_mv' in line and '=' in line:
# #                     match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
# #                     if match:
# #                         dropout_mv = float(match.group(1))
                        
# #                 elif 'line_reg' in line and '=' in line:
# #                     match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
# #                     if match:
# #                         line_reg = float(match.group(1))
                        
# #                 elif 'load_reg' in line and '=' in line:
# #                     match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
# #                     if match:
# #                         load_reg = abs(float(match.group(1)))
            
# #             # Parse from file - CHECK WHERE IT IS
# #             print(f"\nCurrent directory: {os.getcwd()}")
# #             print(f"Looking for: {results_file}")
# #             print(f"File exists: {os.path.exists(results_file)}")
            
# #             # Try both current dir and pdk_dir
# #             file_paths = [
# #                 results_file,
# #                 os.path.join(pdk_dir, results_file),
# #                 os.path.join(os.getcwd(), results_file)
# #             ]
            
# #             content = None
# #             if os.path.exists(results_file):
# #                 print(f"✓ Found file at: {results_file}")
# #                 with open(results_file, 'r') as f:
# #                     content = f.read()
    
            
# #             if content:
# #                 print(f"\n=== FILE CONTENT (first 800 chars) ===")
# #                 print(content[:800])
# #                 print(f"=======================================\n")
                
# #                 match = re.search(r'v\(vout\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     vout = float(match.group(1))
# #                 else:
# #                     print("❌ Could not parse v(vout)")
                
# #                 match = re.search(r'i\(vsup_dc\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     i_supply = abs(float(match.group(1)))
# #                 else:
# #                     print("❌ Could not parse i(vsup_dc)")
                
# #                 match = re.search(r'i_load_val\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     i_load = abs(float(match.group(1)))
# #                 else:
# #                     print("❌ Could not parse i_load_val")
                
# #                 match = re.search(r'vout_temp_min\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     vout_min = float(match.group(1))
# #                 else:
# #                     print("❌ Could not parse vout_temp_min")
                
# #                 match = re.search(r'vout_temp_max\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     vout_max = float(match.group(1))
# #                 else:
# #                     print("❌ Could not parse vout_temp_max")
                
# #                 match = re.search(r'vout_ac_mag\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
# #                 if match:
# #                     vout_ac_mag = float(match.group(1))
# #                 else:
# #                     print("❌ Could not parse vout_ac_mag")
# #             else:
# #                 print(f"❌ File not found in any of these locations:")
# #                 for fpath in file_paths:
# #                     print(f"   - {fpath}")
            
# #             # Calculate derived values
# #             if i_supply and i_load:
# #                 power_uw = (i_supply - i_load) * vdd * 1e6
# #             elif i_supply:
# #                 power_uw = i_supply * vdd * 1e6
            
# #             if vout_ac_mag and vout_ac_mag > 1e-12:
# #                 psrr_db = 20 * math.log10(1.0 / vout_ac_mag)
            
# #             print(f"\n{'='*70}")
# #             print(f"MEASURED VALUES:")
# #             print(f"{'='*70}")
# #             print(f"  Vout:     {vout:.6f} V" if vout else "  Vout:     NOT MEASURED")
# #             print(f"  I_supply: {i_supply:.6e} A" if i_supply else "  I_supply: NOT MEASURED")
# #             print(f"  I_load:   {i_load:.6e} A" if i_load else "  I_load:   NOT MEASURED")
# #             print(f"  Power:    {power_uw:.2f} µW" if power_uw else "  Power:    NOT MEASURED")
# #             print(f"  Dropout:  {dropout_mv:.2f} mV" if dropout_mv else "  Dropout:  NOT MEASURED")
# #             print(f"  Line Reg: {line_reg:.4f} mV/V" if line_reg else "  Line Reg: NOT MEASURED")
# #             print(f"  Load Reg: {load_reg:.6f} mV/mA" if load_reg else "  Load Reg: NOT MEASURED")
# #             print(f"  AC mag:   {vout_ac_mag:.6e}" if vout_ac_mag else "  AC mag:   NOT MEASURED")
# #             print(f"  PSRR:     {psrr_db:.2f} dB" if psrr_db else "  PSRR:     NOT MEASURED")
# #             print(f"  Temp Min: {vout_min:.6f} V" if vout_min else "  Temp Min: NOT MEASURED")
# #             print(f"  Temp Max: {vout_max:.6f} V" if vout_max else "  Temp Max: NOT MEASURED")
# #             print(f"{'='*70}")
    
# #             if vout is not None:
# #                 return LDOResult(
# #                     W_pass=W_pass, L_pass=L_pass, m_pass=m_pass,
# #                     W_diff=W_diff, W_load=W_load, W_bias=W_bias,
# #                     L_amp=L_amp, m_diff=m_diff, m_load=m_load,
# #                     L_r1=L_r1, L_r2=L_r2,
# #                     vref=vref, vout=vout, iload_max_ma=iload_max*1000,
# #                     dropout_mv=dropout_mv, line_reg=line_reg, load_reg=load_reg,
# #                     psrr_db=psrr_db, power_uw=power_uw,
# #                     vout_min=vout_min, vout_max=vout_max
# #                 )
    
# #             return None
    
# #         except Exception as e:
# #             print(f"Error: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return None

# #         finally:
# #             # ALWAYS return to original directory
# #             os.chdir(original_dir)

        