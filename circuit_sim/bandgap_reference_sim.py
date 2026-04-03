import os
import subprocess
import re
import numpy as np
from dataclasses import dataclass


@dataclass
class BandgapResult:
    # MOSFET widths
    W_mp: float
    W_mn: float
    W_mp_startup: float
    W_mn_startup: float

    # MOSFET lengths
    L_mp: float
    L_mn: float
    L_startup: float

    # MOSFET multipliers
    m_mp: int
    m_mn: int
    m_mp_startup: int
    m_mn_startup: int

    # BJT multipliers
    m_q1: int
    m_q2: int
    m_q3: int

    # Resistor lengths
    L_ra: float
    L_rb: float

    # Performance metrics
    vref: float
    vref_min: float
    vref_max: float
    tc: float
    power_uw: float
    i_q1_ua: float
    i_q2_ua: float
    i_out_ua: float

    # Additional metrics
    line_regulation: float = None
    psrr_100hz: float = None
    vref_at_1p8v: float = None
    vref_at_3p3v: float = None


def simulate_bandgap_reference(pdk_lib_path, pnp_model_path,
                                 W_mp=5, W_mn=5, W_mp_startup=5, W_mn_startup=1,
                                 L_mp=2, L_mn=1, L_startup=7,
                                 m_mp=4, m_mn=8, m_mp_startup=1, m_mn_startup=1,
                                 m_q1=1, m_q2=8, m_q3=1,
                                 L_ra=31.2, L_rb=132.6,
                                 vdd=2.0, temp=27,
                                 measure_line_reg=True,
                                 measure_psrr=True,
                                 vdd_range=(1.8, 3.3),
                                 results_dir="./results",
                                 **kwargs):
    """
    Simulate bandgap voltage reference - ALL ANALYSES IN ONE SIMULATION RUN
    """

    sim_dir = os.path.join(results_dir, "ngspice_sim")
    os.makedirs(sim_dir, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(sim_dir)

    try:

        vdd_min, vdd_max = vdd_range
        pdk_dir = os.path.dirname(pdk_lib_path)

        
        op_file = 'bandgap_op.txt'
        dc_file = 'bandgap_dc.txt'
        ac_file = 'bandgap_ac.txt'
        
        netlist = f"""* Bandgap Reference Circuit - Complete Characterization
    .lib {pdk_lib_path} tt
    .include {pnp_model_path}
    .global VDD GND
    .temp {temp}
    
    *** Current Mirror (PMOS) - This is the main current source
    XMP1 NET1 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m={m_mp}
    XMP2 NET2 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m={m_mp}
    XMP3 NET3 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m=1
    
    *** NMOS devices (NOT self-biased - just diode-connected for low impedance)
    XMN1 NET1 NET1 QP1 GND sky130_fd_pr__nfet_01v8_lvt l={L_mn} w={W_mn} m={m_mn}
    XMN2 NET2 NET2 RA_TOP GND sky130_fd_pr__nfet_01v8_lvt l={L_mn} w={W_mn} m={m_mn}
    
    *** Startup Circuit
    XMP4 NET4 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m={m_mp_startup}
    XMP5 NET5 NET2 NET4 VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m={m_mp_startup}
    XMP6 NET7 NET6 NET2 VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m=2
    XMN3 NET6 NET6 NET8 GND sky130_fd_pr__nfet_01v8_lvt l={L_startup} w={W_mn_startup} m={m_mn_startup}
    XMN4 NET8 NET8 GND GND sky130_fd_pr__nfet_01v8_lvt l={L_startup} w={W_mn_startup} m={m_mn_startup}
    
    *** BJTs
    XQP1 GND GND QP1 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q1}
    XQP2 GND GND QP2 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q2}
    XQP3 GND GND QP3 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q3}
    
    *** Resistors
    XRA RA_TOP QP2 VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_ra}
    XRB NET3 QP3 VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_rb}
    
    *** Measurement Points
    VID1 QP1 GND DC 0
    VID2 QP2 GND DC 0
    VID3 QP3 VREF DC 0
    VID4 NET7 NET1 DC 0
    VID5 NET5 NET6 DC 0
    
    VSUP VDD GND DC {vdd} AC 1
    
    .control
    op
    
    *echo "=== DEBUG NODE VOLTAGES ==="
    *print v(QP1) v(QP2) v(QP3) v(NET3) v(VREF)
    *print v(RA_TOP) v(NET1) v(NET2)              
    *print vid1#branch vid2#branch vid3#branch vsup#branch
    *echo "==========================="
    
    set wr_singlescale
    set wr_vecnames
    option numdgt=7
    wrdata {op_file} v(VREF) vid1#branch vid2#branch vid3#branch vsup#branch
    
    dc VSUP {vdd_min} {vdd_max} 0.1
    wrdata {dc_file} v(VREF)
    
    ac dec 10 1 1Meg
    let psrr_db = db(v(VDD)/v(VREF))
    wrdata {ac_file} frequency psrr_db
    
    quit
    .endc
    
    .end
    """
    
        netlist_path = 'bandgap_sim.spice'
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
                print("NGSPICE ERRORS:")
                print(result.stderr)
                return None
    
            # Parse Operating Point
            vref = None
            i_supply = None
            i_q1 = None
            i_q2 = None
            i_out = None
            
            if os.path.exists(op_file):
                with open(op_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line or 'v(VREF)' in line:  # Skip header
                        continue
                        
                    parts = line.split()
                    
                    # File now has 6 columns: index, v(VREF), vid1#branch, vid2#branch, vid3#branch, vsup#branch
                    if len(parts) >= 6:
                        try:
                            # Column mapping:
                            # parts[0] = index (net1)
                            # parts[1] = v(VREF)
                            # parts[2] = vid1#branch (I_Q1)
                            # parts[3] = vid2#branch (I_Q2)
                            # parts[4] = vid3#branch (I_OUT)
                            # parts[5] = vsup#branch (I_SUPPLY)
                            
                            vref = float(parts[1])
                            i_q1 = abs(float(parts[2]))
                            i_q2 = abs(float(parts[3]))
                            i_out = abs(float(parts[4]))
                            i_supply = abs(float(parts[5]))
                            break
                        except ValueError:
                            continue
            
            if vref is None or i_supply is None:
                print("❌ Failed to parse operating point")
                return None
            
            power_uw = i_supply * vdd * 1e6
            i_q1_ua = i_q1 * 1e6 if i_q1 else 0
            i_q2_ua = i_q2 * 1e6 if i_q2 else 0
            i_out_ua = i_out * 1e6 if i_out else 0
            
            # Parse Line Regulation
            line_regulation = None
            vref_at_min_vdd = None
            vref_at_max_vdd = None
            
            if measure_line_reg and os.path.exists(dc_file):
                with open(dc_file, 'r') as f:
                    lines = f.readlines()
                
                vref_values = []
                for line in lines:
                    line = line.strip()
                    if not line or 'v(VREF)' in line:
                        continue
                    parts = line.split()
                    # DC file also has index column
                    if len(parts) >= 2:
                        try:
                            vref_values.append(float(parts[1]))  # Column 1 is v(VREF)
                        except ValueError:
                            continue
                
                if len(vref_values) >= 2:
                    vref_at_min_vdd = vref_values[0]
                    vref_at_max_vdd = vref_values[-1]
                    delta_vref = abs(vref_at_max_vdd - vref_at_min_vdd)
                    delta_vdd = vdd_max - vdd_min
                    line_regulation = (delta_vref / vref) / delta_vdd * 100
            
            # Parse PSRR
            psrr_100hz = None
            
            if measure_psrr and os.path.exists(ac_file):
                with open(ac_file, 'r') as f:
                    lines = f.readlines()
                
                freq_values = []
                psrr_values = []
                
                for line in lines:
                    line = line.strip()
                    if not line or 'frequency' in line or 'psrr_db' in line:  # Skip headers
                        continue
                    parts = line.split()
                    # AC file has 4 columns: freq, freq, 0.0, psrr_db
                    if len(parts) >= 4:
                        try:
                            freq_values.append(float(parts[0]))  # Column 0 is frequency
                            psrr_values.append(float(parts[3]))  # Column 3 is psrr_db
                        except ValueError:
                            continue
                
                if len(psrr_values) > 0:
                    import numpy as np
                    idx = np.argmin(np.abs(np.array(freq_values) - 100))
                    psrr_100hz = psrr_values[idx]
                    print(f"✓ Found PSRR @ 100Hz: {psrr_100hz:.3f} dB")
            
            return BandgapResult(
                W_mp=W_mp, W_mn=W_mn,
                W_mp_startup=W_mp_startup, W_mn_startup=W_mn_startup,
                L_mp=L_mp, L_mn=L_mn, L_startup=L_startup,
                m_mp=m_mp, m_mn=m_mn,
                m_mp_startup=m_mp_startup, m_mn_startup=m_mn_startup,
                m_q1=m_q1, m_q2=m_q2, m_q3=m_q3,
                L_ra=L_ra, L_rb=L_rb,
                vref=vref, vref_min=vref, vref_max=vref,
                tc=0.0,
                power_uw=power_uw,
                i_q1_ua=i_q1_ua, i_q2_ua=i_q2_ua,
                i_out_ua=i_out_ua,
                line_regulation=line_regulation,
                psrr_100hz=psrr_100hz,
                vref_at_1p8v=vref_at_min_vdd,
                vref_at_3p3v=vref_at_max_vdd
            )
    
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        os.chdir(original_dir)
        
    

# import os
# import subprocess
# import re
# import numpy as np
# from dataclasses import dataclass

# @dataclass
# class BandgapResult:
#     # MOSFET widths
#     W_mp: float
#     W_mn: float
#     W_mp_startup: float
#     W_mn_startup: float
    
#     # MOSFET lengths
#     L_mp: float
#     L_mn: float
#     L_startup: float
    
#     # MOSFET multipliers
#     m_mp: int
#     m_mn: int
#     m_mp_startup: int
#     m_mn_startup: int
    
#     # BJT multipliers
#     m_q1: int
#     m_q2: int
#     m_q3: int
    
#     # Resistor lengths
#     L_ra: float
#     L_rb: float
    
#     # Performance metrics
#     vref: float           # Reference voltage at nominal temp (V)
#     vref_min: float       # Min Vref over temperature (V)
#     vref_max: float       # Max Vref over temperature (V)
#     tc: float             # Temperature coefficient (ppm/°C)
#     power_uw: float       # Power consumption (µW)
#     i_q1_ua: float        # Current through Q1 branch (µA)
#     i_q2_ua: float        # Current through Q2 branch (µA)
#     i_out_ua: float       # Output current (µA)
    
#     # Additional metrics
#     line_regulation: float = None      # Line regulation (%/V)
#     psrr_100hz: float = None          # PSRR at 100Hz (dB)
#     vref_at_1p8v: float = None        # Vref at VDD=1.8V
#     vref_at_3p3v: float = None        # Vref at VDD=3.3V



# def simulate_bandgap_reference(pdk_lib_path, pnp_model_path,
#                                  W_mp=5, W_mn=5, W_mp_startup=5, W_mn_startup=1,
#                                  L_mp=2, L_mn=1, L_startup=7,
#                                  m_mp=4, m_mn=8, m_mp_startup=1, m_mn_startup=1,
#                                  m_q1=1, m_q2=8, m_q3=1,
#                                  L_ra=31.2, L_rb=132.6,
#                                  vdd=2.0, temp=27,
#                                  measure_line_reg=True,
#                                  measure_psrr=True,
#                                  vdd_range=(1.8, 3.3),
#                                  results_dir='./results'):
#     """
#     Simulate bandgap voltage reference - ALL ANALYSES IN ONE SIMULATION RUN
#     """


#     sim_dir = os.path.join(results_dir, 'ngspice_sim')
#     os.makedirs(sim_dir, exist_ok=True)
    
#     # Change to simulation directory
#     original_dir = os.getcwd()
#     os.chdir(sim_dir)

#     try:
#         pdk_dir = os.path.dirname(pdk_lib_path)
#         vdd_min, vdd_max = vdd_range
        
#         op_file = 'bandgap_op.txt'
#         dc_file = 'bandgap_dc.txt'
#         ac_file = 'bandgap_ac.txt'
        
#         netlist = f"""* Bandgap Reference Circuit - Complete Characterization
#     .lib {pdk_lib_path} tt
#     .include {pnp_model_path}
#     .global VDD GND
#     .temp {temp}
    
#     *** Current Mirror (PMOS) - This is the main current source
#     XMP1 NET1 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m={m_mp}
#     XMP2 NET2 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m={m_mp}
#     XMP3 NET3 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp} m=1
    
#     *** NMOS devices (NOT self-biased - just diode-connected for low impedance)
#     XMN1 NET1 NET1 QP1 GND sky130_fd_pr__nfet_01v8_lvt l={L_mn} w={W_mn} m={m_mn}
#     XMN2 NET2 NET2 RA_TOP GND sky130_fd_pr__nfet_01v8_lvt l={L_mn} w={W_mn} m={m_mn}
    
#     *** Startup Circuit
#     XMP4 NET4 NET2 VDD VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m={m_mp_startup}
#     XMP5 NET5 NET2 NET4 VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m={m_mp_startup}
#     XMP6 NET7 NET6 NET2 VDD sky130_fd_pr__pfet_01v8_lvt l={L_mp} w={W_mp_startup} m=2
#     XMN3 NET6 NET6 NET8 GND sky130_fd_pr__nfet_01v8_lvt l={L_startup} w={W_mn_startup} m={m_mn_startup}
#     XMN4 NET8 NET8 GND GND sky130_fd_pr__nfet_01v8_lvt l={L_startup} w={W_mn_startup} m={m_mn_startup}
    
#     *** BJTs
#     XQP1 GND GND QP1 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q1}
#     XQP2 GND GND QP2 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q2}
#     XQP3 GND GND QP3 VDD sky130_fd_pr__pnp_05v5_W3p40L3p40 m={m_q3}
    
#     *** Resistors
#     XRA RA_TOP QP2 VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_ra}
#     XRB NET3 QP3 VDD sky130_fd_pr__res_high_po_1p41 w=1.41 l={L_rb}
    
#     *** Measurement Points
#     VID1 QP1 GND DC 0
#     VID2 QP2 GND DC 0
#     VID3 QP3 VREF DC 0
#     VID4 NET7 NET1 DC 0
#     VID5 NET5 NET6 DC 0
    
#     VSUP VDD GND DC {vdd} AC 1
    
#     .control
#     op
    
#     *echo "=== DEBUG NODE VOLTAGES ==="
#     *print v(QP1) v(QP2) v(QP3) v(NET3) v(VREF)
#     *print v(RA_TOP) v(NET1) v(NET2)              
#     *print vid1#branch vid2#branch vid3#branch vsup#branch
#     *echo "==========================="
    
#     set wr_singlescale
#     set wr_vecnames
#     option numdgt=7
#     wrdata {op_file} v(VREF) vid1#branch vid2#branch vid3#branch vsup#branch
    
#     dc VSUP {vdd_min} {vdd_max} 0.1
#     wrdata {dc_file} v(VREF)
    
#     ac dec 10 1 1Meg
#     let psrr_db = db(v(VDD)/v(VREF))
#     wrdata {ac_file} frequency psrr_db
    
#     quit
#     .endc
    
#     .end
#     """
    
#         netlist_path = 'bandgap_sim.spice'
#         with open(netlist_path, 'w') as f:
#             f.write(netlist)
    
#         try:
#             result = subprocess.run(
#                 ['ngspice', '-b', netlist_path],
#                 capture_output=True,
#                 text=True,
#                 timeout=120
#             )
            
#             if result.returncode != 0:
#                 print("NGSPICE ERRORS:")
#                 print(result.stderr)
#                 return None
    
#             # Parse Operating Point
#             vref = None
#             i_supply = None
#             i_q1 = None
#             i_q2 = None
#             i_out = None
            
#             if os.path.exists(op_file):
#                 with open(op_file, 'r') as f:
#                     lines = f.readlines()
                    
#                 for line in lines:
#                     line = line.strip()
#                     if not line or 'v(VREF)' in line:  # Skip header
#                         continue
                        
#                     parts = line.split()
                    
#                     # File now has 6 columns: index, v(VREF), vid1#branch, vid2#branch, vid3#branch, vsup#branch
#                     if len(parts) >= 6:
#                         try:
#                             vref = float(parts[1])
#                             i_q1 = abs(float(parts[2]))
#                             i_q2 = abs(float(parts[3]))
#                             i_out = abs(float(parts[4]))
#                             i_supply = abs(float(parts[5]))
                            
#                             print(f"✓ vref = {vref:.6f}V")
#                             print(f"✓ i_q1 = {i_q1:.6e}A")
#                             print(f"✓ i_q2 = {i_q2:.6e}A")
#                             print(f"✓ i_out = {i_out:.6e}A")
#                             print(f"✓ i_supply = {i_supply:.6e}A")
#                             break
#                         except ValueError:
#                             continue
            
#             if vref is None or i_supply is None:
#                 print("❌ Failed to parse operating point")
#                 return None
            
#             power_uw = i_supply * vdd * 1e6
#             i_q1_ua = i_q1 * 1e6 if i_q1 else 0
#             i_q2_ua = i_q2 * 1e6 if i_q2 else 0
#             i_out_ua = i_out * 1e6 if i_out else 0
            
#             # Parse Line Regulation
#             line_regulation = None
#             vref_at_min_vdd = None
#             vref_at_max_vdd = None
            
#             if measure_line_reg and os.path.exists(dc_file):
#                 with open(dc_file, 'r') as f:
#                     lines = f.readlines()
                
#                 vref_values = []
#                 for line in lines:
#                     line = line.strip()
#                     if not line or 'v(VREF)' in line:
#                         continue
#                     parts = line.split()
#                     # DC file also has index column
#                     if len(parts) >= 2:
#                         try:
#                             vref_values.append(float(parts[1]))  # Column 1 is v(VREF)
#                         except ValueError:
#                             continue
                
#                 if len(vref_values) >= 2:
#                     vref_at_min_vdd = vref_values[0]
#                     vref_at_max_vdd = vref_values[-1]
#                     delta_vref = abs(vref_at_max_vdd - vref_at_min_vdd)
#                     delta_vdd = vdd_max - vdd_min
#                     line_regulation = (delta_vref / vref) / delta_vdd * 100
            
#             # Parse PSRR
#             psrr_100hz = None
            
#             if measure_psrr and os.path.exists(ac_file):
#                 with open(ac_file, 'r') as f:
#                     lines = f.readlines()
                
#                 freq_values = []
#                 psrr_values = []
                
#                 for line in lines:
#                     line = line.strip()
#                     if not line or 'frequency' in line or 'psrr_db' in line:  # Skip headers
#                         continue
#                     parts = line.split()
#                     # AC file has 4 columns: freq, freq, 0.0, psrr_db
#                     if len(parts) >= 4:
#                         try:
#                             freq_values.append(float(parts[0]))  # Column 0 is frequency
#                             psrr_values.append(float(parts[3]))  # Column 3 is psrr_db
#                         except ValueError:
#                             continue
                
#                 if len(psrr_values) > 0:
#                     import numpy as np
#                     idx = np.argmin(np.abs(np.array(freq_values) - 100))
#                     psrr_100hz = psrr_values[idx]
#                     print(f"✓ Found PSRR @ 100Hz: {psrr_100hz:.3f} dB")
            
#             return BandgapResult(
#                 W_mp=W_mp, W_mn=W_mn,
#                 W_mp_startup=W_mp_startup, W_mn_startup=W_mn_startup,
#                 L_mp=L_mp, L_mn=L_mn, L_startup=L_startup,
#                 m_mp=m_mp, m_mn=m_mn,
#                 m_mp_startup=m_mp_startup, m_mn_startup=m_mn_startup,
#                 m_q1=m_q1, m_q2=m_q2, m_q3=m_q3,
#                 L_ra=L_ra, L_rb=L_rb,
#                 vref=vref, vref_min=vref, vref_max=vref,
#                 tc=0.0,
#                 power_uw=power_uw,
#                 i_q1_ua=i_q1_ua, i_q2_ua=i_q2_ua,
#                 i_out_ua=i_out_ua,
#                 line_regulation=line_regulation,
#                 psrr_100hz=psrr_100hz,
#                 vref_at_1p8v=vref_at_min_vdd,
#                 vref_at_3p3v=vref_at_max_vdd
#             )
    
#         except Exception as e:
#             print(f"❌ Simulation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
            
#         finally:
#             # ALWAYS return to original directory
#             os.chdir(original_dir)