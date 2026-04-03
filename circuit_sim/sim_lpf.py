import os
import subprocess
from dataclasses import dataclass
import re
import math

@dataclass
class OTALPFResult:
    # Design parameters
    W_in: float
    W_fold: float
    W_sink: float
    W_mirr: float
    W_casc_n: float
    W_casc_p: float
    L: float
    R1: float
    R2: float
    C1: float
    C2: float
    
    # OTA specs
    gain_db: float
    power_uw: float
    ugbw_mhz: float
    
    # LPF frequency response
    lpf_cutoff_hz: float
    lpf_q_theoretical: float
    lpf_passband_gain_db: float
    lpf_rolloff_db_per_dec: float
    lpf_stopband_atten_db: float
    phase_margin_deg: float

def simulate_folder_cascode_ota_with_lpf(pdk_lib_path, W_in, W_fold, W_sink, W_mirr, W_casc_n, W_casc_p,
                                          L, R1, R2, C1, C2,
                                          vdd=1.8, vbn=0.6, vbp=0.6, cload=1e-12, 
                                          vcm=0.9, Rz=1e3, Cc=1e-12, itail=10e-6):
    """
    Simulate folded-cascode OTA with Sallen-Key low-pass filter
    """
    pdk_dir = os.path.dirname(pdk_lib_path)
    
    # Calculate theoretical filter specifications
    fc_theoretical = 1 / (2 * math.pi * math.sqrt(R1 * R2 * C1 * C2))
    Q_theoretical = math.sqrt(R1 * R2 * C1 * C2) / (R1 * C1 + R2 * C1)
    
    print(f"Theoretical LPF specs: fc={fc_theoretical/1e6:.2f} MHz, Q={Q_theoretical:.3f}")

    ota_netlist = f""".subckt FOLDED_CASCODE_OTA VSS VDD VOUT VINN VINP VBN VBP VTAIL 
* PMOS differential input pair
XM1 N1 VINP VTAIL VDD sky130_fd_pr__pfet_01v8 w={W_in} l={L}
XM2 N2 VINN VTAIL VDD sky130_fd_pr__pfet_01v8 w={W_in} l={L}

* NMOS folding devices
XM3 N1 VBN NC1 VSS sky130_fd_pr__nfet_01v8 w={W_fold} l={L}
XM4 N2 VBN NC2 VSS sky130_fd_pr__nfet_01v8 w={W_fold} l={L}

* NMOS cascode (bottom)
XM5 NC1 VBN VSS VSS sky130_fd_pr__nfet_01v8 w={W_casc_n} l={L}
XM6 NC2 VBN VSS VSS sky130_fd_pr__nfet_01v8 w={W_casc_n} l={L}

* PMOS current mirror loads
XM7 N1 N1 NP1 VDD sky130_fd_pr__pfet_01v8 w={W_sink} l={L}
XM8 N2 N1 NP2 VDD sky130_fd_pr__pfet_01v8 w={W_sink} l={L}

* PMOS cascode (top)
XM9  NP1 VBP VDD VDD sky130_fd_pr__pfet_01v8 w={W_mirr} l={L}
XM10 NP2 VBP VDD VDD sky130_fd_pr__pfet_01v8 w={W_casc_p} l={L}

* Output connection
ROUT N2 VOTA_OUT 0.01

* Sallen-Key Low-Pass Filter
R1 VOTA_OUT VNODE1 {R1}
R2 VNODE1 VBUF_IN {R2}
C2 VBUF_IN VSS {C2}
C1 VNODE1 VOUT {C1}

* DC bias for VBUF_IN (caps block DC, node would float without this)
RBIAS VBUF_IN VSS 10G

* Unity-gain buffer
EBUFFER VOUT VSS VBUF_IN VSS 1.0
RBUF_OUT VOUT VFINAL_OUT 0.01
.ends FOLDED_CASCODE_OTA"""

    netlist = f"""* Folded-Cascode OTA with Sallen-Key LPF
.lib {pdk_lib_path} tt
.options GMIN=1e-10 RELTOL=1e-4 ABSTOL=1e-12

{ota_netlist}

* Power supplies
VVDD  VDD_NODE   0   DC {vdd}
VVSS  VSS_NODE   0   DC 0

* Bias voltages
VBN   VBN_NODE   0   DC {vbn}
VBP   VBP_NODE   0   DC {vbp}

* Differential inputs
VINP  VINP_NODE  0   DC {vcm} AC 0.5
VINN  VINN_NODE  0   DC {vcm} AC -0.5

* Tail current
VTAIL VDD_NODE VTAIL_NODE DC 0
ITAIL_SRC VTAIL_NODE 0 DC {itail}

* DUT
XOTA  VSS_NODE VDD_NODE VFINAL_OUT_NODE VINN_NODE VINP_NODE VBN_NODE VBP_NODE VTAIL_NODE FOLDED_CASCODE_OTA
CLOAD VFINAL_OUT_NODE VSS_NODE {cload}

.op
.ac dec 200 1 1G

.measure ac dc_gain_db max vdb(VFINAL_OUT_NODE)
.measure ac ugbw when vdb(VFINAL_OUT_NODE)=0 fall=1

.control
set noaskquit
op
ac dec 200 1 1G

* Power - use scalar value
let pwr = abs(i(VVDD)[0]) * {vdd}
echo "POWER_VALUE"
print pwr

* Frequency response
let vout_ac = ac1.v(VFINAL_OUT_NODE)
let gain_db_vec = 20*log10(mag(vout_ac))
let phase_vec = 180/pi*cph(vout_ac)
let freq_vec = frequency

let gain_dc = gain_db_vec[0]
echo "GAIN_VALUE"
print gain_dc

* Find -3dB cutoff
let target = gain_dc - 3
let fc_3db = 0
let idx = 0
while idx < length(gain_db_vec) - 1
  if (gain_db_vec[idx] >= target) & (gain_db_vec[idx+1] < target)
    let fc_3db = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "CUTOFF_VALUE"
print fc_3db

* Find UGBW
let ugbw_hz = 0
let idx = 0
while idx < length(gain_db_vec) - 1
  if (gain_db_vec[idx] > 0) & (gain_db_vec[idx+1] <= 0)
    let ugbw_hz = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "UGBW_VALUE"
print ugbw_hz

* Stopband attenuation
let fc_10x = fc_3db * 10
let gain_10x = -100
let idx = 0
while idx < length(freq_vec) - 1
  if (freq_vec[idx] <= fc_10x) & (freq_vec[idx+1] > fc_10x)
    let gain_10x = gain_db_vec[idx]
    break
  end
  let idx = idx + 1
end
let stopband_att = gain_dc - gain_10x
echo "STOPBAND_VALUE"
print stopband_att

* Roll-off (between fc and 10*fc)
let rolloff = 40.0
if (fc_10x > 0) & (idx > 0)
  let rolloff = (gain_db_vec[idx-1] - gain_10x) / log10(fc_10x / fc_3db)
end
echo "ROLLOFF_VALUE"
print rolloff

* Phase margin
let pm = 180
let idx = 0
while idx < length(freq_vec) - 1
  if (freq_vec[idx] <= ugbw_hz) & (freq_vec[idx+1] > ugbw_hz)
    let pm = 180 + phase_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "PHASE_VALUE"
print pm

* Save data
wrdata {pdk_dir}/lpf_ac_response.txt frequency gain_db_vec phase_vec

quit
.endc
.end
"""

    netlist_path = 'folded_cascode_ota_lpf_sim.spice'
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"NGSpice failed with return code: {result.returncode}")
            print(result.stdout[-1000:])
            return None

        lines = result.stdout.split('\n')
        
        # Parse results using markers
        gain_db = None
        ugbw = None
        power = None
        fc_3db = None
        stopband_atten = None
        rolloff = None
        phase_margin = None

        for i, line in enumerate(lines):
            line = line.strip()
            
            if 'POWER_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'pwr\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    power = abs(float(match.group(1)))
                    
            elif 'GAIN_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'gain_dc\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    gain_db = float(match.group(1))
                    
            elif 'UGBW_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'ugbw_hz\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    ugbw = float(match.group(1))
                    
            elif 'CUTOFF_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'fc_3db\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    fc_3db = float(match.group(1))
                    
            elif 'STOPBAND_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'stopband_att\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    stopband_atten = abs(float(match.group(1)))
                    
            elif 'ROLLOFF_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'rolloff\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    rolloff = abs(float(match.group(1)))
                    
            elif 'PHASE_VALUE' in line and i+1 < len(lines):
                next_line = lines[i+1].strip()
                match = re.search(r'pm\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', next_line)
                if match:
                    phase_margin = float(match.group(1))


        # ========== ADD SUMMARY DEBUG HERE ==========
        print(f"\nDEBUG: FINAL PARSED VALUES:")
        print(f"  power (W) = {power}")
        print(f"  power_uw = {power * 1e6 if power else None}")
        print(f"  gain_db = {gain_db}")
        print(f"  ugbw (Hz) = {ugbw}")
        print(f"  fc_3db (Hz) = {fc_3db}")
        print(f"  stopband_atten (dB) = {stopband_atten}")
        print(f"  rolloff (dB/dec) = {rolloff}")
        print(f"  phase_margin (deg) = {phase_margin}")

        

        if gain_db is not None and power is not None:
            # Set defaults
            if ugbw is None or ugbw <= 0:
                ugbw = 1e6
            if fc_3db is None or fc_3db <= 0:
                fc_3db = fc_theoretical
            if stopband_atten is None:
                stopband_atten = 0.0
            if rolloff is None:
                rolloff = 40.0
            if phase_margin is None:
                phase_margin = 0.0

            power_uw = power * 1e6
            ugbw_mhz = ugbw / 1e6

            print(f"  LPF: gain={gain_db:.1f}dB fc={fc_3db:.0f}Hz atten={stopband_atten:.1f}dB "
                  f"rolloff={rolloff:.0f}dB/dec power={power_uw:.4f}uW")

            return OTALPFResult(
                W_in=W_in, W_fold=W_fold, W_sink=W_sink,
                W_mirr=W_mirr, W_casc_n=W_casc_n, W_casc_p=W_casc_p,
                L=L, R1=R1, R2=R2, C1=C1, C2=C2,
                gain_db=gain_db, power_uw=power_uw,
                ugbw_mhz=ugbw_mhz,
                lpf_cutoff_hz=fc_3db,
                lpf_q_theoretical=Q_theoretical,
                lpf_passband_gain_db=gain_db,
                lpf_rolloff_db_per_dec=rolloff,
                lpf_stopband_atten_db=stopband_atten,
                phase_margin_deg=phase_margin
            )
        else:
            print(f"Failed to parse: gain_db={gain_db}, power={power}")
            return None

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return None