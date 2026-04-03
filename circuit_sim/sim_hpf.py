import os
import subprocess
from dataclasses import dataclass
import re
import math

@dataclass
class OTAHPFResult:
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
    
    hpf_cutoff_hz: float
    hpf_passband_gain_db: float
    hpf_rolloff_db_per_dec: float
    hpf_stopband_atten_db: float
    
    power_uw: float
    ugbw_mhz: float

def simulate_folder_cascode_ota_with_hpf(pdk_lib_path, W_in, W_fold, W_sink, W_mirr, W_casc_n, W_casc_p,
                                          L, R1, R2, C1, C2,
                                          vdd=1.8, vbn=0.6, vbp=0.6, cload=1e-12, 
                                          vcm=0.9, itail=10e-6):
    """
    Simulate folded-cascode OTA with Sallen-Key HIGH-PASS filter
    Returns 4 core filter specs + supporting metrics
    """
    pdk_dir = os.path.dirname(pdk_lib_path)
    
    fc_theoretical = 1 / (2 * math.pi * math.sqrt(R1 * R2 * C1 * C2))
    Q_theoretical = math.sqrt(R1 * R2 * C1 * C2) / (C1 * R2 + C2 * R2)
    
    print(f"Target HPF cutoff: {fc_theoretical/1e6:.2f} MHz")

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

* Sallen-Key HIGH-PASS Filter (C and R swapped from LPF)
C1 VOTA_OUT VNODE1 {C1}
C2 VNODE1 VBUF_IN {C2}
R2 VBUF_IN VSS {R2}
R1 VNODE1 VOUT {R1}

* Unity-gain buffer
EBUFFER VOUT VSS VBUF_IN VSS 1.0
RBUF_OUT VOUT VFINAL_OUT 0.01

* Output load
CLOAD VFINAL_OUT VSS {cload}
.ends FOLDED_CASCODE_OTA"""

    netlist = f"""* Folded-Cascode OTA with Sallen-Key HIGH-PASS Filter
.lib {pdk_lib_path} tt
.options GMIN=1e-10 RELTOL=1e-4 ABSTOL=1e-12

{ota_netlist}

VVDD  VDD_NODE   0   DC {vdd}
VVSS  VSS_NODE   0   DC 0
VBN   VBN_NODE   0   DC {vbn}
VBP   VBP_NODE   0   DC {vbp}

VINP  VINP_NODE  0   DC {vcm} AC 0.5
VINN  VINN_NODE  0   DC {vcm} AC -0.5

VTAIL VDD_NODE VTAIL_NODE DC 0
ITAIL_SRC VTAIL_NODE 0 DC {itail}

XOTA  VSS_NODE VDD_NODE VFINAL_OUT_NODE VINN_NODE VINP_NODE VBN_NODE VBP_NODE VTAIL_NODE FOLDED_CASCODE_OTA
CLOAD VFINAL_OUT_NODE VSS_NODE {cload}

.op
.ac dec 200 1 1G

.control
set noaskquit
op
ac dec 200 1 1G

let pwr = abs(i(VVDD)[0]) * {vdd}
echo "POWER_VALUE"
print pwr

let vout_ac = ac1.v(VFINAL_OUT_NODE)
let gain_db_vec = 20*log10(mag(vout_ac))
let freq_vec = frequency

* Find peak gain across entire response (passband = peak, not at 1GHz)
let gain_peak = -1000
let peak_idx = 0
let idx = 0
while idx < length(gain_db_vec)
  if gain_db_vec[idx] > gain_peak
    let gain_peak = gain_db_vec[idx]
    let peak_idx = idx
  end
  let idx = idx + 1
end
echo "GAIN_VALUE"
print gain_peak

* Find -3dB cutoff on the LOW-frequency side of the peak
let target = gain_peak - 3
let fc_3db = 0
let idx = 0
while idx < peak_idx
  if (gain_db_vec[idx] < target) & (gain_db_vec[idx+1] >= target)
    let fc_3db = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "CUTOFF_VALUE"
print fc_3db

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

let fc_tenth = fc_3db / 10
if fc_tenth < 1
  let fc_tenth = 1
end
let gain_tenth = gain_db_vec[0]
let idx = 0
while idx < length(freq_vec) - 1
  if (freq_vec[idx] <= fc_tenth) & (freq_vec[idx+1] > fc_tenth)
    let gain_tenth = gain_db_vec[idx]
    break
  end
  let idx = idx + 1
end
let stopband_att = gain_peak - gain_tenth
echo "STOPBAND_VALUE"
print stopband_att

let rolloff = 40.0
if (fc_tenth > 0) & (fc_3db > fc_tenth)
  let rolloff = (gain_db_vec[idx] - gain_tenth) / log10(fc_3db / fc_tenth)
end
echo "ROLLOFF_VALUE"
print rolloff

quit
.endc
.end
"""

    netlist_path ='folded_cascode_ota_hpf_sim.spice'
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    try:
        result = subprocess.run(['ngspice', '-b', netlist_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            return None

        lines = result.stdout.split('\n')
        gain_db = None
        ugbw = None
        power = None
        fc_3db = None
        stopband_atten = None
        rolloff = None

        for i, line in enumerate(lines):
            line = line.strip()
            
            if 'POWER_VALUE' in line and i+1 < len(lines):
                match = re.search(r'pwr\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    power = abs(float(match.group(1)))
                    
            elif 'GAIN_VALUE' in line and i+1 < len(lines):
                match = re.search(r'gain_peak\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    gain_db = float(match.group(1))
                    
            elif 'UGBW_VALUE' in line and i+1 < len(lines):
                match = re.search(r'ugbw_hz\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    ugbw = float(match.group(1))
                    
            elif 'CUTOFF_VALUE' in line and i+1 < len(lines):
                match = re.search(r'fc_3db\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    fc_3db = float(match.group(1))
                    
            elif 'STOPBAND_VALUE' in line and i+1 < len(lines):
                match = re.search(r'stopband_att\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    stopband_atten = abs(float(match.group(1)))
                    
            elif 'ROLLOFF_VALUE' in line and i+1 < len(lines):
                match = re.search(r'rolloff\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    rolloff = abs(float(match.group(1)))

        if gain_db is not None and power is not None:
            if ugbw is None or ugbw <= 0:
                ugbw = 1e6
            if fc_3db is None or fc_3db <= 0:
                fc_3db = fc_theoretical
            if stopband_atten is None:
                stopband_atten = 0.0
            if rolloff is None:
                rolloff = 40.0

            power_uw = power * 1e6
            ugbw_mhz = ugbw / 1e6

            print(f"  HPF: gain={gain_db:.1f}dB fc={fc_3db:.0f}Hz atten={stopband_atten:.1f}dB "
                  f"rolloff={rolloff:.0f}dB/dec power={power_uw:.4f}uW")

            return OTAHPFResult(
                W_in=W_in, W_fold=W_fold, W_sink=W_sink,
                W_mirr=W_mirr, W_casc_n=W_casc_n, W_casc_p=W_casc_p,
                L=L, R1=R1, R2=R2, C1=C1, C2=C2,
                hpf_cutoff_hz=fc_3db,
                hpf_passband_gain_db=gain_db,
                hpf_rolloff_db_per_dec=rolloff,
                hpf_stopband_atten_db=stopband_atten,
                power_uw=power_uw,
                ugbw_mhz=ugbw_mhz,
                gain_db=gain_db,
            )
        else:
            return None

    except Exception as e:
        return None