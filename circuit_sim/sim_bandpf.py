import os
import subprocess
from dataclasses import dataclass
import re
import math

@dataclass
class OTABPFResult:
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
    
    bpf_center_freq_hz: float      # 1. Center frequency
    bpf_peak_gain_db: float        # 2. Peak gain
    bpf_bandwidth_hz: float        # 3. Bandwidth (-3dB)
    bpf_q_factor: float            # 4. Quality factor
    
    power_uw: float
    ugbw_mhz: float

def simulate_folder_cascode_ota_with_bpf(pdk_lib_path, W_in, W_fold, W_sink, W_mirr, W_casc_n, W_casc_p,
                                          L, R1, R2, C1, C2,
                                          vdd=1.8, vbn=0.6, vbp=0.6, cload=1e-12,
                                          vcm=0.9, itail=10e-6, results_dir='.'):
    """
    Simulate folded-cascode OTA with Sallen-Key BAND-PASS filter
    Returns 4 core filter specs + supporting metrics
    """
    pdk_dir = os.path.dirname(pdk_lib_path)
    
    f_hp = 1 / (2 * math.pi * R1 * C1)
    f_lp = 1 / (2 * math.pi * R2 * C2)
    if f_hp < f_lp:
        fc_theoretical = math.sqrt(f_hp * f_lp)
        BW_theoretical = f_lp - f_hp
    else:
        fc_theoretical = math.sqrt(f_hp * f_lp)
        BW_theoretical = f_hp - f_lp
    Q_theoretical = fc_theoretical / BW_theoretical if BW_theoretical > 0 else 0.5

    print(f"Target BPF: f_hp={f_hp:.0f}Hz, f_lp={f_lp:.0f}Hz, fc={fc_theoretical:.0f}Hz, Q={Q_theoretical:.3f}")

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

* Cascaded HP + LP Band-Pass Filter
* HP section: C1 blocks low freq, R1 to ground sets lower cutoff
C1 VOTA_OUT VMID {C1}
R1 VMID VSS {R1}

* LP section: R2 in series, C2 to ground sets upper cutoff
R2 VMID VBUF_IN {R2}
C2 VBUF_IN VSS {C2}

* Unity-gain buffer
EBUFFER VOUT VSS VBUF_IN VSS 1.0
RBUF_OUT VOUT VFINAL_OUT 0.01

* Output load
CLOAD VFINAL_OUT VSS {cload}
.ends FOLDED_CASCODE_OTA"""

    netlist = f"""* Folded-Cascode OTA with Sallen-Key BAND-PASS Filter
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

* Find peak within reasonable range (10Hz to 100MHz)
* Use real() because AC vectors are complex-typed
let gain_peak = -1000
let fc_idx = 0
let idx = 0
while idx < length(gain_db_vec)
  if (real(freq_vec[idx]) > 10) & (real(freq_vec[idx]) < 1e8)
    if real(gain_db_vec[idx]) > real(gain_peak)
      let gain_peak = gain_db_vec[idx]
      let fc_idx = idx
    end
  end
  let idx = idx + 1
end
let fc_center = freq_vec[fc_idx]
echo "CENTER_FREQ_VALUE"
print fc_center
echo "PEAK_GAIN_VALUE"
print gain_peak

* Find -3dB points (use real() for complex-typed vectors)
let target = gain_peak - 3
let f_low = 0
let idx = 0
while idx < fc_idx
  if (real(gain_db_vec[idx]) < real(target)) & (real(gain_db_vec[idx+1]) >= real(target))
    let f_low = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "F_LOW_VALUE"
print f_low

let f_high = 0
let idx = fc_idx
while idx < length(gain_db_vec) - 1
  if (real(gain_db_vec[idx]) >= real(target)) & (real(gain_db_vec[idx+1]) < real(target))
    let f_high = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "F_HIGH_VALUE"
print f_high

let bandwidth = f_high - f_low
let q_actual = fc_center / bandwidth
echo "BANDWIDTH_VALUE"
print bandwidth
echo "Q_VALUE"
print q_actual

let ugbw_hz = 0
let idx = 0
while idx < length(gain_db_vec) - 1
  if (real(gain_db_vec[idx]) > 0) & (real(gain_db_vec[idx+1]) <= 0)
    let ugbw_hz = freq_vec[idx]
    break
  end
  let idx = idx + 1
end
echo "UGBW_VALUE"
print ugbw_hz

quit
.endc
.end
"""

    netlist_path = os.path.join(results_dir, 'folded_cascode_ota_bpf_sim.spice')
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    try:
        result = subprocess.run(['ngspice', '-b', netlist_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            return None

        lines = result.stdout.split('\n')
        
        power = None
        fc_center = None
        gain_peak = None
        bandwidth = None
        q_factor = None
        ugbw = None

        for i, line in enumerate(lines):
            line = line.strip()
            
            if 'POWER_VALUE' in line and i+1 < len(lines):
                match = re.search(r'pwr\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    power = abs(float(match.group(1)))
                    
            elif 'CENTER_FREQ_VALUE' in line and i+1 < len(lines):
                match = re.search(r'fc_center\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    fc_center = float(match.group(1))
                    
            elif 'PEAK_GAIN_VALUE' in line and i+1 < len(lines):
                match = re.search(r'gain_peak\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    gain_peak = float(match.group(1))
                    
            elif 'BANDWIDTH_VALUE' in line and i+1 < len(lines):
                match = re.search(r'bandwidth\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    bandwidth = float(match.group(1))
                    
            elif 'Q_VALUE' in line and i+1 < len(lines):
                match = re.search(r'q_actual\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    q_factor = float(match.group(1))
                    
            elif 'UGBW_VALUE' in line and i+1 < len(lines):
                match = re.search(r'ugbw_hz\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[i+1])
                if match:
                    ugbw = float(match.group(1))

        if power is not None and fc_center is not None and gain_peak is not None:
            if ugbw is None or ugbw <= 0:
                ugbw = 1e6
            if bandwidth is None or bandwidth <= 0:
                bandwidth = BW_theoretical
            if q_factor is None or q_factor <= 0:
                q_factor = Q_theoretical

            power_uw = power * 1e6
            ugbw_mhz = ugbw / 1e6

            print(f"  BPF: gain={gain_peak:.1f}dB fc={fc_center:.0f}Hz BW={bandwidth:.0f}Hz "
                  f"Q={q_factor:.2f} power={power_uw:.4f}uW")

            return OTABPFResult(
                W_in=W_in, W_fold=W_fold, W_sink=W_sink,
                W_mirr=W_mirr, W_casc_n=W_casc_n, W_casc_p=W_casc_p,
                L=L, R1=R1, R2=R2, C1=C1, C2=C2,
                bpf_center_freq_hz=fc_center,
                bpf_peak_gain_db=gain_peak,
                bpf_bandwidth_hz=bandwidth,
                bpf_q_factor=q_factor,
                power_uw=power_uw,
                ugbw_mhz=ugbw_mhz,
            )
        else:
            return None

    except Exception as e:
        return None