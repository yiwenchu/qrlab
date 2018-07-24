import mclient
reload(mclient)
import numpy as np
from pulseseq import sequencer
import matplotlib as mpl
import math as math


mpl.rcParams['figure.figsize']=[5,3.5]
#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'k']
alz = mclient.instruments['alazar']
fg = mclient.instruments['funcgen']
laserfg = mclient.instruments['laserfg']

# Load old settings.
if 0:
    toload = ['AWG1','ag1','ag2', 'ag3' 'alazar', 'qFC14#1', 'eFC14#1','qubit_DO13#3', 'ef_DO13#3', 'qubit_DO13#4', 'ef_DO13#4']
    mclient.load_settings_from_file(r'c:\_data\settings\20131214\165409.set', toload)    # Last time-Rabi callibration
    bla

qubits = mclient.get_qubits()
qubit_info = mclient.get_qubit_info('qubit2ge')
ef_info = mclient.get_qubit_info('qubit2ef')
#gf_info1 = mclient.get_qubit_info('Qubit1gf')
cavity_info1A = mclient.get_qubit_info('cavity1A')
cavity_info1B = mclient.get_qubit_info('cavity1B')
Qswitch_info1A = mclient.get_qubit_info('Qswitch1A')
Qswitch_info1B = mclient.get_qubit_info('Qswitch1B')

#qubit_info2 = mclient.get_qubit_info('qDblCoax2')
#cavity_info2A = mclient.get_qubit_info('cavity2A')
#cavity_info2B = mclient.get_qubit_info('cavity2B')

# Find read-out cavity and choose a power
if 0:
    from scripts.single_cavity import rocavspectroscopy
    rofreq = 7697.50e6 # 9026.41e6
    freq_range = 1e6
#    seq = sequencer.Sequence([sequencer.Trigger(250), cavity_info1A.rotate(2, 0), cavity_info1B.rotate(2, 0)])
#    seq = sequencer.Sequence([sequencer.Trigger(250), qubit_info1.rotate(np.pi, 0), ef_info1.rotate(np.pi, 0)])
    ro = rocavspectroscopy.ROCavSpectroscopy(qubit_info, np.linspace(-28, 0, 1), np.linspace(rofreq-freq_range, rofreq+freq_range, 61),
                                             qubit_pulse=False, seq=None)
    ro.measure()
    bla

#Find qubit
if 0:
    from scripts.single_qubit import spectroscopy
#    from scripts.single_qubit import spectroscopy_IQ
    qubit_freq = 5002.704e6
    freq_range = 5e6
#    seq = sequencer.Sequence([sequencer.Trigger(250), ef_info.rotate(np.pi, 0)])
    spec = spectroscopy.Spectroscopy(mclient.instruments['brick2'], qubit_info,
                                 np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 121), [0],
                                 plen=20000, amp=0.10, seq=None, plot_seqs=False) #1=1ns5
#    spec = spectroscopy_IQ.Spectroscopy_IQ(client.instruments['gen'], qubit_info,
#                                     np.linspace(702e6, 710e6, 81), [-30],
#                                    plen=250*100, amp=0.1, ssb=False, plot_seqs=False)
    spec.measure()
    bla

if 0: #SSB spec
    from scripts.single_qubit import ssbspec
#    seq = sequencer.Sequence([sequencer.Trigger(250), cavity_info1B.rotate(1, 0)])
    spec = ssbspec.SSBSpec(qubit_info, np.linspace(-10e6, 1e6, 101),  plot_seqs=False,
                           seq=None, singleshotbin=True, generate=True)
    spec.measure()
    bla

if 0: # cavity spectroscopy
    from scripts.single_cavity import cavspectroscopy
    cav_freq = 5556.97e6
    cspec = cavspectroscopy.CavSpectroscopy(mclient.instruments['brick3'], qubit_info, cavity_info1B,
                                            [0.01], np.linspace(cav_freq-0.5e6, cav_freq+0.5e6, 61), Qswitchseq=None)
    cspec.measure()
    bla

if 1: # pump tone spectroscopy
    from scripts.single_cavity import cavspectroscopy
    cav_freq = 7782.70e6
    seq = sequencer.Join([sequencer.Delay(10000),
            sequencer.Combined([
            Qswitch_info1A.rotate(np.pi, 0),    # 250us square pulse pump
            Qswitch_info1B.rotate(np.pi, 0),
            sequencer.Repeat(sequencer.Constant(1000, 0.0001, chan=5), 250),      # Qubit/Readout master switch
            ]), sequencer.Delay(20000)])
    cspec = cavspectroscopy.CavSpectroscopy(mclient.instruments['brick4'], qubit_info, cavity_info1B,
                                            [1.2], np.linspace(cav_freq-0.5e6, cav_freq+0.5e6, 61),
                                            Qswitchseq=seq, extra_info=[Qswitch_info1A, Qswitch_info1B])
    cspec.measure()
    bla

if 0: #Find qubit ef
    from scripts.single_qubit import spectroscopy
    ef_freq = 4959.00e6
    seq = sequencer.Sequence([sequencer.Trigger(250), qubit_info1.rotate(np.pi, 0)])
    postseq = sequencer.Sequence(qubit_info1.rotate(np.pi, 0))
    spec = spectroscopy.Spectroscopy(mclient.instruments['ag3'], ef_info, np.linspace(ef_freq-5e6, ef_freq+5e6, 101), [-32],
                                     plen=2000, amp=0.004,
                                     seq=seq, postseq=postseq,
                                     extra_info=qubit_info, plot_seqs=False)
    spec.measure()
    bla