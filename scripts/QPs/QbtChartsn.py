import mclient
reload(mclient)
import numpy as np
import matplotlib.pyplot as plt
from pulseseq import sequencer, pulselib
import matplotlib as mpl

#mpl.rcParams['figure.figsize']=[5,3.5]
#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'k']
alz = mclient.instruments['alazar']
fg = mclient.instruments['funcgen']
ag3 = mclient.instruments['ag3']

qubits = mclient.get_qubits()
qubit_info = mclient.get_qubit_info('qubit2ge')
ef_info = mclient.get_qubit_info('qubit2ef')
#gf_info = mclient.get_qubit_info('qubit1gf')
cavity_info1R = mclient.get_qubit_info('cavity1R')
cavity_info1A = mclient.get_qubit_info('cavity1A')
cavity_info1B = mclient.get_qubit_info('cavity1B')
Qswitch_info1A = mclient.get_qubit_info('Qswitch1A')
Qswitch_info1B = mclient.get_qubit_info('Qswitch1B')

#qubit_info2 = mclient.get_qubit_info('qDblCoax2')
#ef_info2 = mclient.get_qubit_info('eDblCoax2')

from scripts.single_qubit import rabi

"""Power Rabi -- Pi pulse calibration"""
if 0: # Calibrate pi pulse
    for i in range(8):
#        ag3.set_rf_on(False)
        tr = rabi.Rabi(qubit_info, np.linspace(-0.8, 0.8, 81), plot_seqs=False, update=False, seq=None, selective=False)#, fix_period=0.00924)
        tr.measure()
        tr = rabi.Rabi(qubit_info, np.linspace(-0.06, 0.06, 81), plot_seqs=False, update=False, seq=None, selective=True, singleshotbin=True)#, fix_period=0.003108)
        tr.measure()
        bla

if 1: #EF rabi
    from scripts.single_qubit import efrabi
    for reprate in [100]:
        fg.set_frequency(reprate)
        alz.set_naverages(600)
        efr = efrabi.EFRabi(qubit_info, ef_info, np.linspace(-0.8, 0.8, 61), seq=None, second_pi=True, singleshotbin=True)
        efr.measure()

        period = efr.fit_params['period'].value
        alz.set_naverages(2000)
        efr = efrabi.EFRabi(qubit_info, ef_info, np.linspace(-0.7, 0.7, 61), first_pi=False, force_period= period, singleshotbin=True)
        efr.measure()
    bla

if 0: # T1
    from scripts.single_qubit import T1measurement
#    seq = sequencer.Join([sequencer.Trigger(250), cavity_info1A.rotate(3.0, 0), cavity_info1B.rotate(3.0, 0)])
    t1 = T1measurement.T1Measurement(qubit_info, np.concatenate((np.linspace(0, 80e3, 41), np.linspace(81e3, 300e3, 61))), double_exp=False,
                                     generate=True, seq=None, singleshotbin=True)
    t1.measure()


if 1: # T2
    from scripts.single_qubit import T2measurement
    for j in range(2):
#    for j in [1000]:
#        fg.set_frequency(j)
#        seq = sequencer.Join([sequencer.Trigger(250), cavity_info1A.rotate(0.01, 0)])#, cavity_info1B.rotate(1.0, 0)])
        t2 = T2measurement.T2Measurement(qubit_info, np.linspace(0, 50e3, 101), detune=200e3, double_freq=False,
                                         singleshotbin=True, Qswitch_infoA=Qswitch_info1A, Qswitch_infoB=Qswitch_info1B,
                                         )#extra_info=[Qswitch_info1A, Qswitch_info1B, qubit_info])
        t2.measure()
    bla

if 0: # T2echo
    for j in range(1):
        from scripts.single_qubit import T2measurement
        t2 = T2measurement.T2Measurement(qubit_info, np.linspace(0.5e3, 60e3, 101), detune=200e3,
                                         echotype = T2measurement.ECHO_HAHN, generate=True, singleshotbin=True)
        t2.measure()
    bla

if 0: # FT1
    from scripts.single_qubit import FT1measurement
    ft1 = FT1measurement.FT1Measurement(qubit_info, ef_info, np.linspace(0, 200e3, 101))
    ft1.measure()

if 0: # EFT2
    from scripts.single_qubit import EFT2measurement
    eft2 = EFT2measurement.EFT2Measurement(qubit_info, ef_info, np.linspace(0.5e3, 15e3, 101), detune=800e3, double_freq=False)#, echotype = EFT2measurement.ECHO_HAHN)
    eft2.measure()

if 1: # GFT2
    from scripts.single_qubit import GFT2measurement
    gft2 = GFT2measurement.GFT2Measurement(qubit_info, ef_info, np.linspace(0.5e3, 15e3, 101), detune=800e3, double_freq=False)#, echotype = GFT2measurement.ECHO_HAHN)
    gft2.measure()
    bla

if 0: #number splitting:
    from scripts.single_qubit import spectroscopy
    seq = sequencer.Join([sequencer.Trigger(250), cavity_info.rotate(np.pi, 0)])
#    postseq = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate(np.pi, 0)])
    qubit_freq = 6306.770e6
    spec = spectroscopy.Spectroscopy(mclient.instruments['brick2'], qubit_info, np.linspace(qubit_freq-8e6, qubit_freq+2e6, 101), [11.5],
                                     plen=6000, seq = seq, amp=0.09, extra_info=cavity_info, plot_seqs=True)
    spec.measure()


if 0: #Sideband modulated number splitting:
    from scripts.single_qubit import ssbspec
    seq = sequencer.Join([sequencer.Trigger(250), cavity_info2.rotate(np.pi*2, 0)])
    spec = ssbspec.SSBSpec(qubit_info, np.linspace(-15e6, 1e6, 151),
                           extra_info= cavity_info2,
                           seq =seq,  plot_seqs=False)
    spec.measure()
    bla

if 0: #EF Sideband modulated number splitting:
    from scripts.single_qubit import ssbspec
    seq = sequencer.Join([sequencer.Trigger(250), qubit_info.rotate(np.pi,0), cavity_info2.rotate(np.pi*1.0, 0)])
#    postseq = qubit_info.rotate(np.pi,0)
#    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0))
    spec = ssbspec.SSBSpec(ef_info, np.linspace(-3e6, 1e6, 121),
                           extra_info= [qubit_info, cavity_info2],
                           seq =seq,  postseq = None, plot_seqs=False)
    spec.measure()
    bla

if 0:
    from scripts.single_qubit import rabi_QP
    tr = rabi_QP.Rabi_QP(qubit_info, np.linspace(0, 1, 81), QP_delay = 10e3, inj_len = 30e3)
    tr.measure()

if 1: #mixer calibration:
    from scripts.single_qubit import mixer_calibration
    mixer_cal = mixer_calibration.Mixer_Calibration

    cal = mixer_cal('qubit2ge', 5012.050e6-130.000e6, 'VA', 'AWG2', verbose=True,
                        base_amplitude= 4,
                        va_lo='ag1_RO') # The frequency is the targeted lower sideband frequency, not the carrier
    cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
    cal.tune_lo(mode='coarse')
    cal.tune_osb(mode=(0.25, 1000, 3, 1))
    cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
                            # LO leakage may creep up during osb tuning

    # this function will set the correct qubit_info sideband phase for use in experiments
    #    i.e. combines the AWG skew with the  7036.120e6current sideband phase offset
    cal.set_tuning_parameters(set_sideband_phase=True)
    cal.load_test_waveform()
    cal.print_tuning_parameters()
    bla