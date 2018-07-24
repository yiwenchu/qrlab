import mclient
import numpy as np
from pulseseq import sequencer, pulselib
import matplotlib as mpl


#mpl.rc_params['figure.figsize']=[8,6]

qubit_info = mclient.get_qubit_info('qubit_info')
#ef_info = mclient.get_qubit_info('qubit_ef_info')
#gf_info = mclient.get_qubit_info('Qubit1gf')
alz = mclient.instruments['alazar']
ag1 = mclient.instruments['ag1']
#laser_info = mclient.instruments['laserfg']
#voltages =np.linspace(0.4, 1.2, 1)

# Check histogramming on GE
if 1:
    avg = alz.get_naverages()
    alz.set_naverages(50000)
    from scripts.single_qubit import rabi
#    for power in np.linspace(7, 10, 1):
#        ag1.set_power(power)
#    tr.data.set_attrs(readout_power=ag1.get_power())
#    tr = rabi.Rabi(qubit_info, [qubit_info.pi_amp/2,], real_signals=False, histogram=True, title='|g>+|e>')
#    tr.measure()
    
    tr = rabi.Rabi(qubit_info, [qubit_info.pi_amp,], histogram=True, title='|e>')
    tr.measure()

    #seq = sequencer.Join([sequencer.Trigger(250),sequencer.Combined([cavity_info1A.rotate(1.5, 0), cavity_info1B.rotate(1.5, 0)])])
    tr = rabi.Rabi(qubit_info, [0.00,], seq=None, histogram=True, title='|g>')
    tr.measure()
    
#    seq =sequencer.Sequence()
#    seq.append(sequencer.Trigger(250))
#    seq.append(sequencer.Constant(110e3, 1, chan="3m1"))
#    seq.append(sequencer.Delay(100e3))
#    tr = rabi.Rabi(qubit_info, [0.00,], seq=seq, histogram=True, title='|g>')
#    tr.measure()
#    postseq = sequencer.Sequence(ef_info.rotate(np.pi,0))
#    tr = rabi.Rabi(qubit_info, [qubit_info.pi_amp,], postseq=postseq, histogram=True, title='|f>', extra_info=ef_info)
#    tr.measure()
    alz.set_naverages(avg)
    bla

# Check histogramming on GE
if 0:
    alz.set_naverages(50000)
    from scripts.single_qubit import rabi

#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)

#    tr = rabi.Rabi(qubit_ef_info, [qubit_ef_info.pi_amp,], histogram=True, title='|e>')
#    tr.measure()
    tr = rabi.Rabi(qubit_ef_info, [0.00,],seq=None, histogram=True, title='|g>')
    tr.measure()

    alz.set_naverages(2000)
    bla

if 0:
    from scripts.single_qubit import efrabi
    tr = efrabi.EFRabi(qubit_info, qubit_info, [qubit_info.pi_area,], second_pi=False, real_signals=False, histogram=True, title='|f>')
    tr.measure()
    bla



if 0:
    alz.set_naverages(20000)
    from scripts.single_qubit import rabi_IQ
    tr = rabi_IQ.Rabi(qubit_info, [qubit_info.pi_area,], real_signals=True, histogram=True, title='|e>')
    tr.measure()
    tr = rabi_IQ.Rabi(qubit_info, [0.00,], real_signals=True, histogram=True, title='|g>')
    tr.measure()


if 0:
    from scripts.single_qubit import rabi
    alz.set_naverages(50000)
    seq = sequencer.Sequence([sequencer.Trigger(250), cavity_info1A.rotate(0, 0), cavity_info1B.rotate(0, 0)])
    tr = rabi.Rabi(qubit_info, [0.00,], histogram=True, seq=seq, title='|g>', generate=True, extra_info=[cavity_info1A, cavity_info1B])
    tr.measure()
