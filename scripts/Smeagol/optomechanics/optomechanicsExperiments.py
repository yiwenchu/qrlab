import mclient
from mclient import instruments
from datetime import datetime
import time
import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from matplotlib import gridspec
import matplotlib.pyplot as plt
import logging
import measurementOM
from measurement import *


AS_info = mclient.get_qubit_info('AS_info')
awg1 = instruments['AWG1']
funcgen = instruments['funcgen']
alazar = instruments['alazar']


#AS_r = AS_info.rotate
#AS_r1 = AS_r(np.pi, 0, amp = 1)
AS_SC0 = AS_info.sideband_channels[0]
AS_SC1 = AS_info.sideband_channels[1]
#LO_SC0 = LO_info.sideband_channels[0]
#LO_SC1 = LO_info.sideband_channels[1]
LO_SC0 = 3

#Gaussian square pulse with length (ns), amplitude (V), and risetime (ns)
AS_p0 = GaussSquare(100, 1.0, 10, chan = AS_SC0)
AS_p1 = GaussSquare(100, 0.0, 10, chan = AS_SC1)
AS_p = Combined([AS_p0, AS_p1])

#LO_p0 = Constant(500, 1.0, chan = LO_SC0)
#LO_p1 = Constant(500, 0.0, chan = LO_SC1)
#LO_p = Combined([LO_p0, LO_p1])

LO_p = Sequence(Constant(500, 1.0, chan = LO_SC0))
mod = ModulateSequence(20, 0, 1)
LO_p = mod.apply(LO_p)

RO_p = Sequence(Constant(200, 1.0, chan='4m2'))
#
seq = Combined([AS_p, LO_p, RO_p])

seq1 = Sequence()
seq1.append(Trigger(250))
#seq1.append(LO_p)
seq1.append(seq)
#seq1.append(RO_p)

m = measurementOM.MeasurementOM([AS_info])#, LO_info
s = m.generate(seq1)
m.load()
m.start_awgs()
time.sleep(1)


alz = instruments['alazar']
alz.setup_channels()
alz.setup_clock()
alz.setup_trigger()

if 1:
    alz.setup_shots(1)
    buf = alz.take_raw_shots()
    plt.figure()
    nsamp = alz.get_nsamples()
    plt.plot(buf[:nsamp], label='A')
    plt.plot(buf[nsamp:2*nsamp], label='B')
    plt.suptitle('Raw single shot')
    plt.legend()
    plt.xlabel('Time [ns]')
    
if 1:
    alz.setup_avg_shot(10000)
    buf = alz.take_avg_shot(timeout=50000)

    plt.figure()
    plt.suptitle('Average demodulated shot')

    plt.subplot(211)
    plt.plot(np.abs(buf))
    plt.xlabel('IF period #')

    plt.subplot(212)
    plt.plot(np.real(buf), np.imag(buf))
    plt.xlabel('I')
    plt.ylabel('Q')

#m =measurement.Measurement([AS_info])
#m.measure()
#
#class OMMeasurement(Measurement1D):
#
#    def __init__(self, qubit_info, **kwargs):
##        self.qubit_info = qubit_info
##        self.delays = np.array(np.round(delays), dtype=int)
##        self.xs = delays / 1e3      # For plotting purposes
##        self.double_exp = double_exp
##        if seq is None:
##            seq = Trigger(250)
##        self.seq = seq
##        self.postseq = postseq
##        self.bgcor = bgcor
##
##        npoints = len(delays)
##        if bgcor:
##            npoints += 2
#        super(OMMeasurement, self).__init__(1, infos=qubit_info, **kwargs)
##        self.data.set_attrs(
##            rep_rate=self.instruments['funcgen'].get_frequency()
##        )
##        self.data.create_dataset('delays', data=delays)
#
#    def generate(self):
#        seq1 = Sequence()
#        seq1.append(Trigger(250))
#        #seq1.append(LO_p)
#        seq1.append(seq)
#        
#        s = self.get_sequencer(seq1)
#        seqs = s.render()
#        self.seqs = seqs
#        return seqs
#        
#m = OMMeasurement(AS_info)
##m.measure()