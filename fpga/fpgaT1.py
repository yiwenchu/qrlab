import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
import YngwieEncoding as ye

N = 61                      # Number of elements in sequence
DT = 2500
delays = np.arange(N) * DT
REPRATE_DELAY = 500000      # Delay to get a reasonable repetition rate
m = fpgameasurement.FPGAMeasurement('T1', xs=delays/1000., fit_func='exp_decay', rrec=False)

s = sequencer.Sequence()

# FPGA style, using R13 as dynamic instruction length
if 1:
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(13, 0), master_counter0=N, label='reset'))
    s.append(m.qubit.rotate(np.pi, 0, label='start'))
    s.append(sequencer.Delay(100, inslength=('R13', 10)))       # Length = R13 + 10
#    s.append(sequencer.Constant(512, 1, value=1, chan='m0'))    # For debugging purposes
    s.append(fpgapulses.LongMeasurementPulse(label='measure'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.ADDI(13, DT/4), master_counter0=-1, master_internal_function=ye.FPGASignals.c0))
    s.append(sequencer.Delay(REPRATE_DELAY, jump=('reset', 'start')))

# Old-school
else:
#    m.qubit.rotate = fpgapulses.FPGAMarkerLenRotation(72, 'm0', 1)
    s.append(sequencer.Delay(256, label='start'))
    for curdelay in delays:
        s.append(m.qubit.rotate(np.pi, 0))
        if curdelay != 0:
            s.append(sequencer.Delay(curdelay))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(sequencer.Delay(REPRATE_DELAY - curdelay))
    s.append(sequencer.Delay(256, jump='start'))

m.set_seq(s, len(delays))
m.start_exp()
#m.plot_histogram()
m.plot_se()
plt.suptitle('T1 = %.01f +- %.01f us' % (m.fit_params['tau'].value, m.fit_params['tau'].stderr))
m.save_fig()