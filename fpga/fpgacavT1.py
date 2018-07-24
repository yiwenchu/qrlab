import numpy as np
from pulseseq import sequencer, pulselib
import fpgapulses

import fpgameasurement
REPRATE_DELAY = 1000000      # Delay to get a reasonable repetition rate

delays = np.round(np.linspace(0, 800, 41))
m = fpgameasurement.FPGAMeasurement('cavT1', xs=delays, fit_func='cohstate_decay', fit_func_kwargs=dict(n=0))

s = sequencer.Sequence()
s.append(sequencer.Delay(256, label='start'))
for delay in delays:
#    s.append(sequencer.Constant(1000, 1, chan='m0'))
    s.append(m.cavity.displace(2.0))#*np.exp(1j*np.pi/4)))
    if delay != 0:
        s.append(sequencer.Delay(delay*1000))
    s.append(m.qubit.rotate(np.pi, 0))
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(sequencer.Delay(REPRATE_DELAY - delay*1000))
s.append(sequencer.Delay(256, jump='start'))

m.set_seq(s, len(delays))
#m.generate(plot=True)
m.start_exp()
m.plot_se()
