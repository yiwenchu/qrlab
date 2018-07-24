# Example for the most basic pulses using pulseseq
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()
seq.append(Constant(250, 0.5, chan=1))
seq.append(Gaussian(100, 0.5, chan=2))
seq.append(Lorentzian(100, 0.5, chan=2))
seq.append(Sinc(100, 0.5, chan=1))
seq.append(Triangle(100, 0.5, chan=2))

r = AmplitudeRotation(Triangle, 125, 0.5, chans=(1,2), drag=0.1)
seq.append(r(np.pi, 0))

s = Sequencer(seq)
seqs = s.render()
s.plot_seqs(seqs)
s.print_seqs(seqs)

