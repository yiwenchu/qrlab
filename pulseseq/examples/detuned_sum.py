# Example for the 'sum of detuned pulses' generator
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

tsum = DetunedSum(Triangle, 100)
tsum.add(0.1, 1e12)
tsum.add(0.2, 40)
tsum.add(0.2, 20)
tsum.add(0.2, 10)

gsum = DetunedSum(Gaussian, 50)
gsum.add(0.1, 1e12)
gsum.add(0.2, 40)
gsum.add(0.2, 20)
gsum.add(0.2, 10)

seq = Sequence()
seq.append([
    Combined([tsum(), Triangle(100, 1, chan=3)]),
    Delay(100),
    Combined([gsum(), Gaussian(50, 1, chan=3)]),
])

s = Sequencer(seq)
seqs = s.render()
s.plot_seqs(seqs)
s.print_seqs(seqs)

