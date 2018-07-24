# Stress test for the minimum pulse length feature
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()
seq.append(Constant(30, 0.123, chan=3))
seq.append(Repeat(Constant(60, 0.9, chan=1), 50))
seq.append(Combined([
    Repeat(Constant(140, 0.8, chan=1), 4),
    Repeat(Constant(140, 0.7, chan=2), 4)
]))
seq.append(Repeat(Combined([
    Constant(130, 0.6, chan=1),
    Constant(130, 0.5, chan=2)
]), 10))
seq.append(Repeat(Combined([
    Constant(125, 0.6, chan=1),
    Constant(125, 0.5, chan=2)
]), 10))
seq.append(Repeat(Combined([
    Constant(124, 0.6, chan=1),
    Constant(124, 0.5, chan=2)
]), 10))

s = Sequencer(seq)
seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

