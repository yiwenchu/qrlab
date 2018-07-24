# Stress test for the Repeat pulse.
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()
seq.append(Constant(250, 0.123, chan=3))
seq.append(Repeat(Constant(250, 0.9, chan=1), 5))
# The instruction below should work but could cause problems!
seq.append(Combined([
    Repeat(Constant(250, 0.8, chan=1), 4),
    Repeat(Constant(250, 0.7, chan=2), 4)
]))
seq.append(Repeat(Combined([
    Constant(250, 0.6, chan=1),
    Constant(250, 0.5, chan=2)
]), 3))

s = Sequencer(seq)
seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

