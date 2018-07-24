# Example for generating a marker channel based on two active channels
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()
seq.append(Trigger(250))
seq.append(Gaussian(100, 0.5, chan=2))
seq.append(Gaussian(100, 0.5, chan=1))
seq.append(Delay(5000))
seq.append(Gaussian(100, 0.5, chan=1))
seq.append(Gaussian(100, 0.2, chan=2))
seq.append(Gaussian(100, 0.3, chan=3))

s = Sequencer(seq)
s.add_required_channel(4)
s.add_marker('1m1', 1, bufwidth=5, ofs=+20)
s.add_marker('1m1', 3)
s.add_marker('4m1', 3)
seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

