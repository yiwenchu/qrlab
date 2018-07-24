# Example to use the master/slave feature
# Reinier Heeres, 2014

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()
seq.append(Trigger(250))
seq.append(Gaussian(50, 0.5, chan=1))
seq.append(Gaussian(100, 0.5, chan=2))
seq.append(Gaussian(20, 0.5, chan=5))
seq.append(Trigger(250))
seq.append(Gaussian(100, 0.5, chan=1))
seq.append(Gaussian(20, 0.5, chan=2))
seq.append(Gaussian(50, 0.5, chan=5))

s = Sequencer(seq)
s.add_master_channel([1,2])
s.add_slave_trigger('1m1', 200)
s.add_slave_trigger('1m2', 100)

seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

