# Stress test for the Repeat pulse.
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import cProfile

seq = Sequence()
seq.append(Join([
    Trigger(250),
    Constant(100, 0),
]))
seq.append(Gaussian(5,0.5,chan=1))
seq.append(Repeat(Gaussian(10,0.7,chan=1), 50))
seq.append(Gaussian(5,0.5,chan=1))
s = Sequencer(seq)
cProfile.run('seqs = s.render(debug=True)')
#seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

