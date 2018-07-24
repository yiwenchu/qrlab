# Example for marker bit channels
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

seq = Sequence()

# If we would like to use channel 1 markers we *must* have some content on the
# channel itself as well.
seq.append(Constant(250, 0.5, chan=1))

# A channel <chan>m<i> will be interpreted as marker <i> for channel <chan> by
# the AWG loading code.
seq.append(Combined([
    Constant(1000, 1, chan='1m1'),
    Constant(1000, 1, chan='1m2'),
]))

s = Sequencer(seq)
seqs = s.render()
s.plot_seqs(seqs)
s.print_seqs(seqs)

