# Stress test for the Combined pulse.
# Reinier Heeres, 2013

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *

r = GSRotation(40, 5, 10, 0, 1.0, chans=(1,2))

seq = Sequence()
seq.append(Combined([
    Constant(200, 0.9, chan=1),
    Pad(Constant(150, 0.9, chan=2), 250, pad=PAD_RIGHT),
]))
seq.append(Combined([
    Combined([
        Constant(200, 0.9, chan=1),
        Pad(Constant(150, 0.7, chan=2), 275, pad=PAD_BOTH),
    ], align=ALIGN_LEFT),
    Constant(100, 0.5, chan=3),
], align=ALIGN_RIGHT))
seq.append(Combined([
    r(np.pi, 0),
    Constant(100, 0.5, chan=3),
], align=ALIGN_CENTER))

s = Sequencer(seq)
seqs = s.render(debug=True)
s.plot_seqs(seqs)
s.print_seqs(seqs)

