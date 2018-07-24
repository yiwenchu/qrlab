# Example to show various alignment options using pulseseq
# Reinier Heeres, 2013

from pulseseq.sequencer import *
from pulseseq.pulselib import *
import numpy as np

# Create some Gaussian pulses, with different sigma and amplitude
g1 = Gaussian(50, 0.7, chan=1)
g2 = Square(75, 0.5, chan=2)
g3 = Gaussian(130, 0.3, chan=3)

s = Sequence()

# To play the pulses simultaneously we use the Combined class.
# It takes a list of pulses and the alignment strategy
s.append(Combined([g1, g2], align=ALIGN_LEFT))
s.append(Combined([g1, g2], align=ALIGN_RIGHT))
s.append(Combined([g1, g2], align=ALIGN_CENTER))

# If you want to have at least length 500, you can add a Constant(0)
# (Delay should be ok too in the future but is broken atm)
s.append(Combined([g1, g2, Constant(500, 0.1, chan=0)], align=ALIGN_RIGHT))

# You should try to break up the overall sequence in different blocks that
# you want to align in a certain way.
subseq = Combined([g1, g2], align=ALIGN_CENTER)
s.append(Combined([subseq, g3], align=ALIGN_RIGHT))

# Render sequences using sequencer
s = Sequencer(s)
seqs = s.render(debug=True)

# And plot
s.plot_seqs(seqs)
s.print_seqs(seqs)
