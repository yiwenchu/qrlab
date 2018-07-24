# Example of how combine several logical channels into one physical channel
# (useful for several qubits which are close to each other in frequency).
# Reinier Heeres, 2013

from pulseseq.sequencer import *
from pulseseq.pulselib import *
import numpy as np

# Get rotation generator into logical channel pairs (a1,a2) and (b1,b2)
r1 = GSRotation(50, 20, 40, 0, 0.4, chans=('a1', 'a2'))
r2 = GSRotation(60, 20, 40, 0, 0.4, chans=('b1', 'b2'))

# Get single side band modulators which BOTH output into physical channel
# pair (1,2). We specify replace=False to not replace the content in (1,2),
# but add to it.
ssb1 = SSB(20, ('a1', 'a2'), 0, outchans=(1,2), replace=False)
ssb2 = SSB(20, ('b1', 'b2'), 0, outchans=(1,2), replace=False)

# Define the sequence
s = Sequence()
s.append(r1(np.pi/2, 0))
s.append(r2(np.pi, np.pi/2))

# Combine a pulse in both logical channels simultaneously
s.append(Combined([r1(np.pi/2,0), r2(np.pi, np.pi/2)], align=ALIGN_CENTER))

# Render the sequence
s = Sequencer(s)
seqs = s.render(s)

# Perform SSB, combining the channels into (1,2)
ssb1.modulate(seqs)
ssb2.modulate(seqs)

# And plot
s.plot_seqs(seqs)
s.print_seqs(seqs)

