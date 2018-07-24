# Example for a T2 echo measurement sequence using pulseseq
# Includes single side band modulation
# Reinier Heeres, 2013

from pulseseq.sequencer import *
from pulseseq.pulselib import *
import numpy as np

# Create a rotation generator with a pi-area of 100 on channels 1 and 2
# Gaussian sigma between 5 and 10 (a constant in center to get total area)
# Amplitude between 0 and 0.5
r = GSRotation(100, 5, 10, 0, 0.5, chans=(1,2))

# Single sideband modulation: period 20, channel pair (1,2), phases (0, -pi/2)
ssb = SSB(20, (1,2), 0)

s = Sequence()

# Create a sequence with these delays
delays = np.linspace(0, 10000, 41)
for d in delays:
    s.append(Trigger(250))      # Wait for a trigger
    s.append(r(np.pi/2, 0))     # Rotate pi/2 around X
    s.append(Delay(d/2))
    s.append(r(np.pi, np.pi/2)) # Echo: rotate pi aournd Y
    s.append(Delay(d/2))
    s.append(r(np.pi/2, 0))     # Rotate pi/2 aournd X

# Render sequences using sequencer
s = Sequencer(s)
seqs = s.render()

# Apply SSB
ssb.modulate(seqs)

# And plot
s.plot_seqs(seqs)
s.print_seqs(seqs)

