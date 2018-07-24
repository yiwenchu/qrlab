import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
import fpga_sequences
from YngwieEncoding import *

REPRATE_DELAY = 50000      # Delay to get a reasonable repetition rate

m = fpgameasurement.FPGAMeasurement('cool', rrec=False)

s = sequencer.Sequence()

s.append(sequencer.Constant(512, 1, value=1, chan='m0'))
s.append(fpga_sequences.qubit_cool(m, label='qcool', tgtlabel='measure'))
s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), master_integrate=m.integrate_log, label='measure'))
s.append(fpgapulses.LongMeasurementPulse())
s.append(sequencer.Delay(REPRATE_DELAY, jump='qcool'))

m.set_seq(s, 10)
#m.generate(plot=True)
m.start_exp()
m.plot_histogram()
