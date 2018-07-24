# Simple script to setup a readout sequence.
# Call m.start_exp() after running this to start FPGA

from pulseseq import sequencer, pulselib
import fpgapulses
import fpgameasurement

REPRATE_DELAY = 250000      # Delay to get a reasonable repetition rate
m = fpgameasurement.FPGAMeasurement('RO')

s = sequencer.Sequence()
s.append(sequencer.Delay(200, label='start'))
#s.append(m.qubit.rotate(np.pi,0))
s.append(fpgapulses.LongMeasurementPulse(label='measure'))
s.append(sequencer.Delay(REPRATE_DELAY, jump='start'))

m.set_seq(s, 1)
m.generate()