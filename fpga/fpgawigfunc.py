import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
from YngwieEncoding import *

DMAX = 2
BG = 1
NI = 21
NQ = 21
m = fpgameasurement.FPGAMeasurement('Wfunc', xs=np.linspace(-DMAX,DMAX,NQ), ys=np.linspace(-DMAX,DMAX,NI), qubit_gaussian_chop=5, bg=BG)
stateprep = m.cavity.displace(1*np.exp(1j*np.pi/4))
wigevens = fpgapulses.FPGADetunedSum(m.qubit.rotate.Ipulse, periods=qubit_periods[0::2], amps=qubit_amps[0::2])
wigodds = fpgapulses.FPGADetunedSum(m.qubit.rotate.Ipulse, periods=qubit_periods[1::2], amps=qubit_amps[1::2])

REPRATE_DELAY = 1000000      # Delay to get a reasonable repetition rate

s = sequencer.Sequence()

I0 = DMAX * m.cavity.rotate.pi_amp * 65535
Q0 = I0
DI = round(2.0*I0/(NI - 1))
DQ = round(2.0*Q0/(NQ - 1))

s.append(sequencer.Constant(512, 1, value=1, chan='m0'))

s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(0, +I0), label='Ireset'))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(3, +I0), fpga_counter0=NI))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(2, +Q0), label='Qreset'))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(1, -Q0), fpga_counter1=NQ))

s.append(sequencer.Constant(256, 1, chan=0, label='displace'))
s.append(stateprep)
s.append(m.cavity.displace(1.0, fpga_mixer_mask=(True, True)))
s.append(wigodds(np.pi, 0))
s.append(fpgapulses.LongMeasurementPulse(label='measure'))
s.append(sequencer.Delay(REPRATE_DELAY))
if BG:
    s.append(stateprep)
    s.append(m.cavity.displace(1.0, fpga_mixer_mask=(True, True)))
    s.append(wigevens(np.pi,0))
    s.append(fpgapulses.LongMeasurementPulse(label='measure'))
    s.append(sequencer.Delay(REPRATE_DELAY))

s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(2, -DQ), label='Qstep', fpga_counter1=-1))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(1, +DQ), fpga_internal_function=FPGASignals.c1, jump=('next', 'displace')))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(0, -DI), label='Istep', fpga_counter0=-1))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(3, -DI), fpga_internal_function=FPGASignals.c0, jump=('Ireset', 'Qreset')))

m.set_seq(s, NI*NQ*(BG+1))
m.start_exp()
m.plot2d(swapax=True)