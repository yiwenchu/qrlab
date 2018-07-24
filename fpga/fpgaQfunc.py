import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
from YngwieEncoding import *

DMAX = 2.0
BG = False
NI = 21
NQ = 21
m = fpgameasurement.FPGAMeasurement('Qfunc', xs=np.linspace(-DMAX,DMAX,NQ), ys=np.linspace(-DMAX,DMAX,NI))
stateprep = m.cavity.displace(1.0*np.exp(1j*0*np.pi/4))

REPRATE_DELAY = 2000000      # Delay to get a reasonable repetition rate

s = sequencer.Sequence()

I0 = DMAX * m.cavity.rotate.pi_amp * 65535
Q0 = I0
DI = round(2.0*I0/(NI - 1))
DQ = round(2.0*Q0/(NQ - 1))

s.append(sequencer.Constant(512, 1, value=1, chan='m0'))

s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(0, +I0), label='Ireset'))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(3, +I0), master_counter0=NI))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(2, +Q0), label='Qreset'))
s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(1, -Q0), master_counter1=NQ))

s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), label='displace'))
s.append(stateprep)
s.append(m.cavity.displace(1.0, **m.cavity.loaduse_mixer_kwargs))
s.append(m.qubit.rotate_selective(np.pi, 0))
s.append(fpgapulses.LongMeasurementPulse(label='measure'))
s.append(sequencer.Delay(REPRATE_DELAY))
if BG:
    s.append(m.cavity.displace(D0))
    s.append(m.cavity.displace(1.0, **m.cavity.loaduse_mixer_kwargs))
    s.append(m.qubit.rotate(np.pi, 0))
    s.append(fpgapulses.LongMeasurementPulse(label='measure'))
    s.append(sequencer.Delay(REPRATE_DELAY))

s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(2, -DQ), label='Qstep', master_counter1=-1))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(1, +DQ), master_internal_function=FPGASignals.c1, jump=('next', 'displace')))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(0, -DI), label='Istep', master_counter0=-1))
s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(3, -DI), master_internal_function=FPGASignals.c0, jump=('Ireset', 'Qreset')))
   
m.set_seq(s, NI*NQ)
m.start_exp()
m.plot2d(swapax=True)