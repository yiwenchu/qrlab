import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
import YngwieEncoding as ye

N = 81                      # Number of elements in sequence
DT = 750
ECHO = False
DELTA = 0
delays = np.arange(N) * DT
REPRATE_DELAY = 500000      # Delay to get a reasonable repetition rate

if DELTA != 0:
    fitf = 'exp_decay_sine'
else:
    fitf = 'exp_decay' if ECHO else 'exp_decay_sine'

m = fpgameasurement.FPGAMeasurement('T2', xs=delays/1000., fit_func=fitf)

s = sequencer.Sequence()

# FPGA style, using R13 as dynamic instruction length,
# doesn't include software detuning though.
# There's a bug in pulseseq which requires the extra label _1/_2: we need
# to make sure that the generated delays are not joined.
if 0:
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(13, 0), master_counter0=N, label='reset'))

    s.append(sequencer.Constant(512, 1, value=1, chan='m0', label='start'))
    s.append(m.qubit.rotate(np.pi/2, 0, label='_1'))
    s.append(sequencer.Delay(100, inslength=('R13', 10)))
    s.append(m.qubit.rotate(np.pi/2, 0, label='_2'))
    s.append(fpgapulses.LongMeasurementPulse(label='measure'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.ADDI(13, DT/4), master_counter0=-1, master_internal_function=ye.FPGASignals.c0))
    s.append(sequencer.Delay(REPRATE_DELAY, jump=('reset', 'start')))

# Old-school
else:
#    m.qubit.rotate = fpgapulses.FPGAMarkerLenRotation(72, 'm0', 1)
    s.append(sequencer.Delay(256, label='start'))
    for curdelay in delays:
        s.append(m.qubit.rotate(np.pi/2, 0))
        if ECHO:
            if curdelay != 0:
                s.append(sequencer.Delay(curdelay/2))
            s.append(m.qubit.rotate(np.pi, np.pi))
            if curdelay != 0:
                s.append(sequencer.Delay(curdelay/2))
        else:
            if curdelay != 0:
                s.append(sequencer.Delay(curdelay))
        if DELTA == 0:
            phi = 0
        else:
            phi = DELTA * (curdelay * 1e-9) * 2 * np.pi
        s.append(m.qubit.rotate(np.pi/2, phi))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(sequencer.Delay(REPRATE_DELAY - curdelay))
    s.append(sequencer.Delay(256, jump='start'))

m.set_seq(s, len(delays))
m.start_exp()
#m.plot_histogram()
m.plot_se()
t2type = 'e' if ECHO else 'r'
title = 'T2%s = %.01f +- %.01f us' % (t2type, m.fit_params['tau'].value, m.fit_params['tau'].stderr)
if 'f' in m.fit_params:
    title += ', delta = %.01f kHz' % (m.fit_params['f'].value*1000,)
plt.suptitle(title)
m.save_fig()