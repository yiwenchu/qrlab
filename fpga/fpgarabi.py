import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
from YngwieEncoding import *

N = 41
UPDATE = True
SELECTIVE = False

if SELECTIVE:
    MAXAMP = 0.1
else:
    MAXAMP = 1.0
amps = np.linspace(0, MAXAMP, N)

# After tuning up single-shot histogramming
m = fpgameasurement.FPGAMeasurement(
    'rabi',
    xs=amps, fit_func='sine', fit_func_kwargs=dict(dphi=-np.pi/2),
    probabilities=True)

COOL = False

# For initial tune-up, without probabilities
#m = fpgameasurement.FPGAMeasurement('rabi', xs=amps, fit_func='sine', probabilities=False)
#COOL = False

REPRATE_DELAY = 400000      # Delay to get a reasonable repetition rate
PULSE_REP = 1

s = sequencer.Sequence()

# Old-fashioned method
if 0:
    s.append(sequencer.Delay(256, label='start'))
    m.qubit.rotate.set_pi_amp(1)
    for amp in amps:
        for j in range(PULSE_REP):
            s.append(m.qubit.rotate(np.pi*amp, 0))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(sequencer.Delay(REPRATE_DELAY))
    s.append(sequencer.Delay(256, jump='start'))

# FPGA style
else:
    DA = round((MAXAMP / (N - 1)) * 65535)
    s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(0, 0), master_counter0=N, label='reset'))
    s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(1, 0)))
    s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(2, 0)))
    s.append(fpgapulses.RegOp(RegisterInstruction.MOVI(3, 0)))

    s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), label='start'))
    if COOL:
        s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), master_integrate=m.integrate_nolog, label='cool'))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), length=220, master_internal_function=FPGASignals.s0, jump=('next', 'measure')))
        s.append(m.qubit.rotate(np.pi,0,jump='cool'))

    s.append(fpgapulses.RegOp(RegisterInstruction.NOP(), master_integrate=m.integrate_log, label='measure'))
    if SELECTIVE:
        qubitrotation = m.qubit.rotate_selective(1.0, 0, **m.qubit.loaduse_mixer_kwargs)
    else:
        qubitrotation = m.qubit.rotate(1.0, 0, **m.qubit.loaduse_mixer_kwargs)
    s.append(qubitrotation)
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(0, DA), master_counter0=-1, master_internal_function=FPGASignals.c0))
    if not COOL:
        s.append(sequencer.Delay(REPRATE_DELAY))
    else:
        s.append(sequencer.Delay(50000))
    s.append(fpgapulses.RegOp(RegisterInstruction.ADDI(3, DA), jump=('reset', 'start')))

m.set_seq(s, len(amps))
m.start_exp()
#m.plot_histogram(bins='log')
m.plot_se()
d = m.get_averaged()
plt.suptitle('pi amp = %.06f +- %.06f, phi = %.01f deg\nmin = %.03f, max = %.03f' % (0.5/m.fit_params['f'].value, 0.5/m.fit_params['f'].value**2*m.fit_params['f'].stderr, np.rad2deg(m.fit_params['dphi']-np.pi/2), np.min(d), np.max(d)))
m.save_fig()

if UPDATE:
    piamp = 0.5/m.fit_params['f'].value
    u_piamp = 0.5/m.fit_params['f'].value**2*m.fit_params['f'].stderr
    if u_piamp < 0.1 * piamp:
        if SELECTIVE:
            mclient.instruments[m.qubit.insname].set_pi_amp_selective(piamp)
        else:
            mclient.instruments[m.qubit.insname].set_pi_amp(piamp)
    else:
        print("Returned a bad fit. I did not update pi pulse")
