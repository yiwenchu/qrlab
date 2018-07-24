import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgapulses
import fpgameasurement
import YngwieEncoding as ye

N = 101                   # Number of elements in sequence
REPRATE_DELAY = 1200000
CAVPULSE = True
DISP = np.sqrt(2)
dfs = np.linspace(-14e6, 2e6, N)
mask = (dfs != 0)
periods = np.zeros_like(dfs)
periods[mask] = 1e9 / dfs[mask]
m = fpgameasurement.FPGAMeasurement('SSBspec', xs=dfs/1e6, fit_func='gaussian', rrec=False)

s = sequencer.Sequence()

# Old-style
if 0:
    s.append(sequencer.Delay(240, label='start'))
    for period in periods:
        if CAVPULSE:
    #        s.append(sequencer.Constant(500, 1, chan='m0'))
            s.append(m.cavity.displace(DISP))
        s.append(m.qubit.rotate(np.pi, 0, detune_period=period))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(sequencer.Delay(REPRATE_DELAY))
    s.append(sequencer.Delay(240, jump='start'))

# New style, in logic a13 and up
else:
    F0 = 50000 - round(dfs[0]/1000)     # F0 in kHz
    DF = round(-(dfs[1]-dfs[0])/1000)   # DF in hHz
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(12, F0), master_counter0=N, label='reset'))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.MOVI(0, F0)))
    s.append(sequencer.Delay(1000, label='start', **m.qubit.load_ssb_kwargs))
    if CAVPULSE:
        s.append(m.cavity.displace(DISP))
    s.append(m.qubit.rotate(np.pi, 0))
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(sequencer.Delay(REPRATE_DELAY))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.ADDI(0, DF), length=100))
    s.append(fpgapulses.RegOp(ye.RegisterInstruction.ADDI(12, DF), length=100, master_counter0=-1, master_internal_function=ye.FPGASignals.c0, jump=('reset', 'start')))

m.set_seq(s, len(dfs))
#m.generate(plot=True)
m.start_exp()
m.plot_se()
plt.xlabel('Detuning [MHz]')
plt.suptitle('SSB Spec')
m.save_fig()