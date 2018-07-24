# Tune up read-out with FPGA, should set things up so that a value < threshold
# means |g> and a value > threshold corresponds to |e>

import numpy as np
from pulseseq import sequencer, pulselib
import time
import matplotlib.pyplot as plt
import mclient
from lib.math import fitter
import os
import config

ENV_FILE = os.path.join(config.datadir, 'fpga_envelope.npz')
SAVE_ENV = True
ROTATION = True
ROTATION_REPS = 5   # nshots = R_REPS * naverages
REPRATE_DELAY = 500000      # Delay to get a reasonable repetition rate

mclient.instruments['readout'].set_envelope('1')
execfile('fpgarunro.py')
m.iqrel = True
ret = m.start_exp()
m.stop_exp()

SE, IQ = ret

IQavg = np.average(IQ, 0)
ts = np.arange(len(IQavg))*0.020
f = plt.figure()
plt.suptitle('Avg shot')
ax1 = f.add_subplot(211)
ax1.plot(ts, np.real(IQavg), label='Re')
ax1.plot(ts, np.imag(IQavg), label='Im')
ax1.plot(ts, np.abs(IQavg), label='Abs')
ax1.plot(ts, np.abs(IQavg)**2, label='Pwr')
ax1.set_xlabel('Time [usec]')
ax1.set_ylabel('Signal [mV]')
ax2 = f.add_subplot(212)
ax2.plot(np.real(IQavg), np.imag(IQavg))

fit = fitter.Fitter('exp_decay')
imax = np.argmax(np.abs(IQavg)) + 10
ret = fit.perform_lmfit(ts[imax:], np.abs(IQavg[imax:]))
ax1.plot(ts[imax:], fit.eval_func(ts[imax:]), label='tau = %.01f +- %.01f nsec' % (ret.params['tau'].value*1000, ret.params['tau'].stderr*1000))
ax1.legend()

envelope = IQavg.conj()
envelope *= (1.99 / np.max(np.abs(envelope)))
if SAVE_ENV:
    np.savez(ENV_FILE, envelope=envelope)
mclient.instruments['readout'].set_envelope(ENV_FILE)

if ROTATION:
    IQpoints = []

    for i in 0,1:
        m = fpgameasurement.FPGAMeasurement('pi%d'%i)

        s = sequencer.Sequence()
        s.append(sequencer.Delay(256, label='start'))
        s.append(m.qubit.rotate(np.pi*i, 0))
        s.append(fpgapulses.LongMeasurementPulse())
        s.append(sequencer.Delay(REPRATE_DELAY))
        s.append(sequencer.Delay(256, jump='start'))
        m.set_seq(s, ROTATION_REPS)
        m.start_exp()

        # Determine correction angle
        params = m.analyze_histogram_blob(plot=True)
        IQpoints.append(params['x0'].value + 1j * params['y0'].value)

    alpha = np.angle(IQpoints[1] - IQpoints[0])

    # Determine threshold
    IQpoints[0] *= np.exp(-1j * alpha)
    IQpoints[1] *= np.exp(-1j * alpha)
    thresh = (np.real(IQpoints[0]) + np.real(IQpoints[1])) / 2

    envelope0 = np.cos(alpha) * envelope + np.sin(alpha) * 1j * envelope
    envelope1 = -np.sin(alpha) * envelope + np.cos(alpha) * 1j * envelope
    if SAVE_ENV:
        np.savez(ENV_FILE, envelope0=envelope0, envelope1=envelope1, thresh0=thresh, thresh1=0)

I = np.dot(envelope, IQavg)
Q = np.dot(-1j * envelope, IQavg)
