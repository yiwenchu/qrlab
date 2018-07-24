import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from scripts.single_qubit import T2measurement as t2m
import lmfit
from lib.math import fitter

def analysis(meas, data=None, fig=None, detune=None):
#    ys = meas.get_ys(data, fig)
    ys = meas.get_ys(data)
    fig = meas.get_figure()

    result_pi = t2m.analysis(meas, data=ys[::2], fig=fig, detune=None,)
    result_nopi = t2m.analysis(meas, data=ys[1::2], fig=fig, detune=None,)

    return result_pi, result_nopi

class ZetaRamsey(Measurement1D):

    def __init__(self, ramsey_qubit, pi_qubit, delays,
                 detune=0, double_freq=False, fix_freq=None,
                 seq=None, postseq=None, **kwargs):

        self.ramsey_qubit = ramsey_qubit
        self.pi_qubit = pi_qubit

        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3        # For plotting purposes
        self.detune = detune

        self.double_freq=double_freq
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(ZetaRamsey, self).__init__(2*len(delays), infos=(ramsey_qubit, pi_qubit), **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(
            detune=detune,
            rasmey_qubit=self.ramsey_qubit.insname,
            pi_qubit=self.pi_qubit.insname,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        s = Sequence()
        r = self.ramsey_qubit.rotate
        p = self.pi_qubit.rotate

        for i, dt in enumerate(self.delays):
            for do_pi in [1, 0]:
                s.append(self.seq)

                s.append(p(do_pi*np.pi, X_AXIS))

                s.append(r(np.pi/2, X_AXIS))
                s.append(Delay(dt))

                angle = dt * 1e-9 * self.detune * 2 * np.pi + X_AXIS
                s.append(r(np.pi/2, angle))

                s.append(p(do_pi*np.pi, X_AXIS))

                if self.postseq is not None:
                    s.append(self.postseq)

                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def update(self, avg_data):
        ys = self.get_ys(avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()
        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, ys[::2])
            fig.axes[0].plot(self.xs, ys[1::2])
        else:
            fig.axes[0].plot(ys[::2])
            fig.axes[0].plot(ys[1::2])
        fig.canvas.draw()

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params
