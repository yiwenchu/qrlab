import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D
import lmfit
from lib.math import fitter

'''
Measuring Chi in the non-number-resolved regime, a la Zaki and Steven
'''

def analysis(meas, data=None, fig=None, detune=None):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
    ys, meas_fig = meas.get_ys_fig(data, fig)
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig
    if detune is None:
        detune = meas.detune/1e3 # in kHz and always positive?

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay_sine')
    p = f.get_lmfit_parameters(xs, ys)
    if meas.fix_freq:
        p['f'].value = meas.detune/1e6
        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, p=p, print_report=True, plot=False)
    ys_fit = f.test_values(xs, noise_amp=0.0, p=result.params)

    f = result.params['f'].value * 1e3
    tau = result.params['tau'].value

    txt = 'tau = %0.3f us\n' % (tau)
    txt += 'f = %0.4f kHz\n' % (f)
    txt += 'software df = %0.3f kHz\n' % (detune)
    txt += 'frequency detuning = %0.4f kHz' % (detune-f)



    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys)/ys_fit * 100.0, marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Fit error [%]')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params

class UnresolvedChi(Measurement2D):
    def __init__(self, qubit_info, cavity_info, delays, nbars,
                 seq=None, postseq=None, subtraction=False,
                 detune=0.0,
                 **kwargs):

        self.qubit_info = qubit_info
        self.cavity_info = cavity_info

        self.delays = np.array(np.round(delays), dtype=int)
        self.nbars = np.array(nbars)
        self.detune = detune

        self.xs = delays / 1e3        # For plotting purposes
        self.ys = nbars

        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.subtraction = subtraction

        numpts = len(nbars)*len(delays)
        if subtraction:
            numpts *= 2

        super(UnresolvedChi, self).__init__(numpts,
                  infos=(qubit_info, cavity_info), **kwargs)

        self.data.create_dataset('delays', data=delays)
        self.data.create_dataset('nbars', data=nbars)

        self.data.set_attrs(
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        s = Sequence()

        c = self.cavity_info.rotate
        r = self.qubit_info.rotate

        phases = [0]
        if self.subtraction:
            phases = [0,np.pi]

        for nbar in self.nbars:
            for i, dt in enumerate(self.delays):
                for p in phases:

                    s.append(self.seq)

                    s.append(c(np.pi*np.sqrt(nbar),0))
                    s.append(Delay(50))

                    s.append(r(np.pi/2, 0))
                    s.append(Delay(dt))
                    phase = 2 * np.pi * dt * self.detune * 1e-9
                    s.append(r(np.pi/2, p + phase))

                    if self.postseq is not None:
                        s.append(self.postseq)

                    s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(UnresolvedChi, self).get_ys(data)
        if self.subtraction:
            return ys[::2] - ys[1::2]
        return ys

    def get_all_ys(self, data=None):
        ys = super(UnresolvedChi, self).get_ys(data)
        if self.subtraction:
            return ys[0::2], ys[1::2]
        else:
            return ys

    def analyze(self, data=None, fig=None):
        return 0
#        self.fit_params = analysis(self, data, fig)
#        return self.fit_params