import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import lmfit
from lib.math import fitter


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

    if not meas.double_freq:
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

    else:
        f = fitter.Fitter('exp_decay_double_sine')
        result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
        ys_fit = f.test_values(xs, noise_amp=0.0, p=result.params)

        f = result.params['f'].value
        f1 = result.params['f1'].value
        tau = result.params['tau'].value

        txt = 'tau = %0.3f us\n' % (tau)
        txt += 'f = %0.4f kHz\n' % (f)
        txt += 'f1 = %0.4f kHz\n' % (f1)
        txt += 'software df = %0.3f kHz\n' % (detune)


    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys)/ys_fit * 100.0, marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Fit error [%]')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params, detune-f

class StarkRamsey(Measurement1D):

    def __init__(self, qubit_info, stark_info, delays, stark_len=50e3,
                 stark_amp =1.0, detune=0, double_freq=False,
                 fix_freq=None, seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.stark_info = stark_info
        self.stark_len = stark_len
        self.stark_amp = stark_amp
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3        # For plotting purposes
        self.detune = detune
        self.double_freq=double_freq
        self.fix_freq = fix_freq
        #if seq is None:
        #    seq = Trigger(250)
        #self.seq = seq
        self.postseq = postseq

        super(StarkRamsey, self).__init__(len(delays),
                      infos=[qubit_info, stark_info], **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(
            detune=detune,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate
        stark_r = self.stark_info.rotate
        stark_ch = self.stark_info.sideband_channels

        stark_pulse = lambda length: Constant(length, self.stark_amp,
                                              chan=stark_ch[0])


        plen = r(np.pi,0).get_length()

        for i, dt in enumerate(self.delays):
            s.append(Trigger(250))

            s.append(stark_pulse(self.stark_len))

            s.append(Combined([r(np.pi/2, X_AXIS),
                               stark_pulse(plen)]))

            if dt > 0:
                s.append(Combined([Delay(dt),
                                   stark_pulse(dt)]))

            # Measurement pulse
            angle = dt * 1e-9 * self.detune * 2 * np.pi
            s.append(Combined([r(np.pi/2, angle),
                               stark_pulse(plen)]))

            s.append(Delay(20))
            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params, detuning = analysis(self, data, fig)
        return self.fit_params, detuning