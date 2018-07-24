import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import matplotlib.pyplot as plt
import copy
import mclient
import lmfit

FIT_AMP         = 'AMP'         # Fit simple sine wave
FIT_AMPFUNC     = 'AMPFUNC'     # Try to fit amplitude curve based on pi/2 and pi amp
FIT_PARABOLA    = 'PARABOLA'    # Fit a parabola (to determine min/max pos)

def fit_amprabi(params, x, data):
    est = params['ofs'].value - params['amp'].value * np.cos(2*np.pi*x / params['period'].value + params['phase'].value)
    return data  - est

def fit_amprabi_func(params, x, data, meas):
    coeffs = np.polyfit([0, params['pi2_amp'].value, params['pi_amp'].value], [0, np.pi/2, np.pi], 2)
    phases = (x**2*coeffs[0] + x*coeffs[1] + coeffs[0]) * meas.repeat_pulse
    est = params['ofs'].value - params['amp'].value * np.cos(phases)
    return data  - est

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.amps

    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    amp0 = (np.max(ys) - np.min(ys)) / 2
    if ys[0]>np.average(ys):
        amp0 = -amp0
    fftys = np.abs(np.fft.fft(ys - np.average(ys)))
    fftfs = np.fft.fftfreq(len(ys), xs[1]-xs[0])
    period0 = 1 / np.abs(fftfs[np.argmax(fftys)])

    params = lmfit.Parameters()
    params.add('ofs', value=np.average(ys))
    params.add('amp', value=amp0)
    params.add('phase', value=0, min=-np.pi, max=np.pi, vary=(not meas.fix_phase))

    if meas.fit_type == FIT_AMPFUNC:
        pi_amp = period0 * meas.repeat_pulse / 2
        params.add('pi_amp', value=pi_amp)
        params.add('pi2_amp', value=0.5*pi_amp)
        result = lmfit.minimize(fit_amprabi_func, params, args=(xs, ys, meas))
        txt = ''
        fig.axes[0].plot(xs, -fit_amprabi_func(params, xs, 0, meas), label='fit')
        fig.axes[1].plot(xs, fit_amprabi_func(params, xs, ys, meas), marker='s')

    else:
        if meas.fix_period is not None:
            params.add('period', value=meas.fix_period, vary=False)
        else:
            params.add('period', value=period0, min=0)
        result = lmfit.minimize(fit_amprabi, params, args=(xs, ys))
        txt = 'Amp = %.03f +- %.03e\nPeriod = %.03f +- %.03e\nPi amp = %.06f' % (params['amp'].value, params['amp'].stderr, params['period'].value, params['period'].stderr, params['period'].value/2 )
        fig.axes[0].plot(xs, -fit_amprabi(params, xs, 0), label=txt)
        fig.axes[1].plot(xs, fit_amprabi(params, xs, ys), marker='s')

    lmfit.report_fit(params)

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Pulse amplitude')
    fig.axes[0].legend(loc=0)

    fig.canvas.draw()
    return params

class Rabi(Measurement1D):

    def __init__(self, qubit_info, amps, update=False, seq=None, r_axis=0, fix_phase=True,
                 fix_period=None, repeat_pulse=1, postseq=None, selective=False, fit_type=FIT_AMP, **kwargs):
        self.qubit_info = qubit_info
        self.amps = amps
        self.xs = amps
        self.update_ins = update
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.fix_phase = fix_phase
        self.fix_period = fix_period
        self.repeat_pulse = repeat_pulse
        self.r_axis = r_axis
        self.fit_type = fit_type
        self.selective = selective

        super(Rabi, self).__init__(len(amps), infos=qubit_info, **kwargs)
        self.data.create_dataset('amps', data=amps)

    def generate(self):
        s = Sequence()

        for i, amp in enumerate(self.amps):
            s.append(self.seq)
            if self.selective:
                s.append(Repeat(self.qubit_info.rotate_selective(0, self.r_axis, amp=amp), self.repeat_pulse))
            else:
                s.append(Repeat(self.qubit_info.rotate(0, self.r_axis, amp=amp), self.repeat_pulse))
            if self.postseq is not None:
                s.append(self.postseq)
            s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        if self.fit_type == FIT_PARABOLA:
            return self.analyze_parabola(data=data, fig=fig, xlabel='Amplitude', ylabel='Signal')

        self.fit_params = analysis(self, data=data, fig=fig)
        if self.fit_type == FIT_AMPFUNC:
            pi_amp = self.fit_params['pi_amp'].value
            pi2_amp = self.fit_params['pi2_amp'].value
            if self.update_ins:
                print 'Setting qubit pi-rotation ampltidue to %.06f, pi/2 to %.06f' % (pi_amp, pi2_amp)
                mclient.instruments[self.qubit_info.insname].set_pi_amp(pi_amp)
                mclient.instruments[self.qubit_info.insname].set_pi2_amp(pi2_amp)
        else:
            pi_amp = self.fit_params['period'].value / 2 * self.repeat_pulse
            if self.update_ins:
                print 'Setting qubit pi-rotation ampltidue to %.06f' % pi_amp
                mclient.instruments[self.qubit_info.insname].set_pi_amp(pi_amp)

        return pi_amp
