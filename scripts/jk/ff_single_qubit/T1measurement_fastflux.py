import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit

def exp_decay(params, x, data):
    est = params['ofs'] + params['amplitude'] * np.exp(-x / params['tau'].value)
    return data - est

def double_exp_decay(params, x, data):
    est = params['ofs'].value + params['amplitude'].value * np.exp(-x / params['tau'].value) + params['amplitude2'].value * np.exp(-x / params['tau2'].value)
    return data - est


def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.delays

    fig.axes[0].plot(xs/1e3, ys, 'ks', ms=3)

    if meas.double_exp == False:
        params = lmfit.Parameters()
        params.add('ofs', value=np.min(ys))
        params.add('amplitude', value=np.max(ys))
        params.add('tau', value=xs[-1]/2.0, min=50.0)
        result = lmfit.minimize(exp_decay, params, args=(xs, ys))
        lmfit.report_fit(params)

        fig.axes[0].plot(xs/1e3, -exp_decay(params, xs, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
        fig.axes[0].legend(loc=0)
        fig.axes[0].set_ylabel('Intensity [AU]')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')

    else:
        params = lmfit.Parameters()
        params.add('ofs', value=np.min(ys))
        params.add('amplitude', value=np.max(ys)/2.0)
        params.add('tau', value=xs[-1], min=50.0)
        params.add('amplitude2', value=np.max(ys)/2.0)
        params.add('tau2', value=xs[-1]/4.0, min=50.0)
        result = lmfit.minimize(double_exp_decay, params, args=(xs, ys))
        lmfit.report_fit(params)

        weight1 = params['amplitude'].value / (params['amplitude'].value + params['amplitude2'].value)*100
        weight2 = 100-weight1
        text = 'Fit, tau = %.03f us +/- %.03f us (%.01f%%)\n     tau2 = %.03f us +/- %.03f us (%.01f%%)'%(
                params['tau'].value/1000.0, params['tau'].stderr/1000.0, weight1, params['tau2'].value/1000.0, params['tau2'].stderr/1000.0, weight2)
        fig.axes[0].plot(xs/1e3, -double_exp_decay(params, xs, 0), label=text)
        fig.axes[0].legend(loc=0)
        fig.axes[0].set_ylabel('Intensity [AU]')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, double_exp_decay(params, xs, ys), marker='s')

    fig.canvas.draw()
    return params

class T1MeasurementFastFlux(Measurement1D):

    def __init__(self, qubit_info, delays, double_exp=False, seq=None,
                 postseq=None, qubit_flux=0, target_flux=0, flux_chan=1,
                 pi_scale=1.0, **kwargs):
        self.qubit_info = qubit_info
        self.delays = delays
        self.xs = delays / 1e3      # For plotting purposes
        self.double_exp = double_exp
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.qubit_flux = qubit_flux
        self.target_flux = target_flux
        self.flux_chan = flux_chan

        self.pi_scale = pi_scale

        super(T1MeasurementFastFlux, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)

    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate
        q_pi = r(np.pi * self.pi_scale, 0)

        flux_pulse = lambda duration, flux_level: Constant(duration, flux_level, chan=self.flux_chan)

        for i, dt in enumerate(self.delays):
            s.append(self.seq)

            s.append(flux_pulse(1000, self.qubit_flux))

            s.append(Combined([q_pi,
               flux_pulse(q_pi.get_length(), self.qubit_flux)
               ]))

            s.append(flux_pulse(50, self.qubit_flux))

            if dt != 0:
                s.append(flux_pulse(dt, self.target_flux))

            s.append(flux_pulse(50, 0.0))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse()) # needs to be fast flux!

        s = self.get_sequencer(s)
        seqs = s.render()
        seqs = self.post_process_sequence(seqs, flatten=True)

        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
