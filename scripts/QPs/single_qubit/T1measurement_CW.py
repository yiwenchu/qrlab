import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit
from lib.math import fitter

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

#def analysis(xs, ys, fig, double_exp=False):
#    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
#    ys, fig = meas.get_ys_fig(data, fig)

#    fig.axes[0].plot(xs, ys, 'ks', ms=3)
#
#    if double_exp == False:
#        f = fitter.Fitter('exp_decay')
#        result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
#        ys_fit = f.eval_func()
#        tau = result.params['tau']
#        txt = 'tau = %0.3f us +\- %0.4f us' % (tau.value, tau.stderr)
#    else:
#        f = fitter.Fitter('exp_decay_double')
#        result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
#        ys_fit = f.eval_func()
#
#        A = result.params['A'].value
#        B = result.params['B'].value
#        A_weight = A/(A + B) * 100.0
#        B_weight = B/(A + B) * 100.0
#        tau1 = result.params['tau'].value
#        tau1_err = result.params['tau'].stderr
#        tau2 = result.params['tau1'].value
#        tau2_err = result.params['tau1'].stderr
#
#        txt = 'tau1 (weight) = %0.3f us +/- %0.3f us (%0.2f%%)\n' % (tau1, tau1_err, A_weight)
#        txt += 'tau2 (weight) = %0.3f us +/- %0.3f us (%0.2f%%)' % (tau2, tau2_err, B_weight)
#
#    fig.axes[0].plot(xs, ys_fit, label=txt)
#    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')
#
#    fig.axes[0].legend()
#    fig.axes[0].set_ylabel('Intensity [AU]')
#    fig.axes[1].set_ylabel('Residuals')
#    fig.axes[1].set_xlabel('Time [us]')
#
#    fig.canvas.draw()
#    return result.params

class T1Measurement_CW(Measurement1D):

    def __init__(self, qubit_info, delays, laser_voltage, atten=None, double_exp=False, seq=None,
                 postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3      # For plotting purposes
        self.double_exp = double_exp
        self.laser_voltage = laser_voltage
        self.atten = atten #dB
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(T1Measurement_CW, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.set_attrs(
            rep_rate=self.instruments['funcgen'].get_frequency()
        )
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(laser_voltage = self.laser_voltage)
        self.data.set_attrs(attenuation = self.atten)
        

    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate
        for i, dt in enumerate(self.delays):
            s.append(self.seq)
            s.append(r(np.pi, 0))
            s.append(Delay(dt))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
#        ys = self.get_ys(data)
#        if fig is None:
#            fig = self.get_figure()
#        self.fit_params = analysis(self.xs, ys, fig, double_exp=self.double_exp)
#        return self.fit_params
