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

class T1Measurement(Measurement1D):

    def __init__(self, qubit_info, delays, double_exp=False, seq=None,
                 postseq=None,laser_power = None, Qswitch_infoB=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = delays
        self.xs = delays / 1e3      # For plotting purposes
        self.double_exp = double_exp
        self.laser_power = laser_power
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.QswB = Qswitch_infoB

        super(T1Measurement, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
#        self.data.set_attrs(laser_power =self.laser_power)


    def generate(self):
        s = Sequence()
        s.append(Constant(250, 0, chan=4))
#        s.append(Constant(250, 1, chan='4m1'))
        r = self.qubit_info.rotate
        for i, dt in enumerate(self.delays):
            s.append(self.seq)
            s.append(r(np.pi, 0))
            s.append(Delay(dt))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

#            s.append(Repeat(Delay(1000), 20))   # wait for alazar acquisition to finish
#            s.append(Combined([
#                Repeat(Constant(8000, 0.4, chan=self.QswB.sideband_channels[0]), 70),
#                Repeat(Constant(8000, 0.4, chan=self.QswB.sideband_channels[1]), 70),
#                Repeat(Constant(8000, 1, chan='1m1'), 70),     # Readout pump tone switch
#                Repeat(Constant(8000, 0.0001, chan=5), 70),         # Qubit/Readout master switch
#            ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
