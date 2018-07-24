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

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.delays

    fig.axes[0].plot(xs/1e3, ys, 'ks', ms=3)

    params = lmfit.Parameters()
    params.add('ofs', value=np.min(ys))
    params.add('amplitude', value=np.max(ys)-np.min(ys))
    params.add('tau', value=len(xs)*1000/4.0, min=0)
    result = lmfit.minimize(exp_decay, params, args=(xs, ys))
    lmfit.report_fit(params)

    fig.axes[0].plot(xs/1e3, -exp_decay(params, xs, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
    fig.axes[0].legend(loc=0)
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Time [us]')

    fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')
    fig.canvas.draw()
    return params

class FT1Measurement(Measurement1D):

    def __init__(self, qubit_info, ef_info, delays, **kwargs):
        self.qubit_info = qubit_info
        self.ef_info = ef_info
        self.delays = delays
        self.xs = delays / 1e3      # For plotting purposes

        super(FT1Measurement, self).__init__(len(delays), ssbs=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate
        r_ef = self.ef_info.rotate
        for i, dt in enumerate(self.delays):
            if dt < 250:
                s.append(Join([
                    Trigger(dt=250),
                    r(np.pi, 0),
                    r_ef(np.pi, 0),
                    Delay(dt),
                ]))
            else:
                s.append(Join([
                    Trigger(dt=250),
                    r(np.pi, 0),
                    r_ef(np.pi, 0),
                ]))
                s.append(Delay(dt))
            s.append(r(np.pi/2, 0))
            s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))

        s = Sequencer(s)
        seqs = s.render()
        if self.qubit_info.ssb is not None:
            self.qubit_info.ssb.modulate(seqs)
        if self.ef_info.ssb is not None:
            self.ef_info.ssb.modulate(seqs)
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
