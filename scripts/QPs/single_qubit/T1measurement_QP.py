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
    params.add('amplitude', value=np.max(ys))
    params.add('tau', value=xs[-1]/2.0, min=0)
    result = lmfit.minimize(exp_decay, params, args=(xs, ys))
    lmfit.report_fit(params)

    fig.axes[0].plot(xs/1e3, -exp_decay(params, xs, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
    fig.axes[0].legend(loc=0)
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Time [us]')

    fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')
    fig.canvas.draw()
    return params

class T1Measurement_QP(Measurement1D):

    def __init__(self, qubit_info, delays, QP_delay, inj_len=10e3, seq = None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = delays
        self.QP_delay=QP_delay
        self.xs = delays / 1e3      # For plotting purposes
        self.inj_len=inj_len
        self.seq = seq
        self.marker_chan = kwargs.get('injection_marker',"3m2")
        if self.seq is None:
            self.seq = Trigger(250)

        super(T1Measurement_QP, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(inj_len=inj_len)
        self.data.set_attrs(QP_delay=QP_delay)


        
    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate
        for i, dt in enumerate(self.delays):
            s.append(self.seq)
            s.append(r(np.pi, 0))

            if self.inj_len < 20000 :
                s.append(Constant(self.inj_len, 1, chan=self.marker_chan))
            else:
                n_10us_pulse = int(self.inj_len)/10000 - 1
                s.append(Repeat(Constant(10000, 1, chan=self.marker_chan), n_10us_pulse))
                s.append(Constant(self.inj_len-n_10us_pulse*10000, 1, chan=self.marker_chan))

            s.append(Delay(self.QP_delay))
            s.append(r(np.pi, 0))
            s.append(Delay(dt))
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
