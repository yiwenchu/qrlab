import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit
from lib.math import fitter

def analysis(xs, ys, fig, double_exp=False):

#    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
#    ys, fig = meas.get_ys_fig(data, fig)

    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    if double_exp == False:
        f = fitter.Fitter('exp_decay')
        result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
        ys_fit = f.eval_func()
        tau = result.params['tau']
        txt = 'tau = %0.3f us +\- %0.4f us' % (tau.value, tau.stderr)
    else:
        f = fitter.Fitter('exp_decay_double')
        result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
        ys_fit = f.eval_func()

        A = result.params['A'].value
        B = result.params['B'].value
        A_weight = A/(A + B) * 100.0
        B_weight = B/(A + B) * 100.0
        tau1 = result.params['tau'].value
        tau1_err = result.params['tau'].stderr
        tau2 = result.params['tau1'].value
        tau2_err = result.params['tau1'].stderr

        txt = 'tau1 (weight) = %0.3f us +/- %0.3f us (%0.2f%%)\n' % (tau1, tau1_err, A_weight)
        txt += 'tau2 (weight) = %0.3f us +/- %0.3f us (%0.2f%%)' % (tau2, tau2_err, B_weight)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()
    return result.params

class T1Measurement(Measurement1D):

    def __init__(self, qubit_info, delays, double_exp=False, seq=None, bgcor = False,
                 postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3      # For plotting purposes
        self.double_exp = double_exp
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.bgcor = bgcor

        npoints = len(delays)
        if bgcor:
            npoints += 2
        super(T1Measurement, self).__init__(npoints, infos=qubit_info, **kwargs)
        self.data.set_attrs(
            rep_rate=self.instruments['funcgen'].get_frequency()
        )
        self.data.create_dataset('delays', data=delays)

    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate
        
        if self.bgcor:
            s.append(self.seq)
            s.append(self.get_readout_pulse())            

            s.append(self.seq)
            s.append(r(np.pi, X_AXIS))
            s.append(self.get_readout_pulse())
            
        for i, dt in enumerate(self.delays):
            s.append(self.seq)
            s.append(r(np.pi, 0))
            s.append(Delay(dt))
            
#            self.seq2 = Sequence()
#            self.seq2.append(Constant(dt, 1, chan="3m1"))
#            s.append(self.seq2)

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(T1Measurement, self).get_ys(data)
        if self.bgcor:
            return (ys[2:] - ys[0])/(ys[1]-ys[0])
        return ys

    def analyze(self, data=None, fig=None):
        ys = self.get_ys()
        if fig is None:
            fig = self.get_figure()
        self.fit_params = analysis(self.xs, ys, fig, double_exp=self.double_exp)
        return self.fit_params
