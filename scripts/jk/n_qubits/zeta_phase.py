import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit

import fitting_programs as fp

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
    xs = meas.angles

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    r = fp.Sine(x_data=xs, y_data=ys)
    result, params, ys_fit = r.fit(plot=False)



    txt = 'Frequency = %.03f +- %.03e\nPhase = %.03f +- %.03f' %\
            (params['frequency'].value, params['frequency'].stderr, \
            params['phase'].value, params['phase'].stderr)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, ys-ys_fit, marker='s')

    lmfit.report_fit(params)

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Rotation (Cycle)')
    fig.axes[0].legend(loc=0)

    fig.canvas.draw()
    return params

class RamseyPhase(Measurement1D):

    '''
        list of sequences in <seq> indicate that we want to compare
    '''

    def __init__(self, qubit_info, angles, delay, update=False,
                 seq=None, compare=False, r_axis=0, postseq=None,
                 fit_type=FIT_AMP, **kwargs):
        self.qubit_info = qubit_info
        self.angles = np.array(angles)/(np.pi*2) #This gets the plotting axes right.
        self.xs = np.array(angles)/(np.pi*2) #This gets the plotting axes right.
        self.delay = delay

        self.r_axis = r_axis
        self.fit_type = fit_type
        self.title = 'Ramsey phase: delay = %0.1f ns' % (delay)



        npoints = len(angles)
        if seq is None:
            seq = [Trigger(250)]
        elif type(seq) in [list]:
            npoints *= len(seq)
        else:
            raise ValueError('did not handle this <seq> input')
        self.split = len(seq)
        self.seq = seq
        self.postseq = postseq

        super(RamseyPhase, self).__init__(npoints, infos=qubit_info, **kwargs)
        self.data.create_dataset('angles', data=angles)

    def generate(self):
        s = Sequence()

        for i, angles in enumerate(self.angles):
            for seq in self.seq:
                s.append(seq)
                s.append(self.qubit_info.rotate(np.pi/2, self.r_axis))
                s.append(Delay(self.delay))
                s.append(self.qubit_info.rotate(np.pi/2, self.r_axis+np.pi*2*angles))

                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(RamseyPhase, self).get_ys(data)
        all_ys = []
        for i in np.arange(self.split):
            all_ys.append(ys[i::self.split])
        return np.array(all_ys)

    def update(self, avg_data):

        all_ys = self.get_ys(data=avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()

        if hasattr(self, 'xs'):
            for i in np.arange(len(all_ys)):
                fig.axes[0].plot(self.xs, all_ys[i,:], label=i)
        else:
            for i in len(all_ys):
                fig.axes[0].plot(all_ys[i,:], label=i)
        fig.axes[0].legend(loc='best')

        fig.canvas.draw()

    def analyze(self, data=None, fig=None):
        pass
#        self.fit_params = analysis(self, data=data, fig=fig)
#        return self.fit_params
