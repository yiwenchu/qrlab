import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lib.math import fit
import copy
import mclient
from measurement import Measurement1D
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import lmfit
import math


from lib.math import *

def analysis(self,xs, ys, fig, proj_num=0, vary_ofs=True):
    if 0:
    	fitstyle='exp_decay_sine2'
    else:
        fitstyle='exp_decay_sine'
    
    f = fitter.Fitter(fitstyle)
    p = f.get_lmfit_parameters(xs, ys)
    	
    p['ofs'].value = 0.5*max(ys)
    result = f.perform_lmfit(xs, ys, p=p, plot=False)
    p = result.params
    ys_fit = f.eval_func()

    txt = 'cavity dephasing time: %0.3f +/- %0.4f\n' %(p['tau'].value,p['tau'].stderr)
    txt += 'Amp = %0.3f +/- %0.4f' % (p['A'].value, p['A'].stderr)
   
    fig.axes[0].plot(xs, ys_fit, 'g-', label=txt)
    fig.axes[0].legend()

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Intensity [AU]')
    fig.axes[1].set_xlabel('Delays [ms]')

    fig.axes[1].plot(xs, ys_fit-ys, 'ks-')
    fig.canvas.draw()
    return p


# def analysis(meas, data=None, fig=None):
#     xs = meas.delays
#     ys, fig = meas.get_ys_fig(data, fig)

#     # fig.axes[0].plot(xs/1e3, ys, 'ks', ms=3)

#     amp0 = (np.max(ys) - np.min(ys)) / 2
#     fftys = np.abs(np.fft.fft(ys - np.average(ys)))
#     fftfs = np.fft.fftfreq(len(ys), xs[1]-xs[0])
#     f0 = np.abs(fftfs[np.argmax(fftys)])
#     print 'Oscillation freq: %.03f kHz' % (f0 * 1e6) # - (meas.data.detuning * 1e-3))

#     params = lmfit.Parameters()
#     params.add('ofs', value=amp0)
#     params.add('amp', value=amp0, min=0)
#     params.add('tau', value=xs[-1], min=10, max=200000)
#     params.add('freq', value=f0, min=0)
#     params.add('phi0', value=0, min=-1.2*np.pi, max=1.2*np.pi)
#     result = lmfit.minimize(t2_fit, params, args=(xs, ys))
#     lmfit.report_fit(params)

#     fig.axes[0].plot(xs/1e3, -t2_fit(params, xs, 0), label='Fit, tau=%.03f us, df=%.03f kHz'%(params['tau'].value/1000, params['freq'].value*1e6))
#     fig.axes[0].legend()
#     fig.axes[0].set_ylabel('Intensity [AU]')
#     fig.axes[0].set_xlabel('Time [us]')
#     fig.axes[1].plot(xs/1e3, t2_fit(params, xs, ys), marker='s')
#     fig.canvas.draw()

#     return params

class CavT2SNAP2(Measurement1D):

    def __init__(self, qubit_info, cav_info, angle, delays, proj_num, detune=0, seq=None, bgcor=False, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.angle = angle
        self.delays = delays
        self.proj_num = proj_num
        self.detune = detune
        self.bgcor = bgcor
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.xs = self.delays/1e3

        npoints = len(self.delays)
        if bgcor:
            npoints *= 2
        super(CavT2SNAP2, self).__init__(npoints, infos=(qubit_info, cav_info), **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            initPhase=angle,
            detuning=detune,
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate_selective
        c = self.cav_info.rotate
        for i, delay in enumerate(self.delays):
            for bg in (0, 1):
                if bg and not self.bgcor:
                    continue

                s.append(self.seq)
                s.append(c.displace(0.56, self.angle))
                s.append(r(2*np.pi, X_AXIS))
                s.append(c.displace(0.24, self.angle + np.pi))
                
                if delay > 0:
                    s.append(Delay(delay))

                dphi = 2 * np.pi * self.detune * delay * 1e-9 + np.pi
                s.append(c.displace(0.8, self.angle + dphi))

                if not bg:
                    s.append(r(np.pi, X_AXIS))

                s.append(Combined([
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(CavT2SNAP2, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def get_all_data(self, data=None):
        ys = super(CavT2SNAP2, self).get_ys(data)
        if self.bgcor:
            return ys[::2], ys[1::2]
        return ys[:], None

    def update(self, avg_data):
        data, bg_data = self.get_all_data(avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()
        fig.axes[1].clear()

        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, data, 'rs-', label='raw data')
            if self.bgcor:
                fig.axes[0].plot(self.xs, bg_data, 'bs-', label='background')
                fig.axes[0].plot(self.xs, data-bg_data, 'ks-', label='bg subtracted')
        else:
            fig.axes[0].plot(data, 'rs-', label='raw data')
            if self.bgcor:
                fig.axes[0].plot(bg_data, 'bs-', label='background')
                fig.axes[0].plot(data-bg_data, 'ks-', label='bg subtracted')

        fig.axes[0].legend(loc='best')
        fig.axes[1].legend(loc='best')

        fig.canvas.draw()

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, self.xs, data, fig)
        return self.fit_params['tau'].value
