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
    if self.disp==0:
        snapping = 1
    else:
    	snapping=0

    if not snapping:
        fitstyle='cohstate_decay'
    else:
    	fitstyle = 'exp_decay'

    f = fitter.Fitter(fitstyle)
    p = f.get_lmfit_parameters(xs, ys)
    if fitstyle=='cohstate_decay':	
    	p['n'].value = proj_num
    	p['n'].vary = False
    p['ofs'].value = 0.0
    result = f.perform_lmfit(xs, ys, p=p, plot=False)
    p = result.params
    ys_fit = f.eval_func()

    txt = 'cavity decay time: %0.3f +/- %0.4f\n' %(p['tau'].value,p['tau'].stderr)
    if snapping:
        txt += 'Amp = %0.3f +/- %0.4f' % (p['A'].value, p['A'].stderr)
    else:
    	txt += 'Amp = %0.3f +/- %0.4f\n' % (p['amp'].value, p['amp'].stderr)
    	txt += 'disp = %0.3f +/- %0.4f' % (p['alpha0'].value, p['alpha0'].stderr)
   
    fig.axes[0].plot(xs, ys_fit, 'g-', label=txt)
    fig.axes[0].legend(loc=0)

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Intensity [AU]')
    fig.axes[1].set_xlabel('Delays [ms]')

    fig.axes[1].plot(xs, ys_fit-ys, 'ks-')
    fig.canvas.draw()
    return p



class CavT1SNAP(Measurement1D):

    def __init__(self, qubit_info, cav_info, disp, delays,
                 proj_num, seq=None, extra_info=None, bgcor=True, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.disp = disp
        self.delays = delays
        self.proj_num = proj_num
        self.bgcor = bgcor
        self.postseq = postseq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.xs = self.delays/1e3

        npoints = len(self.delays)
        if bgcor:
            npoints *= 2
        super(CavT1SNAP, self).__init__(npoints, infos=(qubit_info, cav_info), **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            disp=disp,
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate_selective
        c = self.cav_info.rotate

        for i, delay in enumerate(self.delays):
            for bg in (0,1):
                if bg and not self.bgcor:
                    continue

                s.append(self.seq)
                if self.disp ==0:
                	s.append(c.displace(1.14, 0))
	                s.append(r(2*np.pi, X_AXIS))
	                s.append(c.displace(0.58, np.pi))
                else:
                	s.append(c.displace(np.abs(self.disp), np.angle(self.disp)))
                
                s.append(Delay(delay))
                #
                s.append(r(np.pi*(1-bg), X_AXIS))
#                s.append(r(np.pi, 0))
                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(CavT1SNAP, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys


    def get_all_data(self, data=None):
        ys = super(CavT1SNAP, self).get_ys(data)
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