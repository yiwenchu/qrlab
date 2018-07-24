# -*- coding: utf-8 -*-
"""
Created on Mon Oct 06 14:15:10 2014

@author: rsl
"""

import numpy as np
import matplotlib.pyplot as plt
import cavity_analysis

def quick_lor_fit(freq, db_mag, show=False, bkg=False, grads_are_data=False):
    return quick_hanger_fit(freq, db_mag, show=show, bkg=bkg, grads_are_fitdata=True)

def quick_hanger_fit(freq, db_mag, show=False, bkg=False, grads_are_data=False, grads_are_fitdata=False):
    x = freq
    y0 = db_mag
    if bkg:
        slope = np.linspace(y0[0],y0[-1],np.size(x))
        y0 = np.subtract(y0,slope)
    params, fitdata = cavity_analysis.fit_asymmetric_db_hanger(x, y0)
    f0 = params['f0'].value
    qi = params['qi'].value
    qcr = params['qcr'].value
    qci = params['qci'].value
    bw = f0/qi
    fit_label = 'center = %0.5f GHz;\n' % (f0/1e9)
    fit_label += 'Qi = %0.1f;\n' % (qi)
    fit_label += 'Qc = %0.1f (%0.1f + j%0.1f);\n' % (qcr,qcr,qci)
    fit_label += 'BW = %0.2f kHz' % (bw/1e3)
    if not grads_are_fitdata:
        if grads_are_data:
            graddata = y0
        else:
            graddata = fitdata
        grad = np.gradient(np.array([x, graddata])) # derivative of fit
        grads = grad[1][1]
        grad = np.gradient(np.array([x, grads]))
        grads = np.abs(grad[1][1]) # second derivative
    else:
        grads = fitdata
    if show:
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        fig = plt.figure()
        fig.add_subplot(gs[0])
        fig.add_subplot(gs[1])
        fig.axes[0].plot(x, y0, marker='s', ms=3, label='')
        fig.axes[0].plot(x, fitdata, ls='-', ms=3, label=fit_label)
        fig.axes[1].plot(x, grads, ls='-', ms=3, label='')
        fig.axes[0].legend(loc='best')
        fig.axes[0].set_xlabel('Frequency (Hz)')
        fig.axes[0].set_ylabel('Magnitude (dB)')
    return f0, qi, bw, grads # all in Hz
    
if __name__ == '__main__':
    # an example:
    testfname = "Z:\\_Data\\3D\\LZ_140926\\CA\\1\\-20.00dB_26.00mK.dat"
    x, y0, y1 = np.loadtxt(testfname, unpack=True, delimiter='\t')
    fitout = quick_hanger_fit(x, y0, show=True)