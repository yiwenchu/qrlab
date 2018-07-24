# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D, STYLE_LINES
from lib.math import fitter
import lmfit

import scripts.single_cavity.RotFrameTest2 as RFT2

'''
This will call the various rft2.generate for various alphas to put them together
in one script.

unfinished -Jacob 8/3/14
'''

class Kerr(Measurement2D):

    def __init__(self, qubit_info, cav_info, delays, angles, alphas,
                 bgcor=True, seq=None, plot=True,**kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.bgcor = bgcor
        self.delays = np.array(delays)
        self.alphas = alphas
        self.angles = np.array(angles)
        self.plot = plot

        self.seq = seq

        # Set plotting parameters
        self.xs = self.delays
        self.ys = self.angles

        npoints = len(delays) * len(angles) * len(alphas)
        if bgcor:
            npoints *= 2

        super(Kerr, self).__init__(npoints, infos=(qubit_info, cav_info), style=STYLE_LINES, residuals=False, **kwargs)

        self.data.create_dataset('delays', data=self.delays)
        self.data.create_dataset('angles', data=self.angles, dtype=self.angles.dtype)
        self.data.create_dataset('alphas', data=self.alphas, dtype=self.alphas.dtype)


    def seq_append(self,oldseqs, newseqs):
        for ch in oldseqs:
            oldseqs[ch].append(newseqs[ch])
        return oldseqs

    def generate(self):
        '''Basically copied from RotFrameTest2 with an extra loop'''
        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate

        for alpha in self.alphas:
            for i_delay, delay in enumerate(self.delays):
                for i_angle, angle in enumerate(self.angles):
                    for i_bg in [1,0]:
                        if i_bg == 0 and not self.bgcor:
                            continue

                        s.append(self.seq) #trigger, displace
                        if delay > 0:
                            s.append(Delay(delay))

                        s.append(c(np.pi * np.abs(alpha), np.angle(alpha) +np.pi + angle))

                        s.append(r(np.pi*i_bg, 0))
                        s.append(self.get_readout_pulse())


        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def get_ys(self, data=None):
        ys = super(Kerr, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def get_all_ys(self, data=None):
        ys = super(Kerr, self).get_ys(data)
        if self.bgcor:
            return ys[::2], ys[1::2]
        return ys

    def create_figure(self):
        self.fig = plt.figure()
        title = self.title
        if self.data:
            title += ' data in %s' % self.data.get_fullname()
        self.fig.suptitle(title)

        gs = gridspec.GridSpec(1, len(self.alphas), width_ratios=[1,1])

        for i in range(len(self.alphas)):
            self.fig.add_subplot(gs[i])
        return self.fig

    def update(self, avg_data):
        na = len(self.alphas)
        data = self.get_ys(avg_data)

        fig = self.get_figure()

        for i in range(na):
            fig.axes[i].clear()

            if len(self.ys) > 1 and len(self.xs) > 1:
                data = data.reshape((len(self.xs), len(self.ys)))

                ys = self.ys[i*na:(i+1)*na]
                datai = data[:,i*na:(i+1)*na]
                '''A little confused by this part'''

                delta_ys = ys[1] - ys[0]
                ys_plot = np.concatenate((ys, [ys[-1] + delta_ys])) - delta_ys/2.0
                delta_xs = self.xs[1] - self.xs[0]
                xs_plot = np.concatenate((self.xs, [self.xs[-1] + delta_xs])) - delta_xs/2.0

                fig.axes[i].pcolor(ys_plot, xs_plot, datai)
    #            fig.colorbar()
            elif len(self.ys) == 1:
                fig.axes[i].plot(self.xs, data)
            elif len(self.xs) == 1:
                fig.axes[i].plot(self.ys, data)

        if self.plot:
            fig.canvas.draw()

    def analyze(self, data=None, fig=None):

        self.detuning = analysis(self, self.get_ys())


