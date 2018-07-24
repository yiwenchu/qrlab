# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D, STYLE_LINES
from lib.math import fitter
import lmfit

def analysis(meas, data=None):
    if data is None:
        data = meas.get_ys()

    if len(meas.xs) > 1 and len(meas.ys) >1:
        centers = []
        data = data.reshape((len(meas.xs), len(meas.ys)))

        for xidx in range(len(meas.xs)):
            fit = fitter.Fitter('gaussian')
            result = fit.perform_lmfit(meas.ys, data[xidx],
                                       plot=False,
                                       print_report=True)
            centers.append(result.params['x0'].value)

        if meas.plot:
            plt.figure()
            plt.plot(meas.xs, centers, 'ko-')

        fit = fitter.Fitter('linear')
        result = fit.perform_lmfit(meas.xs, centers, plot=False)
        p = result.params
        ys_fit = fit.test_values(meas.xs, p=p)

        detuning_MHz = 1e3*result.params['m'].value/(2*np.pi)

        if meas.plot:
            plt.plot(meas.xs, ys_fit)
            plt.title('Detuning: %0.6f MHz' % detuning_MHz)
            plt.show()

    return detuning_MHz

class RotFrameTest(Measurement2D):

    def __init__(self, qubit_info, cav_info, delays, angles, alpha0 = 1.0,
                 bgcor=True, seq=None, plot=True,**kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.bgcor = bgcor
        self.delays = np.array(delays)
        self.alpha0 = alpha0
        self.angles = np.array(angles)
        self.plot = plot

        # Initial sequence
        if seq is None:
            seq = [Trigger(250), cav_info.rotate(np.pi * np.abs(alpha0), np.angle(alpha0))]

        self.seq = seq

        # Set plotting parameters
        self.xs = self.delays
        self.ys = self.angles

        npoints = len(delays) * len(angles)
        if bgcor:
            npoints *= 2

        super(RotFrameTest, self).__init__(npoints, infos=(qubit_info, cav_info), style=STYLE_LINES, residuals=False, **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.create_dataset('angles', data=self.angles, dtype=self.angles.dtype)
        self.data.set_attrs(
            alpha0=alpha0,
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        for i_delay, delay in enumerate(self.delays):
            for i_angle, angle in enumerate(self.angles):
                for i_bg in [1,0]:
                    if i_bg == 0 and not self.bgcor:
                        continue

                    s.append(self.seq) #trigger, displace
                    if delay > 0:
                        s.append(Delay(delay))

                    s.append(c(np.pi * np.abs(self.alpha0), np.angle(self.alpha0) +np.pi + angle))

                    s.append(r(np.pi*i_bg, 0))
                    s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(RotFrameTest, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def get_all_ys(self, data=None):
        ys = super(RotFrameTest, self).get_ys(data)
        if self.bgcor:
            return ys[::2], ys[1::2]
        return ys

    def update(self, avg_data):
        data = self.get_ys(avg_data)

        fig = self.get_figure()
        fig.axes[0].clear()

        if len(self.ys) > 1 and len(self.xs) > 1:
            data = data.reshape((len(self.xs), len(self.ys)))
            delta_ys = self.ys[1] - self.ys[0]
            ys_plot = np.concatenate((self.ys, [self.ys[-1] + delta_ys])) - delta_ys/2.0
            delta_xs = self.xs[1] - self.xs[0]
            xs_plot = np.concatenate((self.xs, [self.xs[-1] + delta_xs])) - delta_xs/2.0

            fig.axes[0].pcolor(ys_plot, xs_plot, data)
#            fig.colorbar()
        elif len(self.ys) == 1:
            fig.axes[0].plot(self.xs, data)
        elif len(self.xs) == 1:
            fig.axes[0].plot(self.ys, data)

        if self.plot:
            fig.canvas.draw()

    def analyze(self, data=None, fig=None):

        self.detuning = analysis(self, self.get_ys())


