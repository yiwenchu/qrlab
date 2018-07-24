import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from scripts.single_qubit import rabi
import copy
import mclient
import lmfit

import fitting_programs as fp

FIT_AMP         = 'AMP'         # Fit simple sine wave
FIT_AMPFUNC     = 'AMPFUNC'     # Try to fit amplitude curve based on pi/2 and pi amp
FIT_PARABOLA    = 'PARABOLA'    # Fit a parabola (to determine min/max pos)

def analysis(xs, data, fig=None):

    if fig is None:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        fig.add_subplot(gs[0])
        fig.add_subplot(gs[1])

    params = []
    for ys in data:
        p = rabi.analysis(xs, ys, fig)
        params.append(p)

    fig.axes[0].set_xlabel('Ramsey angle')
    fig.canvas.draw()

    return params


class RamseyPhase(Measurement1D):

    '''
        list of sequences in <seq> indicate that we want to compare
    '''

    def __init__(self, qubit_info, phases, delay, update=False,
                 seq=None, compare=False, r_axis=0, postseq=None,
                 fit_type=FIT_AMP, **kwargs):
        self.qubit_info = qubit_info
        self.phases = np.array(phases)
        self.xs = np.array(phases)/(2*np.pi)
        self.delay = delay

        self.r_axis = r_axis
        self.fit_type = fit_type
        self.title = 'Ramsey phase: delay = %0.1f ns' % (delay)

        npoints = len(phases)
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
        self.data.create_dataset('phases', data=phases)

    def generate(self):
        s = Sequence()

        for i, phases in enumerate(self.phases):
            for seq in self.seq:
                s.append(seq)
                s.append(self.qubit_info.rotate(np.pi/2, self.r_axis))
                s.append(Delay(self.delay))
                s.append(self.qubit_info.rotate(np.pi/2, self.r_axis+phases))

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
        self.fit_params = analysis(self.xs, data, fig=self.get_figure())
        return self.fit_params
