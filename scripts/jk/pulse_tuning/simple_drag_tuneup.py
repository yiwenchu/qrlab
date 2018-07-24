import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit
from lib.math import fitter
from matplotlib import gridspec

FIT_PARABOLA    = 'PARABOLA'    # Fit a parabola (to determine min/max pos)

class Simple_DRAG_Tuneup(Measurement1D):

    def __init__(self, qubit_info, drag_range=None, update=False,
                 seq=None, postseq=None, buff_time=None,
                 **kwargs):

        self.qubit_info = qubit_info
        self.buff_time = buff_time
        self.drag_range = drag_range
        if drag_range == None:
            self.drag_range = np.linspace(-1.0, 1.0, 21)

        self.xs = self.drag_range
        self.update_drag = update
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(Simple_DRAG_Tuneup, self).__init__(2*len(self.drag_range),
                                          infos=(qubit_info,), **kwargs)
        # print self.drag_range
        self.data.create_dataset('drag_range', data=self.drag_range)

    def generate(self):
        r = self.qubit_info.rotate

        s = Sequence()

        for amp in self.drag_range:
            for i in (0,1):
                s.append(self.seq)
                if i:
                    s.append(r(np.pi/2, 0.0, drag=amp))
                    if self.buff_time is not None:
                        s.append(Delay(self.buff_time))
                    s.append(r(np.pi, np.pi/2, drag=amp))
                else:
                    s.append(r(np.pi/2, np.pi/2, drag=amp))
                    if self.buff_time is not None:
                        s.append(Delay(self.buff_time))
                    s.append(r(np.pi, 0.0, drag=amp))

                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        data = self.get_ys(data)

        d1 = data[::2]
        d2 = data[1::2]

        f1 = np.polyfit(self.drag_range, d1, 1)
        f2 = np.polyfit(self.drag_range, d2, 1)

        fx = np.linspace(min(self.drag_range), max(self.drag_range),100)

        y1 = np.polyval(f1, fx)
        y2 = np.polyval(f2, fx)

        opt_drag = (f2[1]-f1[1])/(f1[0]-f2[0])


        fig.axes[0].plot(self.xs, d1, 'ro')
        fig.axes[0].plot(self.xs, d2, 'bo')
        
        txt = 'Opt DRAG param: %0.3f' % (opt_drag,)
        fig.axes[0].plot(fx,y1, label=txt)
        fig.axes[0].plot(fx,y2)

        fig.axes[0].legend()



        if self.update_drag:
            print 'Setting drag to %.06f' % opt_drag

            ins = mclient.instruments[self.qubit_info.insname]
            ins.set_drag(opt_drag)

        return opt_drag

    def update(self, avg_data):
        ys = self.get_ys(avg_data)

        fig = self.get_figure()
        fig.axes[0].clear()

        fig.axes[0].plot(self.xs, ys[::2], 'ro')
        fig.axes[0].plot(self.xs, ys[1::2], 'bo')

        fig.canvas.draw()