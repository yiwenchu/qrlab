import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lib.math import fitter
import copy
import mclient
from measurement import Measurement1D
from pulseseq.sequencer import *
from pulseseq.pulselib import *


OP_SET = np.array(["II","XX","YY","XY","YX","xI","yI","xy","yx","xY",
          "yX","Xy","Yx","xX","Xx","yY","Yy","XI","YI","xx","yy"])

def analysis(ys, fig, OP_SET=OP_SET):
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)
    ax = fig.axes[0]

    ax.plot(ys, 'rs')
    ax.set_xticks(np.arange(len(OP_SET)))
    ax.set_xticklabels(OP_SET, rotation=90)

    fig.canvas.draw()

class AllXY(Measurement1D):

    def __init__(self, qubit_info, seq=None, postseq=None,
                 selective=False, buff_time=None, **kwargs):
        self.qubit_info = qubit_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.selective = selective
        self.buff_time = buff_time

        super(AllXY, self).__init__(len(OP_SET), infos=(qubit_info), **kwargs)
        self.data.create_dataset('op_set', data=OP_SET, dtype="S10")
        self.data.set_attrs(
            qubit_info = qubit_info.insname,
            selective = selective
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate
        if self.selective:
            r = self.qubit_info.rotate_selective

        for ops in OP_SET:
            s.append(self.seq)
            for op in ops:
                if op == "X": newPulse = r(np.pi,0)
                if op == "Y": newPulse = r(np.pi,np.pi/2)
                if op == "x": newPulse = r(np.pi/2,0)
                if op == "y": newPulse = r(np.pi/2,np.pi/2)
                if op == "I": newPulse = Delay(r(np.pi,0).get_length())
                s.append(newPulse)
                if self.buff_time is not None:
                    s.append(Delay(self.buff_time))
            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        analysis(data, fig)
