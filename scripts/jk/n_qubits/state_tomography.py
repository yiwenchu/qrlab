import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit

import fitting_programs as fp

beta_inv2 = matrix([[ 0.25,  0.25,  0.25,  0.25],
                    [ 0.25, -0.25,  0.25, -0.25],
                    [ 0.25,  0.25, -0.25, -0.25],
                    [ 0.25, -0.25, -0.25,  0.25]])

class TwoQubitStateTomography(Measurement1D):
    '''It doesn't make sense to do this in a general version.  3 qubit
        tomo is rough, and 1 qubit is trivial.'''

    def __init__(self, qubit_info1, qubit_info2, seq=None, postseq=None,
                 beta_elements = None, **kwargs):

        self.q1 = qubit_info1
        self.q2 = qubit_info2
        self.beta_elements = None #Measure beta along with the tomography


        num_els = 16
        if beta_elements == None:
            num_els += 4

        super(TwoQubitStateTomography, self).__init__(num_els,
                                                      infos=[qubit_info1,qubit_info2],
                                                      **kwargs)
#        self.data.create_dataset('angles', data=angles)

        self.rots = {'I':(0,0),
                     'X':(np.pi,0),
                     'x':(np.pi/2,0),
                     'y':(np.pi/2,np.pi/2)}

        #Order here is maintained in generate and analyze
        self.cal_ops ['II','XI','IX','XX']
        self.pulses = ['II','IX','Ix','Iy',
                       'XI','XX','Xx','Xy',
                       'xI','xX','xx','xy',
                       'yI','yX','yx','yy']

    def s_to_rot(self,s):
        op1 = self.q1.rotation(*self.rots[s[0]])
        op2 = self.q2.rotation(*self.rots[s[1]])
        return Combined([op1,op2])

    def generate(self):
        r1 = self.q1.rotation
        r2 = self.q2.rotation

        s = Sequence()

        if self.beta_elements == None:
            for ops in self.cal_ops:
                s.append(s_to_rot(ops))
                s.append(self.get_readout_pulse())

        for ops in self.pulses:
            if seq is not None:
                s.append(seq)

            s.append(s_to_rot(ops))
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        return None

    def analyze(self, data=None, fig=None):
        #put the data in niceful variables (in the right order).
        c = {}
        op_names = ['c'+el for el in self.cal_ops]
        op_names.extend([el for el in self.pulses])
        for dpt,name in zip(data, opt_names):
            c[name] = dpt


        #first beta analysis
        beta_inv = np.matrix([[ 0.25,  0.25,  0.25,  0.25],
                              [ 0.25, -0.25,  0.25, -0.25],
                              [ 0.25,  0.25, -0.25, -0.25],
                              [ 0.25, -0.25, -0.25,  0.25]])

        K_ii, K_zi, K_iz, K_zz = beta_inv.np.array([c['cii'],c['czi'],
                                                    c['ciz'],c['czz'])

        #then tomographic reconstruction
        pass



#def beta_inv(n):
#    '''
#        yields the vector B_ij for comp state operators (e.g. II, ZI, IZ,ZZ),
#        given a complete vector of voltages from all input computational states
#        (00, 10, 01, 11), for arbitrary number of qubits n.
#
#        i.e.  converts voltages to Z measurements.  With prerotations we also
#        get X, Y measurements.
#    '''
#    b = np.matrix([[1,1],[1,-1]])
#    out = copy.copy(b)
#    for _ in range(n-1):
#        out = np.kron(out,b)
#    return np.linalg.inv(out)

