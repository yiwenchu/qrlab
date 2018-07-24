# -*- coding: utf-8 -*-
"""
Created on Thu Sep 04 10:16:29 2014
Jacob

untested
"""
import mclient
from ro_power_cal import Optimal_Readout_Power as orp
import time
import numpy as np
import matplotlib.pyplot as plt

ENV_FILE = r'C:\qrlab\scripts\calibration\ro_weight_func.npy'


def set_2Daxes_pcolor(xs, ys):
    '''
        xs and data are 1D arrays specifying the mesh.
    '''
    xs = np.array(xs)
    ys = np.array(ys)
    # adjust the 1D range by adding another element and shifting by half a period
    dx = xs[-1] - xs[-2]
    x_range = np.concatenate((xs, [xs[-1] + dx]))
    x_range = x_range - dx/2.0

    dy = ys[-1] - ys[-2]
    y_range = np.concatenate((ys, [ys[-1] + dy]))
    y_range = y_range - dy/2.0

    # generate mesh
    return np.meshgrid(x_range, y_range)


class Optimal_Readout_Power_and_Freq():
    def __init__(self, qubit_info, ro_powers, ro_freqs,
                 lo_gen='ag_lo', ro_gen='ag_ro', ro_obj='readout',
                 ro_if=50e6,
                 plot_best=False, update_readout=False, **kwargs):

        self.ro_ins = mclient.instruments[ro_gen]
        self.lo_ins = mclient.instruments[lo_gen]
        self.ro_obj = mclient.instruments[ro_obj]
        self.alz = mclient.instruments['alazar']

        self.ro_if = ro_if
        self.ro_freqs = ro_freqs
        self.ro_powers = ro_powers

        self.qubit_info = qubit_info

        self.update_readout = update_readout
        self.kwargs = kwargs

        self.best_fid = 0
        self.fids = []
        self.best_power = None
        self.best_freq = self.ro_ins.get_frequency()
        self.best_iqe = None
        self.best_iqg = None
        self.best_env = None

    def measure(self):
        self.old_ro_freq = self.ro_ins.get_frequency()
        self.old_power = self.ro_ins.get_power()
        self.old_nshots = self.alz.get_naverages()
        self.old_iqg = self.ro_obj.get_IQg()
        self.old_iqe = self.ro_obj.get_IQe()

        try:
            for f in self.ro_freqs:
                self.ro_ins.set_frequency(f)
                self.lo_ins.set_frequency(f+self.ro_if)
                time.sleep(0.1)

                o = orp(self.qubit_info, self.ro_powers,
                             plot_best=False,
                             update_readout=False, **self.kwargs)
                o.measure()
                self.fids.append(o.fids.flatten())

                print '\n At ro freq %0.3f GHz, power %.1f dbm, best fid: %0.03f \n' % \
                        (f, o.best_power, o.best_fid)

                if o.best_fid > self.best_fid:
                    self.best_fid = o.best_fid
                    self.best_power = o.best_power
                    self.best_freq = f
                    self.best_iqe = o.best_iqe
                    self.best_iqg = o.best_iqg
                    self.best_env = o.best_envelope

        finally:
            if self.update_readout:
                self.ro_ins.set_power(self.best_power)
                self.ro_ins.set_frequency(self.best_freq)
                self.lo_ins.set_frequency(self.best_freq+self.ro_if)
                self.ro_obj.set_IQg(self.best_iqg)
                self.ro_obj.set_IQe(self.best_iqe)

                np.save(ENV_FILE, self.best_env)

            else:
                self.ro_ins.set_power(self.old_power)
                self.ro_ins.set_frequency(self.old_ro_freq)
                self.lo_ins.set_frequency(self.old_ro_freq+self.ro_if)
                self.ro_obj.set_IQg(self.old_iqg)
                self.ro_obj.set_IQe(self.old_iqe)

            self.alz.set_weight_func(ENV_FILE)
            self.alz.set_naverages(self.old_nshots)

            self.fids = np.array(self.fids)


        self.plot()

    def plot(self):
        plt.figure()

        if len(self.ro_powers) > 1 and len(self.ro_freqs) > 1:
            xx, yy = set_2Daxes_pcolor(self.ro_powers, self.ro_freqs/1e9)
            plt.pcolor(xx, yy, np.array(self.fids))
            plt.colorbar()
            plt.xlabel('Readout Powers, dBm')
            plt.ylabel('Readout Frequencies, GHz')

        elif len(self.ro_powers) > 1:
            plt.plot(self.ro_powers,self.fids.flatten(),'ro-')
            plt.xlabel('Readout Powers, dBm')
            plt.ylabel('Blind Fidelities')
            plt.title('RO power sweep, frequency %0.05f GHz' % self.ro_freqs[0]/1e9)

        elif len(self.ro_freqs) > 1:
            plt.plot(self.ro_freqs,self.fids.flatten(),'ro-')
            plt.xlabel('Readout Freqs, GHz')
            plt.ylabel('Blind Fidelities')
            plt.title('RO freq sweep, power %0.02f dbm' % self.ro_powers[0])

        plt.show()
