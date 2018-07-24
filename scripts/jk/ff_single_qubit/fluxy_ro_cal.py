from measurement import Measurement
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import objectsharer as objsh
import time
import mclient
import scripts.single_qubit.rabi as rabi
import scripts.calibration.thresholding_tools as tt
import fitting_programs as fp
import scripts.calibration.histogram_calibration as hc
import gc

ENV_FILE = r'C:\qrlab\scripts\calibration\ro_weight_func.npy'

class Fluxy_RO_Cal(Measurement):
    '''
    Sweep readout power and/or readout flux voltage.

    At each point in that 2D space it records e&g shot trajectories, then uses
    those trajectories to find an optimal integration window, then uses those
    shots with the new integration window to find the maximal blind fidelity.

    Note that when we use integration windows there's no need to do cumulative
    blind fidelity, only final blind fidelity.
    '''

    def __init__(self, qubit_info, powers, flux_amps, shots=1e5,
                 seq=None, update_readout = False,verbose_plots = False,**kwargs):
        self.qubit_info = qubit_info
        self.powers = powers
        self.shots = shots
        self.flux_amps = flux_amps
        self.update_readout = update_readout
        self.verbose_plots = verbose_plots

        self.num_powers = len(powers)
        self.num_flux_amps = len(flux_amps)

        super(Fluxy_RO_Cal, self).__init__(2*self.num_flux_amps,
                                            infos=qubit_info, **kwargs)

        self.alz = self.instruments['alazar']
        self.ag_ro = self.instruments['ag_ro']

        self.data.create_dataset('powers', data=powers)
        self.data.create_dataset('flux_amps', data=flux_amps)

        self.blind_fids = self.data.create_dataset('blind_fids',
                                 shape=[self.num_powers,
                                        self.num_flux_amps])

        if not hasattr(self.readout_info,'flux_chan'):
            raise ValueError('Need to use ff_readout type')

    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate

        for flux_idx,flux_amp in enumerate(self.flux_amps):
            for amp in [np.pi,0]:
                s.append(Trigger(250))
                s.append(r(amp,0))

#                s.append(Delay(20))  #TODO: DELETE THIS

                self.readout_info.ro_flux_amp = flux_amp
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        return s.render()

    def measure(self):
        self.alz.setup_channels()
        self.alz.setup_clock()
        self.alz.setup_trigger()

        self.save_settings()
        self.alz.set_weight_func('')

        self.fids = np.zeros((self.num_powers,self.num_flux_amps))

        self.best_fid = 0
        self.best_power = None
        self.best_flux = None
        self.best_envelope = None

        for pidx, p in enumerate(self.powers):
            #All of the orders here are important!  Don't muck with!
            self.ag_ro.set_power(p)

            self.stop_awgs()
            self.stop_funcgen()
            time.sleep(1)

            self.alz.setup_demod_shots(self.num_flux_amps*2*self.shots)
            self.play_sequence(load=(pidx==0))

            buf = self.alz.take_demod_shots(timeout=1e6)

            '''pts come in [[delay1e],[delay1g],[delay2e],[delay2g]...]'''
            for fidx,famp in enumerate(self.flux_amps):
                ebuf = buf[2*fidx::2*self.num_flux_amps,:]
                gbuf = buf[2*fidx+1::2*self.num_flux_amps,:]

                #TODO:  This code needs to be faster.  The analysis is quite
                #slow with large datasets.  Consider numexpr?
                #Find the average e and g trajectories to choose an int window
                avg_ebuf = np.average(ebuf,axis=0)
                avg_gbuf = np.average(gbuf,axis=0)

                diff = (avg_ebuf - avg_gbuf)
                diff = (np.real(diff) + 1j * np.imag(diff))
                diff /= np.sum(np.abs(diff))
                r_env = np.real(diff)
                i_env = np.imag(diff)

                if self.verbose_plots:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(np.real(avg_ebuf),np.imag(avg_ebuf))
                    ax.plot(np.real(avg_gbuf),np.imag(avg_gbuf))
                    fig.canvas.draw()

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(r_env)
                    ax.plot(i_env)
                    fig.canvas.draw()

                #Following the analyis in Alazar_Daemon.get_IQ_rel()
                IQ_e =      np.inner(np.real(ebuf), r_env) + \
                       1j * np.inner(np.imag(ebuf), i_env)

                IQ_g =      np.inner(np.real(gbuf), r_env) + \
                       1j * np.inner(np.imag(gbuf), i_env)

                fid = hc.blind_fidelity(IQ_e,IQ_g,plot=self.verbose_plots,
                                        verbose=False, num_steps = 26)

                self.fids[pidx,fidx] = fid

                if fid > self.best_fid:
                    self.best_fid = fid
                    self.best_power = p
                    self.best_flux = famp
                    self.best_envelope = diff

            print 'finished %0.2d dBm: best fid = %0.3f\n' %\
                        (p, np.max(np.array(self.fids[pidx,:]).flatten()))

        self.powers = np.array(self.powers)
        self.flux_amps = np.array(self.flux_amps)

        #print self.fids[:]
        self.analyze()

        if self.update_readout:
            print "HAVN'T UPDATED READOUT FLUX!"
            #self.readout_info.ro_flux_amp self.best_flux
            #needs to return e, g points!
            self.ag_ro.set_power(self.best_power)
            np.save(ENV_FILE, self.best_envelope)
            self.alz.set_weight_func(ENV_FILE)

        return self.best_fid, self.best_power, self.best_flux, self.best_envelope

    def analyze(self):
        #Holy shit I hate matplotlib.
        import fitting_programs as fp
        xs = self.powers
        ys = self.flux_amps
        values = self.fids[:]
        if len(xs) >1 and len (ys)>1:

            xx, yy = fp.set_2Daxes_pcolor(xs,ys)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            im = ax.pcolor(xx, yy, np.transpose(values),
                          vmax=np.nanmax(values),vmin=np.nanmin(values),cmap='jet')
            fig.colorbar(im)

    #        #ax.set_title(txt)
    #
            ax.set_xlim(xx.min()), xx.max()
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(r'RO Power (dBm)')
            ax.set_ylabel(r'RO Flux (DAC units)')

            fig.canvas.draw()
        else:
            print self.fids[:]
#        extent = (xs.min(), xs.max(), ys.min(), ys.max())

#        im = ax.pcolormesh(yy,xx,values)#.reshape(xx.shape))#, aspect='equal')#interpolation='nearest')#,
#                          origin='lower',
#                          extent=extent)
#        fig.colorbar(im)
#
#        #ax.set_title(txt)
#
##        ax.set_xlim(xs.min()), xs.max()
##        ax.set_ylim(ys.min(), ys.max())
#        ax.set_xlabel(r'$Re \{\alpha \}$')
#        ax.set_ylabel(r'$Im \{\alpha \}$')
#
#        fig.canvas.draw()



#        [p_data,f_data,T_data] = [calculate_max1D_cut(self.fids, axis) for axis in [0, 1, 2]]
#        [p_data,f_data] = [calculate_max1D_cut(self.fids, axis) for axis in [0, 1]]
#
#        fig = plt.figure()
#
#        ax = fig.add_subplot(211)
#        ax.plot(self.powers, p_data, 'rs-', label='powers')
#        ax.legend(loc='best')
#
#        ax = fig.add_subplot(212)
#        ax.plot(self.flux_amps, f_data, 'bs-', label='flux amps')
#        ax.legend(loc='best')


#        ax = fig.add_subplot(311)
#        ax.plot(self.powers, p_data, 'rs-', label='powers')
#        ax.legend(loc='best')
#
#        ax = fig.add_subplot(312)
#        ax.plot(self.flux_amps, f_data, 'bs-', label='flux amps')
#        ax.legend(loc='best')

#        ax = fig.add_subplot(313)
#        ax.plot(self.int_time, T_data, label='integration time')
#        ax.legend(loc='best')

#
#def calculate_max1D_cut(fids, axis):
#
#    split_fids = np.split(fids, fids.shape[axis], axis=axis)
#
#    proj_fids = []
#    for const_axis in split_fids:
#        proj_fids.append(np.max(const_axis.flatten()))
#
#    return np.array(proj_fids)

