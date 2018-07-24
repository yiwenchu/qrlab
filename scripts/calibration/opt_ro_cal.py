import numpy as np
from measurement import Measurement
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import mclient

from lib.math import fit
import time

from config import ENV_FILE
#ENV_FILE = r'C:\qrlab\scripts\calibration\ro_weight_func.npy'

INT_WINDOW = ('optimal', 'current', 'flat')

class Optimal_RO_Cal(Measurement):
    '''
    Sweep readout power and/or a sequence parameter to establish an optimal
    blind fidelity.  At each power and parameter an optimal demod window is
    (g-e) is chosen, then the blind fidelity is calculated.

    This class should be inherited by a class which provides a generate method.
    In generate experiments to be subtracted should alternate directly.

    The child class may also provide for updating the ideal parameters,
    updating the RO power is provided by the update_readout flag.  This should
    be done in a "resolve()" function, which may also do further unique
    analysis.

    Input:
    <infos> should contain the relevant qubit infos to set up the sequencer.
    <powers> should be the RO powers to be swept, in dBm.  Providing one power
        is acceptable.
    <swept_params> should be a list of the parameters swept in the sequence.
        Generate should provide a sequence twice as long as this list.
    <shots> the number of individual trajectories to take for each expt.
    <hist_steps> the number of pixels in each dimension of the BF histograms
    <update_readout> controls whether or not the optimal readout power, e and g
        projection axis (iq_e and iq_g), and optimal demod window are set at
        the end.

    Returns an array of blind fidelities, as well as the optimal parameter,
    readout power, iq_e and iq_g, and the optimal demod window for those
    settings.

    With a generate that only produces two elements (e.g. a pi pulse and no pi
    pulse), this is equivalent to histogram_calibration, and is an upgrade to
    the traditional high power readout contrast calibration.
    '''

    def __init__(self, infos=None, powers=None, swept_params=None, shots=1e5,
                hist_steps=26, update_readout=False, verbose_plots=False,
                plot_best=True, use_window='optimal',
                readout='readout', **kwargs):

        self.infos = infos
        self.powers = powers
        self.swept_params = swept_params
        self.shots = shots
        self.hist_steps = hist_steps
        self.use_window = use_window

        self.update_readout = update_readout
        self.verbose_plots = verbose_plots
        self.plot_best = plot_best

        self.num_powers = len(powers)
        self.num_params = len(swept_params)

        super(Optimal_RO_Cal, self).__init__(2*self.num_params,
                                            readout=readout,
                                            infos=self.infos, **kwargs)

        self.alz = self.instruments['alazar']
        self.ag_ro = self.readout_info.rfsource1
        self.ro_instrument = self.instruments[readout]

        self.data.create_dataset('powers', data=powers)
        self.data.create_dataset('swept_params', data=swept_params)

        self.blind_fids = self.data.create_dataset('blind_fids',
                                shape=[self.num_powers,
                                       self.num_params])

    def generate(self):
        raise Exception("Generate should be overwritten by a child class.")

    def resolve(self):
        '''
        This should also be overwritten by a child class. It provides a place
        to do further analysis or change parameter settings.
        '''
        pass

    def measure(self):
        '''
        Since this experiment generates large amounts of (mostly useless) raw
        data, most of the analysis and reduction is done interleaved with the
        measurement.
        '''
        self.alz.setup_channels()
        self.alz.setup_clock()
        self.alz.setup_trigger()

        self.save_settings()
        self.old_window = self.alz.load_weight_func(self.alz.get_weight_func())
        self.old_power = self.ag_ro.get_power()
        self.old_nsamples = self.alz.get_nsamples()
        self.old_nshots = self.alz.get_naverages()

        try:
            if not self.update_readout:
                env_file = self.alz.get_weight_func()
            else:
                env_file = ENV_FILE

            self.alz.set_weight_func('')

            self.fids = np.zeros((self.num_powers,self.num_params))

            self.best_fid = 0
            self.best_power = None
            self.best_param = None
            self.best_envelope = None
            self.best_IQe = [] # windowed traces
            self.best_IQg = []
            self.best_iqe = 0 # averaged values
            self.best_iqg = 0
            self.best_avgebuf = 0
            self.best_avggbuf = 0

            for pidx, p in enumerate(self.powers):
                #All of the orders here are important!  Don't muck with!
                self.ag_ro.set_power(p)

                self.stop_awgs()
                self.stop_funcgen()
                time.sleep(1)

                self.alz.setup_demod_shots(self.num_params*2*self.shots)

                do_load = (pidx==0)
                if not self.do_generate:
                    do_load = False
                self.play_sequence(load=do_load) #Only load the awg once.

                buf = self.alz.take_demod_shots(timeout=1e6)

                '''pts come in [[delay1e],[delay1g],[delay2e],[delay2g]...]'''
                for idx,param in enumerate(self.swept_params):
                    ebuf = buf[2*idx::2*self.num_params,:]
                    gbuf = buf[2*idx+1::2*self.num_params,:]

                    #TODO:  This code needs to be faster.  The analysis is
                    #quite slow with large datasets.  Consider numexpr?
                    #Find the average e and g trajectories -> int window
                    avg_ebuf = np.average(ebuf, axis=0)
                    avg_gbuf = np.average(gbuf, axis=0)
                    self.avg_ebuf = avg_ebuf
                    self.avg_gbuf = avg_gbuf

                    if self.use_window == 'optimal':
                        diff = (avg_ebuf - avg_gbuf)
                        diff = (np.real(diff) + 1j * np.imag(diff))
                        diff /= np.sum(np.abs(diff))
                    elif self.use_window == 'current':
                        diff = self.old_window
                    elif self.use_window == 'flat':
                        num_if_periods = self.alz.get_nsamples() / self.alz.get_if_period()
                        diff = np.array([1.0+1j,]*num_if_periods) / (np.sqrt(2.) * float(num_if_periods))
                    else:
                        Exception('use_window choices: optimal, current, flat')

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

                        # plot re/im trajectory
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.plot(np.real(self.avg_gbuf), label='re g')
                        ax.plot(np.imag(self.avg_gbuf), label='im g')
                        ax.plot(np.real(self.avg_ebuf), label='re e')
                        ax.plot(np.imag(self.avg_ebuf), label='im e')
                        ax.set_ylabel('re/im')
                        ax.legend()

                    #Following the analyis in Alazar_Daemon.get_IQ_rel()
                    #These are arrays of integrated traces.  final IQ points.
                    IQ_e =      np.inner(np.real(ebuf), r_env) + \
                           1j * np.inner(np.imag(ebuf), i_env)

                    IQ_g =      np.inner(np.real(gbuf), r_env) + \
                           1j * np.inner(np.imag(gbuf), i_env)

                    fid = blind_fidelity(IQ_e, IQ_g, plot=self.verbose_plots,
                                           verbose=False,
                                           num_steps=self.hist_steps)

                    self.fids[pidx,idx] = fid

                    if fid > self.best_fid:
                        self.best_fid = fid
                        self.best_power = p
                        self.best_param = param
                        self.best_envelope = diff

                        self.best_avgebuf = avg_ebuf
                        self.best_avggbuf = avg_gbuf

                        #This is the average final IQ point of e, g.
                        self.best_iqe = np.average(IQ_e)
                        self.best_iqg = np.average(IQ_g)

                        self.best_IQe = IQ_e
                        self.best_IQg = IQ_g

                print '\tFinished %0.2f dBm: best fidelity = %0.3f\n' %\
                           (p, np.max(np.array(self.fids[pidx,:]).flatten()))

            self.powers = np.array(self.powers)
            self.flux_amps = np.array(self.swept_params)

            self.plot_results()

            if self.update_readout:
                print 'Setting new power, iqe, iqg, integration window.'
                self.old_power = self.best_power #Overwrites for finally block
                self.old_nsamples = self.alz.get_nsamples()

                self.ro_instrument.set_IQg(self.best_iqg)
                self.ro_instrument.set_IQe(self.best_iqe)

                np.save(ENV_FILE, self.best_envelope)

            self.resolve()

            return self.best_fid, self.best_power, self.best_param, \
                   self.best_envelope, self.best_iqg, self.best_iqe

        finally:
            self.alz.set_nsamples(self.old_nsamples)
            self.ag_ro.set_power(self.old_power)
            self.alz.set_weight_func(env_file)
            self.alz.set_naverages(self.old_nshots)

    def plot_results(self):
        xs = self.powers
        ys = self.swept_params
        values = self.fids[:]

        #Plot the sweep results
        if len(xs) > 1 and len(ys) > 1:
            xx, yy = set_2Daxes_pcolor(xs,ys)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.pcolor(xx, yy, np.transpose(values),
                         vmax=np.nanmax(values),vmin=np.nanmin(values),cmap='jet')
            fig.colorbar(im)
            #        #ax.set_title(txt)
            ax.set_xlim(xx.min()), xx.max()
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(r'RO Power (dBm)')
            ax.set_ylabel(r'Sequence Parameters')
            fig.canvas.draw()

        elif len(ys) > 1:
            title = "Blind fidelity, power = %0.2f dBm" % xs[0]
            plt.figure()
            plt.plot(ys, values.flatten(), 'ro-')
            plt.title(title)
            plt.show()

        elif len(xs) > 1:
            title = "Blind fidelity."
            title += "\nBest power: %0.2f dBm, best fidelity: %0.3f" % \
                        (self.best_power,self.best_fid)
            plt.figure()
            plt.plot(xs, values.flatten(), 'ro-')
            plt.title(title)
            plt.show()

        #PLot the specifics of the best result
        if self.plot_best:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.real(self.best_avgebuf),np.imag(self.best_avgebuf))
            ax.plot(np.real(self.best_avggbuf),np.imag(self.best_avggbuf))
            fig.canvas.draw()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.real(self.best_envelope))
            ax.plot(np.imag(self.best_envelope))
            fig.canvas.draw()

            blind_fidelity(self.best_IQe, self.best_IQg,
                           plot=True, verbose=False,
                           num_steps=self.hist_steps)


def set_2Daxes_pcolor(xs, ys):
    '''
        xs and data are 1D arrays specifying the mesh.
    '''
    # adjust the 1D range by adding another element and shifting by half a period
    dx = xs[-1] - xs[-2]
    x_range = np.concatenate((xs, [xs[-1] + dx]))
    x_range = x_range - dx/2.0

    dy = ys[-1] - ys[-2]
    y_range = np.concatenate((ys, [ys[-1] + dy]))
    y_range = y_range - dy/2.0

    # generate mesh
    return np.meshgrid(x_range, y_range)



def get_limits(series, scale):
    ave = 0.5 * (np.max(series) + np.min(series))
    delta = 0.5 * (np.max(series) - np.min(series))
    return ave + scale * delta, ave - scale * delta

def determine_histogram(IQs1, IQs2, num_steps=26):
    '''
        In order to perform blind fidelity experiments for two datasets, we need
        to generate on global set of histogram bins taking into consideration
        both datasets.

        <num_steps> is the number of pixels in each dimension in the histogram
        to be generated.

        Outputs:
            - counts: histograms counts for each IQ dataset
            - xbins, ybins: global histogram bins
    '''

    IQs = np.concatenate((IQs1, IQs2))
    IQs.flatten()

    hist_xmax, hist_xmin = get_limits(IQs.real, 1.1)
    hist_ymax, hist_ymin = get_limits(IQs.imag, 1.1)

    xbins = np.linspace(hist_xmin, hist_xmax, num_steps)
    ybins = np.linspace(hist_ymin, hist_ymax, num_steps)

    counts = [0, 0]
    for i, expt in enumerate([IQs1, IQs2]):
        counts[i], xedges, yedges = np.histogram2d(expt.real, expt.imag,
                                                    bins=[xbins, ybins])
    return np.array(counts), xbins, ybins

def calculate_bf(counts):
    '''Calculates the blind fidelity between a pair of arrays'''
    counts1, counts2 = counts
    diff_counts = np.abs(counts1 - counts2)
    fid = np.sum(diff_counts) / (2.0 * np.sum(counts1))
    return fid, diff_counts

def blind_fidelity(IQ1, IQ2, plot=True, verbose=False, num_steps=26):
    '''
    Takes arrays of e and g integrated points, histograms, and
    returns the blind fidelity between the two.
    '''

    counts, xbins, ybins = determine_histogram(IQ1, IQ2, num_steps=num_steps)
    counts = [np.rot90(counts[0]), np.rot90(counts[1])]

    fid, diff_counts = calculate_bf(counts)
    if verbose:
        print 'blind fidelity: %0.3f' % fid

    if plot:
        import matplotlib.gridspec as gridspec
        fig = plt.figure()
        plt.suptitle('Blind fidelity histograms, blind fidelity = %0.3f'%fid)
        gs = gridspec.GridSpec(2,2)
        ax_1 = fig.add_subplot(gs[0,0])
        ax_2 = fig.add_subplot(gs[0,1])
        ax_3 = fig.add_subplot(gs[1,:])

        extent = (xbins[0], xbins[-1], ybins[0], ybins[-1])

        ax_1.imshow(counts[0], interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|g>')
        ax_2.imshow(counts[1], interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|e>')
        ax_3.imshow(diff_counts, interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|e> -|g>')

        ax_1.set_title('|g>')
        ax_2.set_title('|e>')
        ax_3.set_title('|e> - |g>')
    return fid