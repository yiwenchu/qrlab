from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
from lib.plotting_support import plotting
import time
import objectsharer as objsh

SPEC   = 0
POWER  = 1

LOR = 'lorentzian'
GAUSS = 'gaussian'
AMP = 'amplitude'
PHASE = 'phase'

class CavityTransmission(Measurement1D):
    '''
    Perform cavity transmission.

    The frequency of <cavity_rfsource> will be swept over <freqs> and
    different read-out powers <ro_powers> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.
    '''

    def __init__(self, freqs, ro_powers, readout='readout',
                 plen=10000, amp=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1,
                 plot_type=None,
                 analyze_data=True,
                 **kwargs):

        self.ro_powers = np.array(ro_powers)
        self.freqs = np.array(freqs)
        self.plen = plen
        self.amp = amp
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.analyze_data = analyze_data

        if plot_type is None:
            if len(ro_powers) > len(freqs):
                plot_type = POWER
            else:
                plot_type = SPEC
        self.plot_type = plot_type

        kwargs['print_progress'] = False
        super(CavityTransmission, self).__init__(1, analyze_data=analyze_data, **kwargs)

        self.data.create_dataset('powers', data=ro_powers)
        self.data.create_dataset('freqs', data=freqs)

        # I want to use self.avg data, but data_changed_cb gets angry
        # essentially the dataset gets overwritten everytime.
        # the result is that I can't save data as it comes in--its only saveable after the iteration is done
        self.IQs = self.data.create_dataset('avg', shape=[len(ro_powers),len(freqs)],
                                             dtype=np.complex)
        self.amps = self.data.create_dataset('avg_pp', shape=[len(ro_powers),len(freqs)],
                                             dtype=np.float)

    def generate(self):
        s = Sequence()

        s.append(Trigger(250))
        s.append(Combined([
            Constant(250, 1, chan=self.readout_info.readout_chan, repeat=int(self.plen / 250.)),
            Constant(250, 1, chan=self.readout_info.acq_chan, repeat=int(self.plen / 250.)),
        ]))

        s = self.get_sequencer(s)
        seqs = s.render(debug=False)
        return seqs

    def measure(self):
        self.default_ro_freq = self.readout_info.rfsource1.get_frequency()
        self.default_lo_freq = self.readout_info.rfsource2.get_frequency()
        self.default_power = self.readout_info.rfsource1.get_power()

        try:
            self.setup_measurement()

            # manually set up data
            self.avg_data = None
            self.pp_data = None
            self.shot_data = None
            started = False

            alz = self.instruments['alazar']
            weight_func = alz.get_weight_func()
            alz.set_weight_func('')

            old_nsamp = alz.get_nsamples()
            alz.set_nsamples(51200)


            IQg = self.readout_info.IQg
            IQe = self.readout_info.IQe
            self.readout_info.IQg = None
            self.readout_info.IQe = None

            for ipower, power in enumerate(self.ro_powers):
                self.readout_info.rfsource1.set_power(power)
                time.sleep(self.pow_delay)

                xs = []
                IQs = []

                try:
                    for ifreq, freq in enumerate(self.freqs):

                        if freq != self.readout_info.rfsource1.get_frequency():
                            if_Hz = 1.0/alz.get_if_period() * 1e9
                            self.readout_info.rfsource1.set_frequency(freq)
                            self.readout_info.rfsource2.set_frequency(freq + if_Hz)
                            time.sleep(self.freq_delay)

                        if not started:
                            self.start_awgs()
                            self.start_funcgen()
                            started = True

                        ret = self.acquisition_loop(alz, fast=(self.cyclelen==1))[0]

                        xs.append(freq / 1e6)
                        IQs.append(ret)

                        print 'F = %.03f MHz --> amp = %.1f, angle = %.01f' % \
                                (freq / 1e6, np.abs(ret), np.angle(ret, deg=True),)

                        self.xs = np.array(xs)
                        self.update(np.abs(IQs))

                    self.IQs[ipower,:] = np.array(IQs)
                    self.amps[ipower,:] = np.abs(np.array(IQs))
                finally:
                    alz.set_nsamples(old_nsamp)
                    alz.set_weight_func(weight_func)

                    self.readout_info.IQg = IQg
                    self.readout_info.IQe = IQe

            if self.analyze_data:
                self.analyze()

            if self.savefig:
                self.save_fig()
            return self.freqs, self.IQs[:]
        finally:
            self.readout_info.rfsource1.set_frequency(self.default_ro_freq)
            self.readout_info.rfsource2.set_frequency(self.default_lo_freq)
            self.readout_info.rfsource1.set_power(self.default_power)

    def analyze(self):
        self.residuals = True
        fig = self.get_figure()
        fig.axes[0].clear()
        self.fit_params = analyze(self.ro_powers, self.freqs, self.IQs[:], plot_type=self.plot_type,
                fig=fig)
        self.save_fig()

def analyze(powers, freqs, IQs, plot_type=SPEC, fit_type=LOR, amp_phase=AMP,
            fig=None):

    if fig is None:
        fig = plt.figure()

    # convert frequency to MHz
    if freqs[0] > 1e9:
        freqs = freqs / 1e6

    fit_params = []
    if plot_type==SPEC:
        for ipower, power in enumerate(powers):
            power_label = 'power = %0.2f dBm; ' % (power)
            result, fig = fit_spectroscopy(freqs, IQs[ipower],
                           amp_phase=amp_phase, fit_type=LOR, fig=fig,
                           label=power_label, plot_residuals=True)
            fit_params.append(result.params)

    if plot_type == POWER:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for ifreq, freq in enumerate(freqs):
            ampdata = np.abs(IQs)[:,ifreq]
            phasedata = np.angle(IQs, deg=True)[:,ifreq]
            ax1.plot(powers, ampdata, label='RF @ %.03f MHz'%(freq,))
            ax2.plot(powers, phasedata, label='RF @ %.03f MHz'%(freq,))
        ax1.legend(loc='best')
        ax2.legend(loc='best')

        ax1.set_ylabel('Intensity [AU]')
        ax2.set_ylabel('Angle [deg]')
        ax1.set_xlabel('Power [dB]')
        ax2.set_xlabel('Power [dB]')

    return fit_params


def fit_spectroscopy(freqs, IQs, amp_phase=AMP, fit_type=LOR,
                     fig=None, label='', plot_residuals=True):

    if fig is None:
        fig = plt.figure()
        if plot_residuals:
            from matplotlib import gridspec
            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
            fig.add_subplot(gs[0])
            fig.add_subplot(gs[1])
        else:
            fig.add_subplot(111)

    if freqs[0] > 1e9:
        freqs = freqs / 1e6

    if amp_phase == AMP:
        data = np.abs(IQs)
    elif amp_phase == PHASE:
        data = np.angle(IQs, deg=True)

    fit = fitter.Fitter(fit_type)
    result = fit.perform_lmfit(freqs, data, print_report=False, plot=False)
    datafit = fit.eval_func()

    x0 = result.params['x0']
    fit_label = label + 'center = %0.4f MHz,' % (x0)
    if fit_type == LOR:
        fit_label += ' width = %0.4f MHz' % (result.params['w'])
    elif fit_type == GAUSS:
        fit_label += ' sigma = %0.4f MHz' % (result.params['sigma'])

    fig.axes[0].plot(freqs, data, marker='s', ms=3, label='')
    fig.axes[0].plot(freqs, datafit, ls='-', ms=3, label=fit_label)

    fig.axes[0].legend(loc='best')
    if amp_phase == AMP:
        fig.axes[0].set_ylabel('Intensity [AU]')
    elif amp_phase == PHASE:
        fig.axes[0].set_ylabel('Phase [deg]')
    fig.axes[0].set_xlabel('Frequency [MHz]')

    if plot_residuals:
        fig.axes[1].plot(freqs, (data - datafit) / datafit *100.0, ls='-')
        fig.axes[1].set_ylabel('error [%]')
        fig.axes[1].set_xlabel('Frequency [MHz]')

    return result, fig

