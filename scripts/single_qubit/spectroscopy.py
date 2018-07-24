import logging
from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
import time

SPEC   = 0
POWER  = 1

LOR = 'lorentzian'
GAUSS = 'gaussian'
AMP = 'amplitude'
PHASE = 'phase'
VOIGT = 'voigt'

class Spectroscopy(Measurement1D):
    '''
    Perform qubit spectroscopy.

    The frequency of <qubit_rfsource> will be swept over <freqs> and
    different read-out powers <ro_powers> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.

    Known keyword arguments/options:
    -   spec_gen (Spectroscopy.spec_gen) : string (Default: 'ag_ro')
        The name of the generator instrument used to do spec.

    -   use_marker (Spectroscopy.use_marker) : bool (Default: False)
        If True, we apply a constant 'on' marker as the spec pulse on the 
        marker channel
    def update(self, avg_data):
        data, bg_data = self.get_all_data(avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()
        fig.axes[1].clear()

        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, data, 'rs-', label='raw data')
            fig.axes[0].plot(self.xs, bg_data, 'bs-', label='background')
            fig.axes[1].plot(self.xs, data-bg_data, 'ks-', label='bg subtracted')
        else:
            fig.axes[0].plot(data, 'rs-', label='raw data')
            fig.axes[0].plot(bg_data, 'bs-', label='background')
            fig.axes[1].plot(data-bg_data, 'ks-', label='bg subtracted')

        fig.axes[0].legend(loc='best')
        fig.axes[1].legend(loc='best')

        fig.canvas.draw()
 of the info object.
        will throw an exception if plen == 0.

    -   spec_pulse_repeat (Spectroscopy.spec_pulse_repeat) : int (Default: 1)
        if > 1, repeat the pulse <N> times (for saving memory for long pulses)
        For pi-pulse spec, this option is ignored.
    '''

    def __init__(self, qubit_info, freqs, spec_params,
                 plen=50e3, amp=1, seq=None, postseq=None,
                 freq_delay=0.1,
                 fit_type=LOR, plot_type=None,
                 analyze_data=True, use_weight=False, use_IQge=False,
                 subtraction = False,
                 **kwargs):

        numpts = len(freqs)
        numseq = 1
        if subtraction:
            numseq = 2

        kwargs['print_progress'] = False
        self.ro_gen = kwargs.pop('ro_gen', 'ag_ro')
        self.use_marker = kwargs.pop('use_marker', False)
        self.spec_pulse_repeat = kwargs.pop('spec_pulse_repeat', 1)

        super(Spectroscopy, self).__init__(numseq, infos=qubit_info, **kwargs)

        self.qubit_info = qubit_info
        self.freqs = freqs
        self.qubit_rfsource, self.spec_power = spec_params

        self.plen = plen
        self.amp = amp

        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        
        self.postseq = postseq
        self.fit_type = fit_type
        self.use_weight = use_weight
        self.use_IQge = use_IQge
        self.subtraction = subtraction
        self.analyze_data = analyze_data

        # optional settings


        # create tasty tasty data
        self.data.create_dataset('freqs', data=freqs)
        self.IQs = self.data.create_dataset('avg', shape=[numpts],
                                             dtype=np.complex)
        self.reals = self.data.create_dataset('avg_pp', shape=[numpts],
                                             dtype=np.float)

    def generate(self):
        s = Sequence()

        if self.subtraction:
            plist = [1.0, 0.0]
        else:
            plist = [1.0]

        for amp in plist:
            s.append(self.seq)

            if not self.use_marker:
                if self.plen is not None and self.plen != 0:
                    chs = self.qubit_info.sideband_channels
                    pulse = Constant(self.plen, amp*self.amp, chan=chs[0])
                    s.append(Repeat(pulse, self.spec_pulse_repeat))
                else:
                    pulse = self.qubit_info.rotate(amp*np.pi, 0.0)
                    if self.spec_pulse_repeat > 1:
                        logging.info("If we do pi-pulse spec, \
                            pulse repetition is ignored.")
                    s.append(pulse)
            
            else:
                if self.plen is None or self.plen == 0:
                    raise ValueError("I'm afraid I can't let you do that, \
                        Dave. plen==0, really?")
                chs = self.qubit_info.marker['channel']
                ssbchs = self.qubit_info.sideband_channels
                pulse =Combined([Constant(self.plen, amp*1.0, chan=chs),
                    Constant(self.plen,amp*.0001,chan=ssbchs[0])])
                # print pulse.pulse_data.keys()
#                s.append(pulse)
                s.append(Repeat(pulse, self.spec_pulse_repeat))
                # print pulse.pulse_data.keys()
            if self.postseq:
                s.append(self.postseq)

            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def measure(self):
        self.old_freq = self.qubit_rfsource.get_frequency()

        alz = self.instruments['alazar']
        weight_func = alz.get_weight_func()

        if not self.use_weight:
            alz.set_weight_func('')
        IQg = self.readout_info.IQg
        IQe = self.readout_info.IQe

        xs = []
        IQs = []
        reals = []

        try:
            self.setup_measurement() #Stops AWGs, Fg loads seq
            self.avg_data = None
            self.pp_data = None
            self.shot_data = None
            started = not self.cyclelen==1 # False

            if not self.use_IQge:
                self.readout_info.IQg = 0.0
                self.readout_info.IQe = 0.0

            if self.spec_power != None:
                self.qubit_rfsource.set_power(self.spec_power)
            
            # we want to have the option to use either bricks or ags
            # but don't turn on pulse mode unless the generator should be pulsed!!
            if self.use_marker is True:
                if self.qubit_rfsource.get_type() != 'LabBrick_RFSource':
                    self.qubit_rfsource.set_pulse_on(True)

            # spectroscopy loop
            for ifreq, freq in enumerate(self.freqs):
                self.qubit_rfsource.set_frequency(freq)
                time.sleep(self.freq_delay)
                if not started:
                    self.start_awgs()
                    self.start_funcgen()
                    started = True

                ret = self.acquisition_loop(alz, fast=(self.cyclelen==1))

                if self.use_IQge:
                    real = self.complex_to_real(ret)
                else:
                    real = np.abs(ret)

                if self.subtraction:
                    real = - real[1] + real[0]
                    ret = - ret[1] + ret[0]
                else:
                    real = real[0]
                    ret = ret[0]

                xs.append(freq / 1e6)

                IQs.append(ret)
                reals.append(real)

                print 'F = %.03f MHz --> amp = %.1f, angle = %.01f, real = %0.1f' % \
                        (freq / 1e6, np.abs(ret), np.angle(ret, deg=True), real)

                self.xs = np.array(xs)
                self.update(np.array(reals))

                if self.subtraction:
                     self.stop_funcgen()
                     self.stop_awgs()

            self.IQs[:] = np.array(IQs)
            self.reals[:] = np.array(reals)

            if self.analyze_data:
                self.analyze()

            if self.savefig:
                self.save_fig()
            return self.freqs, self.IQs[:], self.reals[:]

        finally:
            #Reset the instruments
            self.qubit_rfsource.set_frequency(self.old_freq)

            if not self.use_weight:
                alz.set_weight_func(weight_func)
            if not self.use_IQge:
                self.readout_info.IQg = IQg
                self.readout_info.IQe = IQe

    def analyze(self):
        self.residuals = True
        fig = self.get_figure()
        fig.axes[0].clear()
        
        # let's not assume that any type of generator has the get_power func.
        try:
            ro_power = self.instruments[self.ro_gen].get_power()
        except:
            ro_power = None

        self.fit_params = analyze(self.freqs, self.IQs[:],
                                  reals=self.reals[:], fig=fig,
                                  sp_pow=self.spec_power,
                                  ro_pow=ro_power,
                                  fit_type=self.fit_type)

        self.center_freq = self.fit_params['x0'].value
        self.save_fig()
        return self.fit_params

    def get_ys(self, data=None):
        return data

def analyze(freqs, IQs, reals=None, fit_type=LOR, amp_phase=AMP,
            fig=None, sp_pow=None, ro_pow=None):

    if fig is None:
        fig = plt.figure()

    # convert frequency to MHz
    if freqs[0] > 1e9:
        freqs = freqs / 1e6
    
    if ro_pow != None:
        power_label = 'ro power = %0.2f dBm; ' % (ro_pow)
    else:
        power_label = 'ro power n/a; '
    
    if sp_pow is not None:
        power_label += 'sp power = %0.2f dBm; ' % (sp_pow)
    
    if reals is not None:
        data = reals
    else:
        data = np.abs(IQs)

    result, fig = fit_spectroscopy(freqs, data,
                   amp_phase=amp_phase, fit_type=fit_type, fig=fig,
                   label=power_label, plot_residuals=True)

    return result.params


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
        data = IQs #np.abs(IQs)
    elif amp_phase == PHASE:
        data = np.angle(IQs, deg=True)


    fit = fitter.Fitter(fit_type)
    p = fit.get_lmfit_parameters(freqs, data)
    result = fit.perform_lmfit(freqs, data, print_report=False, plot=False, p=p)
    datafit = fit.eval_func()

    x0 = result.params['x0']
    fit_label = label + 'center = %0.4f MHz,' % (x0)
    if fit_type == LOR:
        fit_label += ' width = %0.4f MHz' % (result.params['w'])
    elif fit_type == GAUSS:
        fit_label += ' sigma = %0.4f MHz' % (result.params['sigma'])
    elif fit_type == VOIGT:
        fg, fl, fv = fit.module.hwhm(result.params)
        fit_label += '\nw_g, w_L, w_V = %0.4f, %0.4f, %0.4f MHz' % (fg, fl, fv)


    fig.axes[0].plot(freqs, data, marker='s', ms=3, label='')
    fig.axes[0].plot(freqs, datafit, ls='-', ms=3, label=fit_label)

    fig.axes[0].legend(loc='best')
    if amp_phase == AMP:
        fig.axes[0].set_ylabel('Intensity [AU]')
    elif amp_phase == PHASE:
        fig.axes[0].set_ylabel('Phase [deg]')
    fig.axes[0].set_xlabel('Frequency [MHz]')

    if plot_residuals:
        fig.axes[1].plot(freqs, data-datafit, ls='-')
        fig.axes[1].set_ylabel('residual')
        fig.axes[1].set_xlabel('frequency [MHz]')

    return result, fig

