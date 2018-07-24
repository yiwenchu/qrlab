from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
import time
from PyQt4 import QtGui
import objectsharer as objsh

import fitting_programs as fp

from scripts.single_qubit.spectroscopy import Spectroscopy
SPEC   = 0
POWER  = 1

LOR = 'lorentzian'
GAUSS = 'gaussian'
AMP = 'amplitude'
PHASE = 'phase'

'''Spectroscopy should really be a measurement class.  This should
inherit spectroscopy and overwrite self.generate().'''

class StarkSpectroscopy(Spectroscopy):
    '''
    Perform qubit spectroscopy.

    The frequency of <spec_rfsource> will be swept over <freqs> and
    different read-out powers <ro_powers> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.
    '''

    def __init__(self, spec_info, stark_info,
                 freqs, spec_params, # format (spec_rfsource, ro_power)
                 ro_powers, plen=1000, stark_length=5000,
                 amp=1, amp_stark=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1, plot_type=None,
                 extra_info=None,
                 analyze_data=True, use_weight=False, use_IQge=False,
                 subtraction=False,
                 **kwargs):

        self.spec_info = spec_info
        self.stark_info = stark_info
        self.infos = [self.spec_info, self.stark_info]

        self.freqs = freqs
        self.spec_rfsource, self.spec_power = spec_params
        self.stark_length = stark_length

        self.ro_powers = ro_powers
        self.plen = plen
        self.amp = amp
        self.amp_stark = amp_stark
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.use_weight = use_weight
        self.use_IQge = use_IQge
        self.subtraction = subtraction

        self.analyze_data = analyze_data

        if plot_type is None:
            if len(ro_powers) > len(freqs):
                plot_type = POWER
            else:
                plot_type = SPEC
        self.plot_type = plot_type

        numpts = len(self.freqs)
        numseq = 1
        if self.subtraction:
            numseq = 2

        kwargs['print_progress'] = False
        kwargs['extra_info'] = extra_info


        super(StarkSpectroscopy, self).__init__(numseq,
                                        infos=self.infos, **kwargs)
        self.data.create_dataset('powers', data=ro_powers)
        self.data.create_dataset('freqs', data=freqs)

        self.IQs = self.data.create_dataset('avg', shape=[len(ro_powers),numpts],
                                             dtype=np.complex)
        self.amps = self.data.create_dataset('avg_pp', shape=[len(ro_powers),numpts],
                                             dtype=np.float)


    def generate(self):
        spec_r = self.spec_info.rotate
        stark_r = self.stark_info.rotate

        chs = self.spec_info.sideband_channels
        stark_ch = self.stark_info.sideband_channels

        stark_pulse = lambda length: Constant(length, self.amp_stark, chan=stark_ch[0])

        if self.subtraction:
            plist = [1.0, 0.0001]
        else:
            plist = [1.0]

        s = Sequence(self.seq)

        s.append(stark_pulse(self.stark_length))

        for amp in plist:
            if self.plen is not None:
                pulse = Constant(amp*self.plen, self.amp, chan=chs[0])
            else:
                pulse = spec_r(amp*np.pi, 0.0)

            s.append(Combined([
                stark_pulse(pulse.get_length()),
                pulse]))

            s.append(Delay(20))
            if self.postseq:
                s.append(self.postseq)

            s.append(self.get_readout_pulse())

#            s.append(Combined([
#                Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
#                Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
#            ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs



    def analyze(self):
        from scripts.single_qubit.spectroscopy import analyze
        self.residuals = True
        fig = self.get_figure()
        fig.axes[0].clear()
        self.fit_params = analyze(self.ro_powers, self.freqs, self.IQs[:],
                                  amps=self.amps[:], plot_type=self.plot_type, fig=fig,
                                  sp_pow = self.spec_power)
        self.save_fig()
        return self.fit_params

#def analyze(powers, freqs, IQs, amps=None, plot_type=SPEC, fit_type=LOR, amp_phase=AMP,
#            fig=None, sp_pow=None):
#
#    if fig is None:
#        fig = plt.figure()
#
#    # convert frequency to MHz
#    if freqs[0] > 1e9:
#        freqs = freqs / 1e6
#
#    if plot_type==SPEC:
#        for ipower, power in enumerate(powers):
#            power_label = 'ro power = %0.2f dBm; ' % (power)
#            if sp_pow is not None:
#                power_label += 'sp power = %0.2f dBm; ' % (sp_pow)
#            if amps is not None:
#                data = amps
#            else:
#                data = np.abs(IQs)
#            result, fig = fit_spectroscopy(freqs, data[ipower],
#                           amp_phase=amp_phase, fit_type=LOR, fig=fig,
#                           label=power_label, plot_residuals=True)
#
#    if plot_type == POWER:
#        ax1 = fig.add_subplot(211)
#        ax2 = fig.add_subplot(212)
#        for ifreq, freq in enumerate(freqs):
#            ampdata = np.abs(IQs)[:,ifreq]
#            phasedata = np.angle(IQs, deg=True)[:,ifreq]
#            ax1.plot(powers, ampdata, label='RF @ %.03f MHz'%(freq,))
#            ax2.plot(powers, phasedata, label='RF @ %.03f MHz'%(freq,))
#        ax1.legend(loc='best')
#        ax2.legend(loc='best')
#
#        ax1.set_ylabel('Intensity [AU]')
#        ax2.set_ylabel('Angle [deg]')
#        ax1.set_xlabel('Power [dB]')
#        ax2.set_xlabel('Power [dB]')
#
#    return result.params
#
#
#def fit_spectroscopy(freqs, IQs, amp_phase=AMP, fit_type=LOR,
#                     fig=None, label='', plot_residuals=True):
#
#    if fig is None:
#        fig = plt.figure()
#        if plot_residuals:
#            from matplotlib import gridspec
#            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
#            fig.add_subplot(gs[0])
#            fig.add_subplot(gs[1])
#        else:
#            fig.add_subplot(111)
#
#    if freqs[0] > 1e9:
#        freqs = freqs / 1e6
#
#    if amp_phase == AMP:
#        data = IQs #np.abs(IQs)
#    elif amp_phase == PHASE:
#        data = np.angle(IQs, deg=True)
#
#    fit = fitter.Fitter(fit_type)
#    result = fit.perform_lmfit(freqs, data, print_report=False, plot=False)
#    datafit = fit.test_values(freqs, p=result.params)
#
#    x0 = result.params['x0']
#    fit_label = label + 'center = %0.4f MHz,' % (x0)
#    if fit_type == LOR:
#        fit_label += ' width = %0.4f MHz' % (result.params['w'])
#    elif fit_type == GAUSS:
#        fit_label += ' sigma = %0.4f MHz' % (result.params['sigma'])
#
#    fig.axes[0].plot(freqs, data, marker='s', ms=3, label='')
#    fig.axes[0].plot(freqs, datafit, ls='-', ms=3, label=fit_label)
#
#    fig.axes[0].legend(loc='best')
#    if amp_phase == AMP:
#        fig.axes[0].set_ylabel('Intensity [AU]')
#    elif amp_phase == PHASE:
#        fig.axes[0].set_ylabel('Phase [deg]')
#    fig.axes[0].set_xlabel('Frequency [MHz]')
#
#    if plot_residuals:
#        fig.axes[1].plot(freqs, (data - datafit) / datafit *100.0, ls='-')
#        fig.axes[1].set_ylabel('error [%]')
#        fig.axes[1].set_xlabel('Frequency [MHz]')
#
#    return result, fig

