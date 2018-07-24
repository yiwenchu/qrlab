from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import pulseseq.sequencer_subroutines as ss
#from lib.math import fit
import time
from PyQt4 import QtGui
import objectsharer as objsh

import fitting_programs as fp

SPEC   = 0
POWER  = 1

class SpectroscopyFastFlux(Measurement1D):
    '''
    Flux voltage goes
    _____               ____________
         |             |            |________
         |_____________|
            t_target        t_base
    base    target_flux     base_flux     ro_flux

    while the qubit tone goes

    __________/\  /\  /\__________________
                \/  \/

    with the center of the qubit pulse offset from the target-base step
    by times 'Delays'.  Negative and the center of the qubit pulse is before the
    voltage switches from target to base.

    This is designed to look at the arrival of the flux pulse at base.

    '''
    def __init__(self, qubit_info, q_freqs, pulse_delays, qubit_rfsource,
                 flux_chan=None, target_flux=0, base_flux=0, ro_flux=0,
                 t_target=400, t_base=150, flux_delay=0,
                 seq=None, postseq=None,
                 freq_delay=0.1,
                 do_analyze=True, kernel_path = None,
                 **kwargs):

        self.qubit_info = qubit_info
        self.q_freqs = q_freqs
        self.pulse_delays = pulse_delays

        self.qubit_rfsource = qubit_rfsource
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.flux_chan = flux_chan
        self.target_flux = target_flux
        self.base_flux = base_flux
        self.ro_flux = ro_flux
        self.t_target = t_target
        self.t_base = t_base

        self.pre_ro_flux = 1
        self.post_ro_flux = 2000

        self.flux_delay = flux_delay

        self.xs = pulse_delays

        self.do_analyze = do_analyze
        self.kernel_path = kernel_path

        kwargs['print_progress'] = False
        super(SpectroscopyFastFlux, self).__init__(len(self.pulse_delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('pulse delays', data=self.pulse_delays)
        self.data.create_dataset('freqs', data=self.q_freqs)
        self.IQs = self.data.create_dataset('IQs', shape=[len(self.q_freqs),len(self.pulse_delays)],
                                                          dtype=np.complex)
        self.realdata = self.data.create_dataset('realdata', shape=[len(self.q_freqs),len(self.pulse_delays)],
                                                          dtype=np.float)

    # only the pi pulse moves here--the measurement occurs at a variable time
    # after the pi pulse.
    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate
        qb_chan = r(0,0).get_channels()[0]

        flux_pulse = lambda duration, flux_level: Constant(duration, flux_level, chan=self.flux_chan)

        init_base_flux_len = 1000
        flux_waveform = Join([
                            flux_pulse(init_base_flux_len, self.base_flux), #JB
                            flux_pulse(self.t_target, self.target_flux),
                            flux_pulse(self.t_base, self.base_flux),
                        ])
        flux_len = flux_waveform.get_length()

        q_pi = r(np.pi,0)
        q_length = q_pi.get_length()

#        center_timing = int(self.t_target - q_length / 2.0)
#        center_timing = int(init_base_flux_len+self.t_target - q_length / 2.0)
        #This timing should center the pi pulse at the end of the t_target flux.

        # NEW TIMING: 0 delay aligns the flux step with the start of the pi pulse
        # i.e. the earliest time after a flux step the pi pulse will be useful
        initial_timing = int(init_base_flux_len + self.t_target)
        for delay in self.pulse_delays:
            s.append(self.seq)

            print flux_len, initial_timing, delay, q_length
            print flux_len-(initial_timing+delay+q_length)
            # generate qubit pulse with appropriate delays
            qubit_waveform = Join([
                Constant(initial_timing+delay, 0, chan=qb_chan),
                q_pi,
#                Constant(int(flux_waveform.get_length()-(initial_timing+delay+q_length)), 0, chan=qb_chan)
            ])

#            print qubit_waveform.get_length(), flux_waveform.get_length()

            s.append(Combined([
                flux_waveform, qubit_waveform
                ], align=ALIGN_LEFT))

            s.append(flux_pulse(self.pre_ro_flux, self.ro_flux))
            s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                    flux_pulse(self.readout_info.pulse_len, self.ro_flux)
                    ]))
            s.append(flux_pulse(self.post_ro_flux, self.ro_flux))

            s.append(flux_pulse(250, self.base_flux))

        s = self.get_sequencer(s, fast_flux=[1, self.flux_delay])
        seqs = s.render()
        seqs = self.post_process_sequence(seqs, flatten=True, kernel_path=self.kernel_path)

        return seqs

    def measure(self):

        alz = self.instruments['alazar']
        weight_func = alz.get_weight_func()
        #alz.set_weight_func('')

        self.setup_measurement_data() # ugly, this will get overwritten everytime

        try:
            for ifreq, freq in enumerate(self.q_freqs):
                self.qubit_rfsource.set_frequency(freq)
                time.sleep(self.freq_delay)

                self.setup_measurement()
                self.do_generate = False # only load once

                alz = self.instruments['alazar']
                ret = self.acquisition_loop(alz) # calls update function

                self.IQs[ifreq,:] = ret
                self.realdata[ifreq,:] = self.complex_to_real(ret)
        finally:
            alz.set_weight_func(weight_func)

            if self.savefig:
                self.save_fig()

        return self.q_freqs, self.pulse_delays, self.IQs[:], self.realdata[:]

    def analyze(self):
        self.fig = self.get_figure()
        ax_data, ax_res = self.fig.axes

        if self.plot_type == SPEC:
            for ipower, power in enumerate(self.ro_powers):
#                ax_data.plot(self.q_freqs/1e6, self.ampdata[ipower,:],
#                         label='readout power %.01f dB'%power)

                fs = self.q_freqs
                amps = self.ampdata[0,:]
                f = fp.Lorentzian(x_data=fs, y_data=amps)
                result, params, y_final = f.fit(plot=False)

                f0 = params['position'].value / 1e6

                txt = 'Center = %.03f MHz at %0.01f dBm' % (f0, power)
                print 'Fit gave: %s' % (txt,)
                ax_data.plot(fs/1e6, y_final, label=txt)

                ax_data.legend()
                ax_data.set_ylabel('Intensity [AU]')
                plt.xlabel('Frequency [MHz]')
                title = 'Pulsed spectroscopy: '
#                title += 'Center = %0.03f MHz, ' % f0
                if self.instruments['qubit_yoko'] is not None:
                    qubit_yoko = self.instruments['qubit_yoko']
#                    title += 'V_yoko = %0.03f, ' % qubit_yoko.get_voltage()
                if self.qubit_rfsource is not None:
                    title += 'spec %0.02f dBm' % self.spec_power
#                title += 'ro %0.02f dBm' % power
                ax_data.set_title(title)

                ax_res.plot(fs/1e6, y_final - self.ampdata[ipower,:])

        if self.plot_type == POWER:
            ax1 = f.add_subplot(2,1,1)
            ax2 = f.add_subplot(2,1,2)
            for ifreq, freq in enumerate(self.q_freqs):
                ax1.plot(self.ro_powers, self.ampdata[:,ifreq], label='RF @ %.03f MHz'%(freq/1e6,))
                ax2.plot(self.ro_powers, self.phasedata[:,ifreq], label='RF @ %.03f MHz'%(freq/1e6,))
            ax1.legend()
            ax2.legend()
            ax1.set_ylabel('Intensity [AU]')
            ax2.set_ylabel('Angle [deg]')
            ax1.set_xlabel('Power [dB]')
            ax2.set_xlabel('Power [dB]')

        self.save_fig()

def find_centers(delays, freqs, iqs):

    centers = np.zeros(len(delays))

    for i in np.arange(len(delays)):

        f = fp.Gaussian(x_data=freqs/1e9, y_data=iqs[:,i])
        result, params, y_data = f.fit(plot=False)

        centers[i] = params['position'].value

    return centers

def find_params(delays, freqs, iqs):
    params = []

    for i in np.arange(len(delays)):

        f = fp.Gaussian(x_data=freqs/1e9, y_data=iqs[:,i])
        result, p, y_data = f.fit(plot=False)

        params.append(p)

    return params

def delay_cuts(delays, freqs, iqs):

    params_list = []
    for i in np.arange(len(delays)):

        f = fp.Gaussian(x_data=freqs/1e9, y_data=iqs[:,i])
        result, params, y_data = f.fit(plot=False)

        params_list.append(params)

    return params_list

def frequency_cut(select_freqs, delays, freqs, iqs, freq_window=None,
                  scale=False, exp_fit=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle('fast flux spectroscopy frequency cuts')

    if not type(select_freqs) in [list, np.ndarray]:
        select_freqs = [select_freqs]

    for select_freq in select_freqs:
        freq_index = np.argmin(np.abs(select_freq - freqs))
        data_freq = iqs[freq_index,:]

        if freq_window is not None:
            avg_indices = np.where(np.abs(select_freq - freqs) < freq_window)[0]
            data_freq = np.average(iqs[avg_indices,:], axis=0)

        if exp_fit:
            f = fp.ErrorFunctionExp(x_data=delays, y_data=data_freq)
        else:
            f = fp.ErrorFunction(x_data=delays, y_data=data_freq)
        result, params, fit_vals = f.fit(plot=False)

        fp.report_fit_params(params)

        if scale:
            fit_offset = params['offset']
            fit_amp = params['amplitude']
            data_freq -= fit_amp
            data_freq /= np.abs(fit_amp)

            if exp_fit:
                f = fp.ErrorFunctionExp(x_data=delays, y_data=data_freq)
            else:
                f = fp.ErrorFunction(x_data=delays, y_data=data_freq)
            result, params, fit_vals = f.fit(plot=False)

        txt = 'frequency cut: %0.3f' % (select_freq)
        ax.plot(delays, data_freq, marker='s', label=txt)

        txt = 'fit position: %0.3f, fit scale: %0.3f' % (params['position'], params['x_scale'])
        ax.plot(delays, fit_vals, linestyle='-', label=txt)
    ax.legend(loc='best')






