from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import time
from PyQt4 import QtGui
import objectsharer as objsh

import fitting_programs as fp

SPEC   = 0
POWER  = 1

class SpectroscopyFastFlux(Measurement1D):
    '''
    Flux voltage goes
                _ _ _ _ __________
               |   |   |       |  |________
    ___________|___|___|
    t_target        t_base
    target_flux     base_flux     ro_flux

    while the qubit tone is a fixed delay before the measurement

    __________/\__________________

    with the center of the qubit pulse offset from the target-base step
    by times 'Delays'.  Negative and the center of the qubit pulse is before the
    voltage switches from target to base.

    This is designed to look at the arrival of the flux pulse at base.

    '''
    def __init__(self, qubit_info, q_freqs, pulse_delays, qubit_rfsource,
                 flux_chan=None, target_flux=0, base_flux=0, ro_flux=0,
                 t_target=400, t_base=150, seq=None, postseq=None,
                 freq_delay=0.1,
                 do_analyze=True,
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

        self.xs = pulse_delays

        self.do_analyze = do_analyze

        kwargs['print_progress'] = False
        super(SpectroscopyFastFlux, self).__init__(len(self.pulse_delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('pulse delays', data=self.pulse_delays)
        self.data.create_dataset('freqs', data=self.q_freqs)
        self.IQs = self.data.create_dataset('IQs', shape=[len(self.q_freqs),len(self.pulse_delays)],
                                                          dtype=np.complex)

    # only the pi pulse moves here--the measurement occurs at a variable time
    # after the pi pulse.
    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate

        flux_pulse = lambda duration, flux_level: Constant(duration, flux_level, chan=self.flux_chan)

        flux_waveform = Join([
                            flux_pulse(self.t_target, self.target_flux),
                            flux_pulse(self.t_base, self.base_flux),
                        ])

        q_pi = r(np.pi,0)
        q_length = q_pi.get_length()
        center_timing = int(self.t_target - q_length / 2.0)
        for delay in self.pulse_delays:
            s.append(self.seq)

            # generate qubit pulse with appropriate delays
            qubit_waveform = Join([
                Delay(center_timing+delay),
                q_pi
            ])

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

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def measure(self):

        alz = self.instruments['alazar']
        weight_func = alz.get_weight_func()
        alz.set_weight_func('')

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
        finally:
            alz.set_weight_func(weight_func)

        if self.savefig:
            self.save_fig()

        return self.q_freqs, self.pulse_delays, self.IQs[:]

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
