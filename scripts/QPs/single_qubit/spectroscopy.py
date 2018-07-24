from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import time
import objectsharer as objsh

SPEC   = 0
POWER  = 1

class Spectroscopy(Measurement1D):
    '''
    Perform qubit spectroscopy.

    The frequency of <qubit_rfsource> will be swept over <q_freqs> and
    different read-out powers <ro_powers> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.
    '''

    def __init__(self, qubit_rfsource, qubit_info, q_freqs, ro_powers,
                 plen=1000, amp=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1, plot_type=None,
                 **kwargs):
        self.qubit_rfsource = qubit_rfsource
        self.qubit_info = qubit_info
        self.ro_powers = ro_powers
        self.q_freqs = q_freqs
        self.plen = plen
        self.amp = amp
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        if plot_type is None:
            if len(ro_powers) > len(q_freqs):
                plot_type = POWER
            else:
                plot_type = SPEC
        self.plot_type = plot_type

        super(Spectroscopy, self).__init__(1, infos=qubit_info, **kwargs)
        self.data.create_dataset('powers', data=ro_powers)
        self.data.create_dataset('freqs', data=q_freqs)
        self.ampdata = self.data.create_dataset('amplitudes', shape=[len(ro_powers),len(q_freqs)])
        self.phasedata = self.data.create_dataset('phases', shape=[len(ro_powers),len(q_freqs)])

    def generate(self):
        s = Sequence(self.seq)
        chs = self.qubit_info.sideband_channels
        s.append(Constant(self.plen, self.amp, chan=chs[0]))
        if self.postseq:
            s.append(self.postseq)

        s.append(Combined([
            Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
            Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
        ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        
        if self.plot_seqs:
            s.plot_seqs(seqs)
            
        return seqs

    def measure(self):
        alz = self.instruments['alazar']
        alz.set_interrupt(False)

        # Generate and load sequences
        seqs = self.generate()
        self.load(seqs)
        self.start_awgs()

        for ipower, power in enumerate(self.ro_powers):
            self.readout_info.rfsource1.set_power(power)
            time.sleep(self.pow_delay)

            amps = []
            phases = []
            for freq in self.q_freqs:
                self.qubit_rfsource.set_frequency(freq)
                time.sleep(self.freq_delay)

                alz.setup_avg_shot(alz.get_naverages())
                ret = alz.take_avg_shot(async=True)
                try:
                    while not ret.is_valid():
                        objsh.helper.backend.main_loop(100)
                except Exception, e:
                    alz.set_interrupt(True)
                    print 'Error: %s' % (str(e), )
                    return
                IQ = np.average(ret.get())
                amps.append(np.abs(IQ))
                phases.append(np.angle(IQ, deg=True))
                print 'F = %.03f GHz --> amp = %.1f, angle = %.01f' % (freq / 1e9, np.abs(IQ), np.angle(IQ, deg=True))

            self.ampdata[ipower,:] = amps
            self.phasedata[ipower,:] = phases

        self.analyze()

    def analyze(self):
        f = plt.figure()

        if self.plot_type == SPEC:
            for ipower, power in enumerate(self.ro_powers):
                plt.plot(self.q_freqs/1e6, self.ampdata[ipower,:], label='Power %.01f dB'%power)

            fs = self.q_freqs
            amps = self.ampdata[0,:]
            f = fit.Lorentzian(fs, amps)
            h0 = np.max(amps)
            w0 = 2e6
            pos = fs[np.argmax(amps)]
            p0 = [np.min(amps), w0*h0, pos, w0]
            p = f.fit(p0)
            txt = 'Center = %.03f MHz' % (p[2]/1e6,)
            print 'Fit gave: %s' % (txt,)
            plt.plot(fs/1e6, f.func(p, fs), label=txt)

            plt.legend()
            plt.ylabel('Intensity [AU]')
            plt.xlabel('Frequency [MHz]')

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
