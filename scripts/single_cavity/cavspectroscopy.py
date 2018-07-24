from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import objectsharer as objsh
import time

SPEC   = 0
POWER  = 1

def analysis(cav_disps, cav_freqs, ampdata, phasedata=None, plot_type=POWER, fig=None):
    if fig is None:
        fig = plt.figure()

    if plot_type == SPEC:
        for idisp, disp in enumerate(cav_disps):
            fig.axes[0].plot(cav_freqs/1e6, ampdata[idisp,:], label='Amp %.03f'%disp)
            fig.axes[1].plot(cav_freqs/1e6, phasedata[idisp,:], label='Amp %.03f'%disp)

        fs = cav_freqs
        amps = ampdata[0,:]
        f = fit.Lorentzian(fs, amps)
        h0 = np.max(amps)
        w0 = 2e6
        pos = fs[np.argmax(amps)]
        p0 = [np.min(amps), w0*h0, pos, w0]
        p = f.fit(p0)
        txt = 'Center = %.03f MHz' % (p[2]/1e6,)
        print 'Fit gave: %s' % (txt,)
#        plt.plot(fs/1e6, f.func(p, fs), label=txt)

        plt.legend()
        plt.ylabel('Intensity [AU]')
        plt.xlabel('Frequency [MHz]')

    if plot_type == POWER:
        ax1 = f.add_subplot(2,1,1)
        ax2 = f.add_subplot(2,1,2)
        for ifreq, freq in enumerate(cav_freqs):
            fig.axes[0].plot(cav_disps, ampdata[:,ifreq], label='RF @ %.03f MHz'%(freq/1e6,))
            fig.axes[1].plot(cav_disps, phasedata[:,ifreq], label='RF @ %.03f MHz'%(freq/1e6,))
        ax1.legend()
        ax2.legend()
        ax1.set_ylabel('Intensity [AU]')
        ax2.set_ylabel('Angle [deg]')
        ax1.set_xlabel('Power [dB]')
        ax2.set_xlabel('Power [dB]')

class CavSpectroscopy(Measurement1D):

    def __init__(self, cav_source, qubit_info, cav_info, cav_disps, cav_freqs, plot_type=None, delay=100, **kwargs):
        self.cav_source = cav_source
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.cav_disps = cav_disps
        self.cav_freqs = cav_freqs
        self.delay = delay

        if plot_type is None:
            if len(cav_disps) > len(cav_freqs):
                plot_type = POWER
            else:
                plot_type = SPEC
        self.plot_type = plot_type

        super(CavSpectroscopy, self).__init__(1, infos=(qubit_info, cav_info), **kwargs)
        self.data.create_dataset('disps', data=cav_disps)
        self.data.create_dataset('freqs', data=cav_freqs)
        self.ampdata = self.data.create_dataset('amplitudes', shape=[len(cav_disps),len(cav_freqs)])
        self.phasedata = self.data.create_dataset('phases', shape=[len(cav_disps),len(cav_freqs)])

    def generate(self, amp):
        s = Sequence()
        s.append(Trigger(250))
        s.append(Join([self.cav_info.rotate(amp, 0), Delay(self.delay)]))
        s.append(self.qubit_info.rotate(np.pi, 0))

        s.append(Combined([
            Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
            Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
        ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        if self.plot_seqs:
            s.plot_seqs(seqs)

        self.seqs = seqs
        return seqs

    def measure(self):
        # Generate and load sequences
        alz = self.instruments['alazar']

        for idisp, disp in enumerate(self.cav_disps):
            seqs = self.generate(disp)
            self.load(seqs)
            self.start_awgs()

            amps = []
            phases = []
            for freq in self.cav_freqs:
                self.cav_source.set_frequency(freq)
                time.sleep(0.05)

                alz.setup_avg_shot(alz.get_naverages())
                ret = alz.take_avg_shot(async=True)
                try:
                    while not ret.is_valid():
                        objsh.helper.backend.main_loop(100)
                except:
                    alz.set_interrupt(True)

                IQ = np.average(ret.get())
                amps.append(np.abs(IQ))
                phases.append(np.angle(IQ, deg=True))
                print 'F = %.03f GHz --> amp = %.1f, angle = %.01f' % (freq / 1e9, np.abs(IQ), np.angle(IQ, deg=True))

            self.ampdata[idisp,:] = amps
            self.phasedata[idisp,:] = phases

        self.analyze()

    def analyze(self):
        fig = self.create_figure()
        analysis(self.cav_disps, self.cav_freqs, self.ampdata, self.phasedata, self.plot_type, fig=fig)
