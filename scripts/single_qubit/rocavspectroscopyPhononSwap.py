from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import objectsharer as objsh
import time

SPEC   = 0
POWER  = 1

def analysis(powers, freqs, ampdata, phasedata=None, plot_type=POWER, fig=None, txt =''):
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)
    
    
    for ipower, power in enumerate(powers):
        txt += 'Power %.02f dB'%power
        fig.axes[0].plot(freqs/1e6, ampdata[ipower,:], label=txt)

    fs = freqs
    amps = ampdata[0,:]
    f = fit.Lorentzian(fs, amps)
    h0 = np.max(amps)
    w0 = 2e6
    pos = fs[np.argmax(amps)]
    p0 = [np.min(amps), w0*h0, pos, w0]
    p = f.fit(p0)
    txt += 'Center = %.03f MHz' % (p[2]/1e6,)
#    print 'Fit gave: %s' % (txt,)
#        plt.plot(fs/1e6, f.func(p, fs), label=txt)

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Frequency [MHz]')
    
#    fig.canvas.draw()
    
    return [f._fit_params, f._fit_err]

class ROCavSpectroscopyPS(Measurement1D):

    def __init__(self, qubit_info, powers, freqs, 
                 phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 plot_type=None, qubit_pulse=False, seq=None, **kwargs):
        self.qubit_info = qubit_info
        self.freqs = freqs
        self.powers = powers
        self.qubit_pulse = qubit_pulse
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.seq = seq        
        
        if plot_type is None:
            if len(powers) > len(freqs):
                plot_type = POWER
            else:
                plot_type = SPEC
        self.plot_type = plot_type

        super(ROCavSpectroscopyPS, self).__init__(1, infos=qubit_info, **kwargs)
        self.data.create_dataset('powers', data=powers)
        self.data.create_dataset('freqs', data=freqs)
        self.ampdata = self.data.create_dataset('amplitudes', shape=(len(powers),len(freqs)))
        self.phasedata = self.data.create_dataset('phases', shape=(len(powers),len(freqs)))

    def generate(self):
        s = Sequence()
        if self.seq is not None:
            s.append(self.seq)
        else:
            s.append(Trigger(250))
            
        s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))

        if type(self.qubit_pulse) in (types.IntType, types.FloatType):
#            s.append(Trigger(250))
            s.append(self.qubit_info.rotate(self.qubit_pulse, 0))
        elif self.qubit_pulse:
#            s.append(Trigger(250))
            s.append(self.qubit_info.rotate(np.pi/2, 0))
#        else:
#            s.append(Trigger(250))
        s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        self.seqs = seqs
        return seqs

    def measure(self):
        # Generate and load sequences
        alz = self.instruments['alazar']
        alz.set_interrupt(False)

        seqs = self.generate()
        self.load(seqs)
        self.start_awgs()
        self.start_funcgen()

        for ipower, power in enumerate(self.powers):
            self.readout_info.rfsource1.set_power(power)
            print 'Power = %s' % (power, )
            time.sleep(2)

            amps = []
            phases = []

            for ifreq, freq in enumerate(self.freqs):
                self.readout_info.rfsource1.set_frequency(freq)
                self.readout_info.rfsource2.set_frequency(freq+50e6)
                time.sleep(0.05)

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
                print 'F = %.03f MHz --> re = %.01f, amp = %.1f, angle = %.01f' % (freq / 1e6, np.real(IQ), np.abs(IQ), np.angle(IQ, deg=True))

            self.ampdata[ipower,:] = amps
            self.phasedata[ipower,:] = phases
            
        
        self.analyze(fig=self.get_figure())
        if self.savefig:
            self.save_fig()
        

    def analyze(self, data=None, fig = None):
#        pax = ax if (ax is not None) else plt.figure().add_subplot(111)
        ampdata = data if (data is not None) else self.ampdata
        txt = 'phonon pi = %0.3f ns\n' % (self.phonon_pi)
#        txt += 'software df = %0.3f kHz\n' % (self.detune/1e3)
        txt += 'amp = %0.3f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        self.fit_params = analysis(self.powers, self.freqs, ampdata, self.phasedata, self.plot_type, fig = fig, txt = txt)
