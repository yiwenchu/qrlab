from measurement import Measurement1D
import scripts.jk.single_cavity.cavity_transmission as cav_trans
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
from lib.plotting_support import plotting
import time
import objectsharer as objsh

from mclient import save_fig
SPEC   = 0
POWER  = 1

class CavityTransmissionYoko(Measurement1D):
    '''
    Perform cavity transmission.

    The frequency of <cavity_rfsource> will be swept over <freqs> and
    different yoko voltages <yoko_voltages> set on <qubit_yoko> for a single
    read-out power <ro_power> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length 100 * <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.
    '''

    def __init__(self, freqs, qubit_yoko, yoko_voltages, ro_power,
                 plen=10000, amp=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1,
                 extra_info=None, plot_type=None,
                 **kwargs):

        self.freqs = freqs
        self.qubit_yoko = qubit_yoko
        self.yoko_voltages = yoko_voltages
        self.ro_power = ro_power
        self.plen = plen
        self.amp = amp
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.extra_info = extra_info

        self.plot_type = None # TODO figure out how to fit data.

        super(CavityTransmissionYoko, self).__init__(1, **kwargs)

        self.data.create_dataset('yoko_voltages', data=yoko_voltages)
        self.data.create_dataset('freqs', data=freqs)
        self.IQs = self.data.create_dataset('avg_data', shape=[len(yoko_voltages),len(freqs)],
                                           dtype=np.complex)
        self.amps = self.data.create_dataset('avg_pp', shape=[len(yoko_voltages),len(freqs)],
                                            dtype=np.float)


    def measure(self):

        cav_spec = lambda: cav_trans.CavityTransmission(self.freqs,
                                       [self.ro_power],
                                       plen=self.plen,
                                       amp=self.amp,
                                       seq=self.seq,
                                       postseq=self.postseq,
                                       pow_delay=self.pow_delay,
                                       extra_info=self.extra_info,
                                       freq_delay=self.freq_delay,
                                       plot_type=self.plot_type,
                                       analyze_data=False,
                                       keep_data=True) # THIS IS IMPORTANT

        # initialize yoko to a known voltage
        self.qubit_yoko.set_voltage(0)
        self.qubit_yoko.set_output_state(1)
        time.sleep(1)

        for iyv, yv in enumerate(self.yoko_voltages):
            self.qubit_yoko.set_voltage(yv)

            ct = cav_spec()
            ct.title = 'Yoko Trans, V = %0.3f, (%d/%d)\n' % \
                        (yv, iyv+1, len(self.yoko_voltages))
            ct.do_generate = (iyv == 0)
            ct.measure()
             #So we don't load after the first time

            # save data
            self.IQs[iyv,:] = ct.IQs[:][0,:]
            self.amps[iyv,:] = np.abs(ct.IQs[:][0,:])

        del ct.datafile[ct._groupname]
        self.qubit_yoko.set_voltage(0)
        self.analyze()
        return self.yoko_voltages, self.freqs, self.IQs[:]

    def analyze(self):
        if len(self.yoko_voltages) > 1 and len(self.freqs) > 1:
            f = plotting.pcolor_2d(1000*self.yoko_voltages, self.freqs/1e9,
                       np.transpose(np.abs(self.IQs[:]))) #the [:] is important.

            save_fig(f, 'YokoTrans')
