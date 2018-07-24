from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import time
import objectsharer as objsh
from lib.plotting_support import plotting

SPEC   = 0
POWER  = 1

class SpectroscopyYoko(Measurement1D):
    '''
    Perform qubit spectroscopy

    The frequency of <cavity_rfsource> will be swept over <freqs> and
    different yoko voltages <yoko_voltages> set on <qubit_yoko> for a single
    read-out power <ro_power> will be set on readout_info.rfsource1.

    The spectroscopy pulse has length 100 * <plen> ns.

    If <seq> is specified it is played at the start (should start with a trigger)
    If <postseq> is specified it is played at the end, right before the read-out
    pulse.
    '''

    def __init__(self, qubit_info, freqs,
                 qubit_yoko, yoko_voltages, spec_params,
#                 ro_powers,
                 plen=50e3, amp=1, seq=None, postseq=None,
                 freq_delay=0.1, #pow_delay=1,
                 do_analyze=True,
                 use_weight=False, use_IQge=False,
                 subtraction=False,
                 **kwargs):

        self.qubit_info = qubit_info
        self.freqs = freqs

        self.qubit_yoko = qubit_yoko
        self.yoko_voltages = yoko_voltages

        self.spec_params = spec_params
        self.qubit_rfsource, self.spec_power = spec_params

#        self.ro_powers = ro_powers # this is ugly and needs to be fixed

        self.plen = plen
        self.amp = amp
#        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.use_weight = use_weight
        self.use_IQge = use_IQge
        self.subtraction = subtraction

#        self.plot_type = SPEC

        self.extra_info = None
        super(SpectroscopyYoko, self).__init__(1, infos=qubit_info, **kwargs)
        self.data.create_dataset('yoko_voltages', data=yoko_voltages)
        self.data.create_dataset('freqs', data=freqs)

        data_shape = [len(yoko_voltages), len(freqs)]
        self.IQs = self.data.create_dataset('avg', shape=data_shape, dtype=np.complex)
        self.reals = self.data.create_dataset('avg_pp', shape=data_shape)

    def measure(self):
        from scripts.single_qubit import spectroscopy

        spec = lambda yv: \
            spectroscopy.Spectroscopy(
                                       self.qubit_info,
                                       self.freqs,
                                       [self.qubit_rfsource, self.spec_power],
#                                       self.ro_powers,
                                       plen=self.plen,
                                       amp=self.amp,
                                       seq=self.seq,
                                       postseq=self.postseq,
#                                       pow_delay=self.pow_delay,
                                       extra_info=self.extra_info,
                                       freq_delay=self.freq_delay,
#                                       plot_type=self.plot_type,
                                       title='Pulsed spectroscopy: (ro, spec) = (%0.2f, %0.2f) dBm; yoko %0.3f V\n' % (self.instruments['ag_ro'].get_power(), self.spec_power, yv),
                                       analyze_data=True,
                                       keep_data=True,
                                       use_weight=self.use_weight,
                                       use_IQge=self.use_IQge,
                                       subtraction=self.subtraction
                                       )

        # initialize yoko to a known voltage
        self.qubit_yoko.set_voltage(0)
        self.qubit_yoko.set_output_state(1)

        time.sleep(1)
        for idx, voltage in enumerate(self.yoko_voltages):
            self.qubit_yoko.set_voltage(voltage)

            # run spec at this voltage
            spec_exp = spec(voltage)
            spec_exp.measure()
#            spec_exp.started = False #trying to fix syncing.  Something is wrong atm.

            #save daata
            self.IQs[idx,:] = spec_exp.IQs[:]
            self.reals[idx,:] = spec_exp.reals[:]

        self.qubit_yoko.set_voltage(0)

        self.analyze()
        if self.savefig:
            self.save_fig()
        return self.yoko_voltages, self.freqs, self.reals

    def analyze(self):

        plotting.pcolor_2d(self.yoko_voltages, self.freqs/1e6,
                         np.transpose(self.reals[:])) #the [:] is important.