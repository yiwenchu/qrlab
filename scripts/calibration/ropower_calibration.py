from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import objectsharer as objsh
import time

SPEC   = 0
POWER  = 1

'''
DEPRECATED: use scripts.calibration.ro_power_cal
'''

def analysis(powers, ampdata_pulse, ampdata_nopulse,
             phasedata_pulse, phasedata_nopulse):
    fig = plt.figure()


    ###########################################
    # Plot pulse and no pulse data - amplitude
    ###########################################

    ax = fig.add_subplot(211)
    ax.plot(powers, ampdata_pulse, 'rs-', label='with pulse')
    ax.plot(powers, ampdata_nopulse, 'bs-', label='without pulse')
    ax.legend(loc='best')

    ###########################################
    # contrast
    ###########################################
    diff = ampdata_pulse - ampdata_nopulse
    convolved_diff = convolve_data(diff, kernel_size=3)

    best_power = powers[np.argmax(np.abs(diff))]
    best_convolved = powers[np.argmax(np.abs(convolved_diff))]

    ax = fig.add_subplot(212)
    ax.plot(powers, diff, 'k-', label='contrast, best power: %0.3f dBm' % best_power)
    ax.plot(powers, convolved_diff, 'gs-', label='convolved, best power: %0.3f dBm' % best_convolved)
    ax.legend(loc='best')

    ax.set_xlabel('ro power (dBm)')

    return best_convolved

def convolve_data(data, kernel_size=None):
    if kernel_size is None:
        kernel_size = 3

    ksm1 = kernel_size-1
    # prepend and append with the first and last data, respectively to
        # take into account boundary effects
    pre_data = data[0] * np.ones(ksm1)
    post_data = data[-1] * np.ones(ksm1)
    temp_data = np.append(pre_data, data)
    temp_data = np.append(temp_data, post_data)
    convolve_kernel = np.ones(kernel_size, dtype=float) / kernel_size

    return np.convolve(temp_data, convolve_kernel, 'same')[ksm1:-ksm1]

PI_PULSE = 1
SAT_PULSE = 0

class ROPower_Calibration(Measurement1D):

    def __init__(self, qubit_info, powers, qubit_pulse=PI_PULSE,
                 seq=None, pulse_len=50000, simuldrive=False,**kwargs):
        self.qubit_info = qubit_info
        self.powers = powers
        self.qubit_pulse = qubit_pulse
        self.pulse_len = pulse_len
        self.simuldrive = simuldrive
        if simuldrive:
            assert qubit_pulse == SAT_PULSE
        if seq is None:
            seq = Trigger(250)
        self.seq = seq

        kwargs['print_progress'] = False
        super(ROPower_Calibration, self).__init__(2, infos=qubit_info, **kwargs)
        self.data.create_dataset('powers', data=powers)
        self.ampdata_pulse = self.data.create_dataset('amplitudes_pulse', shape=[len(powers)])
        self.phasedata_pulse = self.data.create_dataset('phases_pulse', shape=[len(powers)])
        self.ampdata_nopulse = self.data.create_dataset('amplitudes_nopulse', shape=[len(powers)])
        self.phasedata_nopulse = self.data.create_dataset('phases_nopulse', shape=[len(powers)])

    def generate(self):
        s = Sequence()

        for expt in [True, False]:
            s.append(self.seq)
            if expt:
                if self.qubit_pulse:
                    # saturation experiment
                    s.append(self.qubit_info.rotate(np.pi, 0))
                else:
                    s.append(Constant(self.pulse_len, 1, chan=self.qubit_info.channels[0]))


            if self.simuldrive and expt:
                rop = self.get_readout_pulse()
                rol = rop.get_length()
                
                drive_pulse = Constant(rol, 1, chan=self.qubit_info.channels[0])
                s.append(Combined([drive_pulse, rop]))

            else:
                s.append(self.get_readout_pulse())
                
        s = self.get_sequencer(s)
        seqs = s.render()

#        s.plot_seqs(seqs)#

        return seqs

    def measure(self):
        # Generate and load sequences
        alz = self.instruments['alazar']

        seqs = self.generate()
        self.load(seqs)

        self.shot_data = None
        # If saving complex data, save both raw signal and post-processed version
        if not self.real_signals:
            self.avg_data = self.data.create_dataset('avg', [self.cyclelen,], dtype=np.complex)
            self.pp_data = self.data.create_dataset('avg_pp', [self.cyclelen,], dtype=np.float)
        else:
            self.avg_data = self.data.create_dataset('avg', [self.cyclelen,], dtype=np.float)
            self.pp_data = None
        self.save_settings()

        alz.setup_clock()
        alz.setup_channels()
        alz.setup_trigger()
        alz.set_real_signals(self.real_signals)     # Doesn't do anything for histrograms

        amps_pulse = []
        phases_pulse = []
        amps_nopulse = []
        phases_nopulse = []
        for power in self.powers:
            self.readout_info.rfsource1.set_power(power)
            self.stop_funcgen()  #Sometimess the experiment hangs, are these needed?
            self.stop_awgs()
            time.sleep(.1)

            data = self.acquisition_loop(alz)

            amps_pulse.append(np.abs(data[0]))
            phases_pulse.append(np.angle(data[0], deg=True))
            amps_nopulse.append(np.abs(data[1]))
            phases_nopulse.append(np.angle(data[1], deg=True))

            print 'P = %.03f dBm\n \
                 ---> pulse = (%.03f, %.03f); no pulse = (%.03f, %.03f)' % \
                (power, np.abs(data[0]), np.angle(data[0], deg=True), \
                        np.abs(data[1]), np.angle(data[1], deg=True))

        # TODO: default plotting is not correct, so we need to close those plots
        plt.close(self.get_figure())

        self.ampdata_pulse[:] = np.array(amps_pulse)
        self.phasedata_pulse[:] = np.array(phases_pulse)
        self.ampdata_nopulse[:] = np.array(amps_nopulse)
        self.phasedata_nopulse[:] = np.array(phases_nopulse)

        self.analyze()
        if self.savefig:
            self.save_fig()
        return self.ampdata_pulse[:], self.phasedata_pulse[:], self.ampdata_nopulse[:], self.phasedata_nopulse[:]

    def analyze(self, data=None, ax=None):
#        pax = ax if (ax is not None) else plt.figure().add_subplot(111)
#        ampdata = data if (data is not None) else self.ampdata
        analysis(self.powers, self.ampdata_pulse[:], self.ampdata_nopulse[:],
                 self.phasedata_pulse[:], self.phasedata_nopulse[:])

    def get_best_power(self):
        diff = self.ampdata_pulse[:] - self.ampdata_nopulse[:]
        print 'diff list:',diff

        convolved_diff = convolve_data(diff, kernel_size=3)
        print 'list of convolved diffs:',convolved_diff
        '''That list is wrong, for some reason.'''
#        best_power = self.powers[np.argmax(np.abs(diff))]
        best_convolved = self.powers[np.argmax(np.abs(convolved_diff))]
        print 'best_convolved:',best_convolved

        return best_convolved