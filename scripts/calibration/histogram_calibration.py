from measurement import Measurement
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import objectsharer as objsh
import time
import mclient
import scripts.single_qubit.rabi as rabi
import fitting_programs as fp

#SPEC   = 0
#POWER  = 1
#alz = mclient.instruments['alazar']
#IF_PERIOD = alz.get_if_period()
#
#def analysis(expt1, expt2):
#
#    expt1_integrated = numerical_integrate_expt(expt1)
#    expt2_integrated = numerical_integrate_expt(expt2)
#
#    expt1_ave_demod = averaged_demod_expt(expt1)
#    expt2_ave_demod = averaged_demod_expt(expt2)
#
#
#    ###########################################
#    # Plot averaged trajectories
#    ###########################################
#    fig = plt.figure()
#
#    fig.suptitle('Averaged trajectories')
#
#    ax = fig.add_subplot(211)
#    ax.plot(expt1_ave_demod.real, expt1_ave_demod.imag, label='expt 1')
#    ax.plot(expt2_ave_demod.real, expt2_ave_demod.imag, label='expt 2')
#    ax.legend(loc='best')
#    ax.set_title('averaged measurement trajectory')
#
#    # plot amplitude of measurement envelope
#    ax = fig.add_subplot(212)
#    ax.plot(np.abs(expt1_ave_demod))
#    ax.plot(np.abs(expt2_ave_demod))
#    ax.set_xlabel('if period')
#    ax.set_ylabel('amplitude')
#    ax.set_title('Averaged measurement envelope')
#
#    ###########################################
#    # Blind fidelities
#    ###########################################
#
#    # group cumulative integrated data by IF period
#    expt1_integrated_if = np.transpose(expt1_integrated)
#    expt2_integrated_if = np.transpose(expt2_integrated)
#
#    fids = []
#    for expt1_iqs, expt2_iqs in zip(expt1_integrated_if, expt2_integrated_if):
#        counts, xbins, ybins = determine_histogram(expt1_iqs, expt2_iqs)
#        fids.append(calculate_blind_fidelity(counts))
#
#    # plot IQs for the best readout integration time.
#    fig = plt.figure()
#    ax = fig.add_subplot(121)
#    best_fid = np.argmax(fids)
#    counts, xbins, ybins = determine_histogram(expt1_integrated_if[best_fid],
#                                               expt2_integrated_if[best_fid])
#
#
#
#    return np.array(fids)
#
#def numerical_integrate_expt(expt_records):
#    '''
#        returns the cumulative sum for each trace in <expt_records>:
#
#        input:
#            expt_records: [[A1, A2, A3, ..., AN],
#                           [B1, B2, B3, ..., BN],
#                           ...]
#
#        output:
#            num_int_records: [[A1, sum(A, 1, 2), sum(A, 1, 3), ..., sum(A, 1, N)],
#                               ...]
#
#    '''
#
#    return np.cumsum(expt_records, axis=1)
#
#def averaged_demod_expt(expt_records):
#    '''
#        returns the IQ-index averaged record.
#
#        input:
#            expt_records: [[A1, A2, A3, ..., AN],
#                           [B1, B2, B3, ..., BN],
#                           ...]
#
#        output:
#            ave_expt: [ave(*1), ave(*2), ... , ave(*N)]
#    '''
#
#    return np.average(hi, axis=0)
#
#def calculate_blind_fidelity(counts):
#
#    counts1, counts2 = counts
#
#    difference = counts1 - counts2
#    fid = difference  / (2 * np.sum(counts1)) # this should be true: sum counts1 == counts2
#
#    return fid
#
#
#def determine_histogram(IQs1, IQs2, num_steps=101):
#
#    IQs = np.concatenate((IQs1, IQs2))
#    IQs.flatten()
#
#    hist_xmax, hist_xmin = get_limits(IQs.real, 1.1)
#    hist_ymax, hist_ymin = get_limits(IQs.imag, 1.1)
#
#    xbins = np.linspace(hist_xmin, hist_xmax, num_steps)
#    ybins = np.linspace(hist_ymin, hist_ymax, num_steps)
#
#    counts = [0, 0]
#    for i, expt in enumerate([IQs1, IQs2]):
#        counts[i], xedges, yedges = np.histogram2d(expt.real, expt.imag,
#                                                    bins=[xbins, ybins])
#    return np.array(counts), xbins, ybins
#
#
#def get_limits(series, scale):
#    ave = np.averages(series)
#    delta = 0.5 * (np.max(series) - np.min(series))
#
#    return ave + scale * delta, ave - scale * delta
#
#PI_PULSE = 0
#
#class Histogram_Readout(Measurement1D):
#
#    def __init__(self, qubit_info, powers, qubit_pulse=PI_PULSE, seq=None, **kwargs):
#        self.qubit_info = qubit_info
#        self.powers = powers
#        self.qubit_pulse = qubit_pulse
#        self.seq = seq
#
#        super(Histogram_Readout, self).__init__(2, infos=qubit_info, **kwargs)
##        self.data.create_dataset('powers', data=powers)
#
#    def generate(self):
#        s = Sequence()
#
#        for expt in [True, False]:
#            s.append(Trigger(250))
#
#            if self.seq is not None:
#                s.append(self.seq)
#
#            if expt:
#                if self.qubit_pulse:
#                    # saturation experiment
#                    s.append(Constant(self.qubit_pulse, 1, chan=self.qubit_info.channels[0]))
#                else:
#                    s.append(self.qubit_info.rotate(np.pi, 0))
#
#            s.append(self.get_readout_pulse())
#
#        s = self.get_sequencer(s)
#        seqs = s.render()
#
#        return seqs
#
#    def measure(self):
#        # Generate and load sequences
#        alz = self.instruments['alazar']
#        if alz is None:
#            logging.error('Alazar instrument not found!')
#            return
#        num_shots = alz.get_naverages()
#
#        seqs = self.generate()
#        self.load(seqs)
#
#        alz.setup_shots(num_shots)
#        buf = alz.take_raw_demod_shots()
#
#        expt1 = buf[::2]
#        expt2 = buf[1::2]
#
#        self.expt1 = expt1
#        self.expt2 = expt2
##        self.analyze(expt1, expt2)
#        return expt1, expt2

def get_limits(series, scale):
    ave = 0.5 * (np.max(series) + np.min(series))
    delta = 0.5 * (np.max(series) - np.min(series))
    return ave + scale * delta, ave - scale * delta

def determine_histogram(IQs1, IQs2, num_steps=26):
    '''
        In order to perform blind fidelity experiments for two datasets, we need
        to generate on global set of histogram bins taking into consideration
        both datasets.

        Outputs:
            - counts: histograms counts for each IQ dataset
            - xbins, ybins: global histogram bins
    '''


    IQs = np.concatenate((IQs1, IQs2))
    IQs.flatten()

    hist_xmax, hist_xmin = get_limits(IQs.real, 1.1)
    hist_ymax, hist_ymin = get_limits(IQs.imag, 1.1)

    xbins = np.linspace(hist_xmin, hist_xmax, num_steps)
    ybins = np.linspace(hist_ymin, hist_ymax, num_steps)

    counts = [0, 0]
    for i, expt in enumerate([IQs1, IQs2]):
        counts[i], xedges, yedges = np.histogram2d(expt.real, expt.imag,
                                                    bins=[xbins, ybins])
    return np.array(counts), xbins, ybins

def calculate_bf(counts):
    counts1, counts2 = counts
    diff_counts = np.abs(counts1 - counts2)
    fid = np.sum(diff_counts) / (2.0 * np.sum(counts1))
    return fid, diff_counts

def blind_fidelity(IQ1, IQ2, plot=True, verbose=False, num_steps=26):

    counts, xbins, ybins = determine_histogram(IQ1, IQ2, num_steps=num_steps)
    counts = [np.rot90(counts[0]), np.rot90(counts[1])]

    fid, diff_counts = calculate_bf(counts)
    if verbose:
        print 'blind fidelity: %0.3f' % fid

    if plot:
        import matplotlib.gridspec as gridspec
        fig = plt.figure()
        plt.suptitle('Blind fidelity histograms, blind fidelity = %0.3f'%fid)
        gs = gridspec.GridSpec(2,2)
        ax_1 = fig.add_subplot(gs[0,0])
        ax_2 = fig.add_subplot(gs[0,1])
        ax_3 = fig.add_subplot(gs[1,:])

        extent = (xbins[0], xbins[-1], ybins[0], ybins[-1])
#        print extent

        ax_1.imshow(counts[0], interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|g>')
        ax_2.imshow(counts[1], interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|e>')
        ax_3.imshow(diff_counts, interpolation='nearest', extent=extent, cmap='hot', aspect='auto', label='|e> -|g>')

        ax_1.set_title('|g>')
        ax_2.set_title('|e>')
        ax_3.set_title('|e> - |g>')
    return fid

def generate_cumulative_integrated_iqs(iq_demod):
    int_iqs = np.cumsum(iq_demod, axis=1)
    return np.transpose(int_iqs) / np.arange(1, len(int_iqs[0])+1)[:,np.newaxis]

def cumulative_blind_fidelity(g_iq_demod, e_iq_demod):
    g_int_iq = generate_cumulative_integrated_iqs(g_iq_demod)
    e_int_iq = generate_cumulative_integrated_iqs(e_iq_demod)

    # for each integrated if index, calculate blind fidelity
    blind_fids = []
    for i, (g_iq, e_iq) in enumerate(zip(g_int_iq, e_int_iq)):
        blind_fids.append(blind_fidelity(g_iq, e_iq, plot=False, verbose=False))
    return np.array(blind_fids)

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

class Histogram_Readout(Measurement):

    def __init__(self, qubit_info, powers, shots=1e5,#qubit_pulse=PI_PULSE,
                 shots_per_alz_call=1000, seq=None, **kwargs):
        self.qubit_info = qubit_info
        self.powers = powers
        self.shots = shots
        self.shots_per_alz_call = shots_per_alz_call
        self.seq = seq

        super(Histogram_Readout, self).__init__(2, infos=qubit_info, **kwargs)
        self.alz = self.instruments['alazar']
        self.ag_ro = self.instruments['ag_ro']

        self.data.create_dataset('powers', data=powers)
        self.blind_fids = self.data.create_dataset('blind_fidelities',
                                 shape=[len(powers),
                                        self.alz.get_nsamples()/20])



    def raw_demod(self):
        '''acquisitions, not averages'''
        reps = []
        for i in np.arange(self.shots / self.shots_per_alz_call):
            print 'soft average: %d' % (i * self.shots_per_alz_call)
            self.alz.setup_avg_shot(self.shots_per_alz_call)
            reps.append(self.alz.take_raw_demod_shots(acqtimeout=50000))
        return np.concatenate(reps)

    def measure(self):
        # Generate and load sequences

        self.alz.setup_channels()
        self.alz.setup_clock()
        self.alz.setup_trigger()

        self.ebufs = []
        self.gbufs = []
#        self.blind_fids = []
        for idx, p in enumerate(self.powers):

            self.ag_ro.set_power(p)
            time.sleep(1)

            tre = rabi.Rabi(self.qubit_info, [self.qubit_info.pi_amp,],
                            real_signals=False,)
            tre.play_sequence()
            self.alz.setup_avg_shot(self.shots)
            ebuf = self.raw_demod()

            trg = rabi.Rabi(self.qubit_info, [0.00001,], real_signals=False)
            trg.play_sequence()
            self.alz.setup_avg_shot(self.shots)
            gbuf = self.raw_demod()

#            self.ebufs.append(ebuf)
#            self.gbufs.append(gbuf)'

            fids = cumulative_blind_fidelity(gbuf, ebuf)
#            self.blind_fids.append(fids)
            self.blind_fids[idx,:] = fids


#        best_power,best_length,smoothed_bfs = self.analyze()
#        return best_power, best_length, self.ebufs, self.gbufs, self.powers
        best_power, best_length, smoothed_bfs = process_blind_fidelities(self.blind_fids[:], self.powers)
        return best_power, best_length, self.blind_fids[:], self.powers

    def analyze(self):

        return bf_analyze(self.ebufs,self.gbufs,self.powers)

def process_blind_fidelities(blind_fids, powers, ax=None):
    bfs = np.array(blind_fids)
    demod_times = (np.arange(len(blind_fids[0])) + 1) * 20
    #smooth the data
    smoothed_bfs = np.zeros(np.shape(bfs))
    for idx,power_data in enumerate(bfs):
        smoothed_bfs[idx,:] = convolve_data(bfs[idx],kernel_size=3)

    best_idx = np.unravel_index(np.argmax(smoothed_bfs),np.shape(smoothed_bfs))

    best_power = powers[best_idx[0]]
    best_length = 20*(best_idx[1]+1)
    txt = 'Best power: %0.02f dBms, best integration length %d ns, fidelity: %.03f' %(best_power,best_length,smoothed_bfs[best_idx])

    print txt
    if np.shape(bfs)[0] > 1:
        # weird thing going on:
        # even though I call a new figure, the imshow() insists on plotting in
        # the current figure..
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x_min, x_max = demod_times[0], demod_times[-1]
        y_min, y_max = powers[0], powers[-1]
        ax.imshow(bfs, interpolation='nearest', origin='lower',
                  aspect='auto', extent=(x_min, x_max, y_min, y_max))
        ax.set_title(txt)
#        fp.plot_2d(demod_times, powers, smoothed_bfs)
    else:
        plt.figure()
#        plt.plot(np.transpose(bfs))
        plt.plot(np.transpose(smoothed_bfs))
        plt.title(txt)
        plt.show()

    return best_power, best_length, smoothed_bfs


def bf_analyze(ebufs, gbufs, powers):
    bfs =[]
    for ebuf,gbuf in zip(ebufs,gbufs):
        fids = cumulative_blind_fidelity(gbuf, ebuf)
        bfs.append(fids)

    process_blind_fidelities(bfs, powers)

