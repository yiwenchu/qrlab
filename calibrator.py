import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lib.math import fit

mpl.rcParams['legend.fontsize'] = 7

class AWGCalibrator:

    def __init__(self, awg, sa, delay=0.05, channel_amp=0.6, marker=None):
        self.awg = awg
        self.sa = sa
        self.delay = delay
        self.channel_amp = channel_amp
        self.marker = marker

    def find_channel_offset(self, chan, V0, vrange, N=15, plot=True):
        '''
        Find (and set) a channel offset using spectrum analyzer sa:
        - V0: center voltage
        - vrange: from -vrange to +vrange around V0
        - N: number of steps
        - Navg: number of averages per step

        Returns:
        - (Vmin, Pmin)

        '''
        vs = np.linspace(V0 - vrange, V0 + vrange, N)
        ps = []
        for v in vs:
            self.awg.set('ch%s_offset'%chan, v)
            self.awg.get('ch%s_offset'%chan)
#            time.sleep(self.delay)
            plevel = self.sa.get_power()
            print 'V %.03f, power %.03f' % (v, plevel, )
            ps.append(plevel)

        ps = np.array(ps)
        f = fit.Polynomial(vs, ps, order=2)
        p0 = [np.min(ps), 0, 1]
        p = f.fit(p0)
        print 'Fit parameters: %s' % (p,)
        center = -p[1] / 2.0 / p[2]

        if plot:
            plt.plot(vs, ps, label='chan%d, V %.03f +- %.03f, C = %.03f'%(chan, V0, vrange, center))

        imin = np.argmin(ps)
        print 'Minimum power at %s = %.03f, fit = %.03f' % (chan, vs[imin], center)

        self.awg.set('ch%s_offset'%chan, vs[imin])
        return vs[imin]
#        self.awg.set('ch%s_offset'%chan, center)
#        return center

    def calibrate_offsets(self, F0, chans, vrange=0.4, plot=True, incremental=False, iterations=6):
        '''
        Find (and set) optimal offset voltages on a pair of AWG channels
        by taking several voltage sweeps with binary decreasing voltage range.
        vrange: starting voltage range
        '''

        self.sa.set_rf_on(True)

        self.sa.set_frequency(F0)
        self.awg.output_zeros(chans)
        if not incremental:
            self.awg.set('ch%s_amplitude'%chans[0], self.channel_amp)
            self.awg.set('ch%s_amplitude'%chans[1], self.channel_amp)
            v1, v2 = 0, 0
            time.sleep(0.2)
        else:
            v1 = self.awg.get('ch%s_offset'%chans[0])
            v2 = self.awg.get('ch%s_offset'%chans[1])

        if plot:
            plt.figure()

        for i in range(iterations):
            v1 = self.find_channel_offset(chans[0], v1, vrange, plot=plot)
            v2 = self.find_channel_offset(chans[1], v2, vrange, plot=plot)
            vrange /= 2

        if plot:
            plt.legend(loc='best')

        self.sa.set_rf_on(False)

        return v1, v2

    def optimize_delay_time(self, chan, t0, trange, f1, f2, N=21, ax=None):
        ts = np.linspace(t0 - trange, t0 + trange, N)
        rs = []
        rs2 = []
        for t in ts:
            self.awg.set('ch%s_skew'%chan, t)
            self.awg.get('ch%s_skew'%chan)
            p1 = self.sa.get_power_at(f1)
            p2 = self.sa.get_power_at(f2)
            rs.append(p1 - p2)
            rs2.append(p2)

        if ax:
            ax.plot(ts, rs, label='T +- %d ps' % (trange,))
            ax.plot(ts, rs2, label='T +- %d ps [p1]' % (trange,))

        imax = np.argmin(rs2)
        self.awg.set('ch%s_skew'%chan, ts[imax])
        return ts[imax]

    def optimize_phase(self, chans, period, phi0, phirange, f1, f2, N=21, ax=None):
        phis = np.linspace(phi0 - phirange, phi0 + phirange, N)
        rs = []
        rs2 = []
        for phi in phis:
            self.awg.sideband_modulate(period, dphi=phi, chans=chans)
            time.sleep(0.2)
            p1 = self.sa.get_power_at(f1)
            p2 = self.sa.get_power_at(f2)
            print 'Phi %.03f, p1 = %.03f, p2 = %.03f' % (phi, p1, p2)
            rs.append(p1 - p2)
            rs2.append(p2)

        if ax:
            ax.plot(phis, rs, label='Phi +- %.03f' % (phirange,))
            ax.plot(phis, rs2, label='Phi +- %.03f [p1]' % (phirange,))

        imax = np.argmin(rs2)
        self.awg.sideband_modulate(period, dphi=phis[imax], chans=chans)
        return phis[imax]

    def optimize_amplitude(self, chan, amp0, amprange, f1, f2, N=21, ax=None):
        amps = np.linspace(amp0 - amprange, amp0 + amprange, N)
        rs = []
        rs2 = []
        for amp in amps:
            self.awg.set('ch%s_amplitude'%chan, amp)
            self.awg.get('ch%s_amplitude'%chan)
            time.sleep(self.delay)
            p1 = self.sa.get_power_at(f1)
            p2 = self.sa.get_power_at(f2)
            print 'Amp %.03f, p1 = %.03f, p2 = %.03f' % (amp, p1, p2)
            rs.append(p1 - p2)
            rs2.append(p2)

        if ax:
            ax.plot(amps, rs, label='Amp +- %.03f' % (amprange,))
            ax.plot(amps, rs2, label='Amp +- %.03f' % (amprange,))

        imax = np.argmin(rs2)
        self.awg.set('ch%s_amplitude'%chan, amps[imax])
        return amps[imax]

    def calibrate_sideband_skew(self, chan, f1, f2, period, chans=(1,2), plot=True):
        '''
        Optimize IQ mixer for single sideband modulation.
        Frequencies around f1 and f2, awg period <period> on channels <chans>
        '''

        # The starting point
        self.awg.sideband_modulate(period, dphi=0, chans=chans)
        self.awg.set('ch%s_skew'%chans[0], 0)
        self.awg.set('ch%s_skew'%chans[1], 0)
        self.awg.set('ch%s_amplitude'%chans[0], self.channel_amp)
        self.awg.set('ch%s_amplitude'%chans[1], self.channel_amp)
        time.sleep(0.5)

        # Find frequencies more accurately
#        f1 = self.sa.find_peak(f1, 5e6, 41, plot=False)
#        f2 = self.sa.find_peak(f2, 5e6, 41, plot=False)
        print 'Sidebands @ f1 = %.03f MHz, f2 = %.03f MHz' % (f1/1e6, f2/1e6)

        # Find whether we need a pi phase shift on the first channel to get to the requested sideband
        p1_noflip = self.sa.get_power_at(f1)
        p2_noflip = self.sa.get_power_at(f2)
        r_noflip = p1_noflip - p2_noflip
        self.awg.sideband_modulate(period, dphi=np.pi, chans=chans)
        time.sleep(0.5)
        p1_flip = self.sa.get_power_at(f1)
        p2_flip = self.sa.get_power_at(f2)
        r_flip = p1_flip - p2_flip

        if r_noflip > r_flip:
            print '  Pi phase-shift NOT required.'
            self.awg.sideband_modulate(period, flip=False)
        else:
            print '  Pi phase shift required.'

        if plot:
            ax_time = plt.figure().add_subplot(111)
            ax_amp = plt.figure().add_subplot(111)
        else:
            ax_time = None
            ax_amp = None

        # Starting parameters
        t0 = 0
        trange = 4000
        amp0 = self.awg.get('ch%s_amplitude'%chan)
        amprange = 0.25 * amp0

        for i in range(3):
            t0 = self.optimize_delay_time(chan, t0, trange, f1, f2, ax=ax_time)
            amp0 = self.optimize_amplitude(chan, amp0, amprange, f1, f2, ax=ax_amp)
            print 'Optimized t = %s, amp = %.03f' % (t0, amp0)

            trange /= 2
            amprange /= 2

        if plot:
            ax_time.autoscale(tight=True)
            ax_time.legend(loc='best')
            ax_amp.autoscale(tight=True)
            ax_amp.legend(loc='best')

    def calibrate_sideband_phase(self, chans, f1, f2, period, iterations=4, plot=True, divide=2.5, find_peaks=False):
        '''
        Optimize IQ mixer for single sideband modulation.
        Frequencies around f1 and f2, awg period <period> on channels <chans>.
        Returns tuple <amplitude>, <phase> for the optimal point.
        '''

        # The starting point
        self.awg.sideband_modulate(period, dphi=0, chans=chans)
#        self.awg.set_channel_skew(0, chans[0])
#        self.awg.set_channel_skew(0, chans[1])
        self.awg.set('ch%s_amplitude'%chans[0], self.channel_amp)
        self.awg.set('ch%s_amplitude'%chans[1], self.channel_amp)
        self.awg.get('ch%s_amplitude'%chans[0])
        time.sleep(0.5)

        # Find frequencies more accurately
        if find_peaks:
            f1 = self.sa.find_peak(f1, freqrange=500e3, N=21)
            f2 = self.sa.find_peak(f2, freqrange=500e3, N=21)
        print 'Sidebands @ f1 = %.03f MHz, f2 = %.03f MHz' % (f1/1e6, f2/1e6)

        self.sa.set_rf_on(True)

        if plot:
            ax_phase = plt.figure().add_subplot(111)
            ax_amp = plt.figure().add_subplot(111)
        else:
            ax_phase = None
            ax_amp = None

        # Starting parameters
        phi0 = np.pi
        phirange = np.pi
        amp0 = self.awg.get('ch%s_amplitude'%chans[0])
        amprange = 0.25 * amp0

        # Amplitude and phase seem to be quite independent
        for i in range(iterations):
            phi0 = self.optimize_phase(chans, period, phi0, phirange, f1, f2, ax=ax_phase, N=19)
            amp0 = self.optimize_amplitude(chans[0], amp0, amprange, f1, f2, ax=ax_amp, N=15)
            print 'Optimized phi = %s, amp = %.03f' % (phi0, amp0)

            phirange /= divide
            amprange /= divide

        if plot:
            ax_phase.autoscale(tight=True)
            ax_phase.legend(loc='best')
            ax_amp.autoscale(tight=True)
            ax_amp.legend(loc='best')

        self.sa.set_rf_on(False)

        return amp0, phi0

    def get_power_vs_delay(self, chan, t0, trange, f, N=21, ax=None):
        ts = np.linspace(t0 - trange, t0 + trange, N)
        rs = []
        for t in ts:
            self.awg.set('ch%s_skew'%chan, t)
            rs.append(self.sa.get_power_at(f))

        if ax:
            ax.plot(ts, rs, label='T +- %d ps' % (trange,))

        return np.array(ts), np.array(rs)