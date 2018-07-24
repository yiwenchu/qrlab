from sequencer import Combined, Sequence, Sequencer, Delay, Pulse, Instruction, Constant
import numpy as np
import scipy.special
import scipy.signal
import ampgen

#########################################
# Constants
#########################################

X = 0
Y = np.pi/2
X_AXIS = 0
Y_AXIS = np.pi/2
XY = np.pi/4
DELTA = 5e-3

def derivative(ar):
    ret = np.zeros(len(ar))
    ret[0] = (ar[1] - ar[0]) / 2.0
    ret[1:-1] = (ar[2:] - ar[:-2]) / 2.0
    ret[-1] = (ar[-1] - ar[-2]) / 2.0
    return ret

#########################################
# Some basic pulses
#########################################

class Gaussian(Pulse):
    '''
    A gaussian pulse with sigma <s>, amplitude <a> in a waveform
    with a total width of <chop> * <s>
    '''

    def __init__(self, s, a, chan=1, chop=4., **kwargs):
        blockwidth = np.round(chop*s)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth, endpoint=True)
        ys = a * np.exp(-ts**2/(2.0 * s**2))
        name = 'gauss(%.2f,%.5f)' % (s, a)
        super(Gaussian, self).__init__(name, ys, chan=chan, **kwargs)

    def get_area(self, s, a):
        return s * a * np.sqrt(2 * np.pi)

    def get_parameter_for_area(self, **kwargs):
        if 's' in kwargs:
            return
        elif 'a' in kwargs:
            return

class Lorentzian(Pulse):
    '''
    A Lorentzian pulse with FWHM <w>, amplitude <a> in a waveform
    with a total width of <chop> * <w>.
    '''

    def __init__(self, w, a, chan=1, chop=3., **kwargs):
        blockwidth = np.ceil(chop*w)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth+1, endpoint=True)
        ys = a * (w/2)**2 / (ts**2 + (w/2.0)**2)
        name = 'lorentz(%.2f,%.5f)' % (w, a)
        super(Lorentzian, self).__init__(name, ys, chan=chan, **kwargs)

class Square(Pulse):
    '''
    A square pulse with width <w> and amplitude <a> in a waveform with
    length w + 2*pad.
    In the frequency domain this gives a sinc function with zeros at N / w.
    '''

    def __init__(self, w, a, pad=2, chan=1, **kwargs):
        blockwidth = np.ceil(w+2*pad)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth, endpoint=True)
        ys = np.zeros_like(ts)
        ys[(ts>=-w/2)&(ts<=w/2)] = a
        name = 'square(%.1f,%.5f)' % (w, a)
        super(Square, self).__init__(name, ys, chan=chan, **kwargs)

    def get_area(self, w, a):
        return w * a

    def get_parameter_for_area(self, **kwargs):
        if 'w' in kwargs:
            return
        elif 'a' in kwargs:
            return

class Triangle(Pulse):
    '''
    A triangle pulse with width 2<w> and amplitude <a> in a waveform with
    length 2<w>.
    In the frequency domain this gives a sinc^2 function with zeros at N / w.
    '''

    def __init__(self, w, a, chan=1, **kwargs):
        blockwidth = np.ceil(2*w)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth, endpoint=True)
        ys = np.zeros_like(ts)
        ys = a * np.maximum(0, 1 - np.abs(ts)/w)
        name = 'tri(%.2f,%.5f)' % (w, a)
        super(Triangle, self).__init__(name, ys, chan=chan, **kwargs)

    def get_area(self, w, a):
        return w * a

class GaussSquare(Pulse):
    '''
    A square pulse with width <w> and amplitude <a> in a waveform
    with width <w> + <chop> * <sigma>
    '''

    def __init__(self, w, a, sigma, chan=1, chop=4, **kwargs):
        blockwidth = np.ceil(w+chop*sigma)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth, endpoint=True)
        ys = np.zeros_like(ts)
        ys[(ts>-w/2)&(ts<w/2)] = a
        mask = (ts<=-w/2)
        ys[mask] = a * np.exp(-((ts[mask]+w/2)/sigma)**2)
        mask = (ts>=w/2)
        ys[mask] = a * np.exp(-((ts[mask]-w/2)/sigma)**2)
        name = 'gsquare(%.2f,%.5f,%.2f)' % (w, a, sigma)
        super(GaussSquare, self).__init__(name, ys, chan=chan, **kwargs)

class Sinc(Pulse):
    '''
    A sinc pulse with width <w> between the first zero crossings and amplitude
    <a> in a waveform of length <chop> * <w>.
    In the frequency domain this gives a (pretty wavy) square of width 1/w.
    '''

    def __init__(self, w, a, chan=1, chop=4, **kwargs):
        blockwidth = np.ceil(chop*w)
        ts = np.linspace(-blockwidth/2, blockwidth/2, blockwidth, endpoint=True)
        ys = a * np.sinc(ts / (0.5 * w))
        name = 'sinc(%.2f,%.5f)' % (w, a)
        super(Sinc, self).__init__(name, ys, chan=chan, **kwargs)

class PhotonControlPulse(Pulse):
    '''
    Readout pulse for faster cavity ringup and down.
    Base amplitude A. Ringup amplitude and time constant Bu and gu.
    Ringdown amplitude and time constant -Bd and gd
    '''
    def __init__(self, length, A, Bu=0, gu=10, Bd=0, gd=10, chan=1):
        ts = np.arange(length)
        base = A*np.ones(length)
        ring_up = float(Bu)*np.exp(-np.arange(length)/float(gu))
        ring_down = -float(Bd)*np.exp(-(length-np.arange(length))/float(gd))
        ys = base+ring_up+ring_down
        name = 'photon_control(%d,%0.03f,%0.03f,%0.03f,%0.03f,%0.03f)' % \
            (l,A,Bu,gu,Bd,gd)
        super(PhotonControlPulse, self).__init__(name, ys, chan=chan)

#########################################
# Rotation generators
#########################################

class GSRotation(object):

    def __init__(self, pi_area, smin, smax, hmin, hmax, area_frac=1, chans=(1,2), chop=4, chirp=None, switch=False, switch_channel=None):
        self.pi_area = pi_area
        self.smin = smin
        self.smax = smax
        self.hmin = hmin
        self.hmax = hmax
        self.area_frac = area_frac
        self.chans = chans
        self.chop = chop
        self.chirp = chirp
        self.switch = switch
        self.switch_channel = switch_channel

    def set_pi(self, val):
        self.pi_area = val

    def __call__(self, alpha, phase):
        if alpha == 0:
            return Sequence()

        # Requested area
        if alpha < 0:
            alpha *= -1
            sign = -1
        else:
            sign = 1
        area = (alpha / np.pi) * self.pi_area / self.area_frac
        h = self.hmax

        # Minimum area of Gaussian sides
        gminarea = self.smin * np.sqrt(np.pi) * self.hmax
        # Calculate required square wave size
        if area > gminarea:
            ws = np.floor((area - gminarea) / self.hmax)
        else:
            h = area / np.sqrt(np.pi) / self.smin
            ws = 0
        # Calculate are for Gaussian sides
        garea = area - ws * h
        sigma = garea / np.sqrt(np.pi) / h

        a1 = sign * h * np.cos(phase)
        a2 = sign * h * np.sin(phase)
        p1 = GaussSquare(ws, a1, sigma, chan=self.chans[0], chop=self.chop)
        p2 = GaussSquare(ws, a2, sigma, chan=self.chans[1], chop=self.chop)
        if self.chirp:
            p1 = Chirp(p1, self.chirp, chan=self.chans[0])
            p2 = Chirp(p2, self.chirp, chan=self.chans[1])

        # no need for pulses
        if a1 == 0 and a2 == 0:
            return Sequence()
#        elif a1 == 0:
#            return Combined([Delay(p2len, chan=self.chans[0]), p2])
#        elif a2 == 0:
#            return Combined([p1, Delay(p1len, chan=self.chans[1])])
        if self.switch is True:
            switchMarker = Constant((4*sigma+ws+8), 1, chan = self.switch_channel)
            return Combined([p1, p2, switchMarker], align = 1)

#        print 'Requested area: %.03f, actual areas: %.03f / %.03f' % (area, p0.get_area(), p1.get_area())
        return Combined([p1, p2])

# Rotation generators should have a __call__ function that takes an rotation
# angle and a phase as an argument. The latter will control around which
# axis is being rotated

class AmplitudeRotation(object):
    '''
    A rotation with an angle that is controlled using the pulse amplitude.
    Both Pi and pi/2 amplitude can be specified and the amplitude for any
    given amplitude will be interpolated.
    '''

    def __init__(self, base, w, pi_amp, chans=(0,1), drag=0, pi2_amp=0, **kwargs):
        self.base = base
        self.w = w
        self.chans = chans
        self.kwargs = kwargs
        self.drag = drag
        self.ampgen = ampgen.AmpGen()
        self.set_pi_amp(pi_amp, pi2_amp)

    def set_pi_amp(self, pi_amp, pi2_amp=0):
        self.pi_amp = pi_amp
        self.pi2_amp = pi2_amp
        if pi2_amp != 0:
            self.ampgen.set_amp_spec([pi2_amp, pi_amp])
        else:
            self.ampgen.set_amp_spec(pi_amp)

    def __call__(self, alpha, phase, amp=None, drag=None):
        '''
        Generate a rotation pulse of angle <alpha> around axis <phase>.
        If <amp> is specified that amplitude is used and <alpha> is ignored.
        '''
        if amp is None:
            amp = self.ampgen(alpha)
        p0 = self.base(self.w, amp * np.cos(phase), chan=self.chans[0], **self.kwargs)
        p1 = self.base(self.w, amp * np.sin(phase), chan=self.chans[1], **self.kwargs)
        
        if drag is None:
            drag = self.drag
            
        if drag:
            p0d = p0.data + drag * derivative(p1.data)
            p1d = p1.data - drag * derivative(p0.data)
            p0 = Pulse('dragI(%s,%.5f)'%(p0.name, drag), p0d, chan=self.chans[0])
            p1 = Pulse('dragQ(%s,%.5f)'%(p1.name, drag), p1d, chan=self.chans[1])

#        if self.drag:
#            p0d = p0.data + self.drag * derivative(p1.data)
#            p1d = p1.data - self.drag * derivative(p0.data)
#            p0 = Pulse('dragI(%s,%.5f)'%(p0.name, self.drag), p0d, chan=self.chans[0])
#            p1 = Pulse('dragQ(%s,%.5f)'%(p1.name, self.drag), p1d, chan=self.chans[1])

        return Combined([p0, p1])

# A gaussian b / d / sqrt(pi/2) * exp(-2(x/d)**2) has an area <b> and fw at exp(-0.5) of <d>

class LengthRotation(object):
    '''
    A rotation with an angle that is controlled using the pulse length.
    '''
    def __init__(self, base, amp, pi_len, chans=(0,1)):
        self.base = base
        self.amp = amp
        self.pi_len = pi_len
        self.chans = chans

    def set_pi(self, val):
        self.pi_len = val

    def __call__(self, alpha, phase):
        plen = alpha / np.pi * self.pi_len
        if plen < 0:
            plen = -plen
            amp = -self.amp
        else:
            amp = self.amp
        p0 = self.base(plen, amp * np.cos(phase), chan=self.chans[0])
        p1 = self.base(plen, amp * np.sin(phase), chan=self.chans[1])
        return Combined([p0, p1])

class GaussianRotation(object):
    '''
    A Gaussian pulse with a given area.

    Algorithm:
    - Calculate area, <area_frac> corresponds to area actually generated,
      because the pulse is chopped in time.
    - Try pulse with maximum intensity
    - If shorter than minimum length, decrease amplitude
    '''

    def __init__(self, pi_area, smin, smax, hmin, hmax, area_frac=1.0, chans=(1,2)):
        self.pi_area = pi_area
        self.smin = smin
        self.smax = smax
        self.hmin = hmin
        self.hmax = hmax
        self.area_frac = area_frac
        self.chans = chans

    def set_pi(self, val):
        self.pi_area = val

    def __call__(self, alpha, phase):
        area = (alpha / np.pi) * self.pi_area / self.area_frac

        sigma = area / np.sqrt(np.pi) / self.hmax
        if sigma < self.smin:
            sigma = self.smin
        if sigma > self.smax:
            sigma = self.smax
            raise Exception('Maximum Gaussian width exceeded')

        h = area / np.sqrt(np.pi) / sigma
        amp1 = h * np.cos(phase)
        amp2 = h * np.sin(phase)
        if np.abs(amp1) > 1 or np.abs(amp2) > 1:
            raise Exception('Amplitude >1 required for rotation')
        p0 = Gaussian(sigma, amp1, chan=self.chans[0])
        p1 = Gaussian(sigma, amp2, chan=self.chans[1])
        return Combined([p0, p1])

class DetunedSum(object):
    '''
    A generator for pulses containing a sum of one or more detuned rotation
    pulses in a single pair of I/Q channels.
    '''

    LAST_ID = 0

    def __init__(self, base, sigma, chans=(1,2), **kwargs):
        self.base = base
        self.sigma = sigma
        self.chans = chans
        self.kwargs = kwargs
        self._pulses = []

    def add(self, amp, period, phases=(0,-np.pi/2), IQamps=(1,1)):
        self._pulses.append((amp, period, phases, IQamps))

    def get_pulse(self, phi0=0):
        '''
        Return a Combined pulse with activity in both I/Q channels.
        '''
        bufs = {}
        for amp, period, phases, IQamps in self._pulses:
            pulse = self.base(self.sigma, amp, chan='_0', **self.kwargs).generate(0, '_0')
            buflen = len(pulse.get_data())
            phis = 2 * np.pi * np.arange(0, buflen) / period + phi0
            for ichan, chan in enumerate(self.chans):
                data = pulse.get_data() * np.cos(phis + phases[ichan]) * IQamps[ichan]
                if chan not in bufs:
                    bufs[chan] = data
                else:
                    bufs[chan] += data

        items = []
        for ichan, chan in enumerate(self.chans):
            name = 'detsum%d_ch%s' % (DetunedSum.LAST_ID, chan)
            items.append(Pulse(name, bufs[chan], chan=chan))

        DetunedSum.LAST_ID += 1
        return Combined(items)

    def __call__(self, phi0=0):
        return self.get_pulse(phi0=phi0)

class DetunedGaussians(DetunedSum):
    '''
    A generator for pulses containing one or more detuned Gaussians in a single
    pair of I/Q channels.
    '''

    def __init__(self, sigma, chans=(1,2), **kwargs):
        self._sigma = sigma
        super(DetunedGaussians, self).__init__(Gaussian, **kwargs)

    def add_gaussian(self, amp, period, phases=(0,-np.pi/2), area=None):
        '''
        Add Gaussian description, phases contains the I and Q phases.
        If area is specified, amplitude will be calculated.
        '''
        if area is not None:
            amp = area / np.sqrt(np.pi) / self.sigma
        super(DetunedGaussians, self).add(amp, period, phases)

#########################################
# Pulse modifiers
#########################################

class Chirp(Pulse):
    '''
    Add frequency chirp to a pulse.
    '''

    def __init__(self, p, deltaf, chan):
        name = 'chirp(%03d,%s)' % (round(deltaf*1000), p.get_name(),)
        ys = p.get_data()
        xs = np.arange(len(ys))
        alpha = deltaf * 2 * np.pi / len(ys)
        ys = ys * np.sin(0.5 * alpha * xs**2)
        super(Chirp, self).__init__(name, ys, chan=chan)

#########################################
# More fancy pulses
#########################################

class Slepian(Pulse):
    '''
    Slepian window pulse, width <w>, amplitude <a>, bw <bw>
    '''

    def __init__(self, w, a, bw=0.3, chan=1):
        name = 'slepian(%.3f,%.1f,%.5f)' % (bw, w, a)
        w = np.ceil(w)
        ys = a * scipy.signal.slepian(w, bw)
        super(Slepian, self).__init__(name, ys, chan=chan)

class Kaiser(Pulse):
    '''
    Kaiser window pulse, width <w>, amplitude <a>, alpha (2)
    '''

    def __init__(self, w, a, alpha=2, chan=1):
        name = 'kaiser(%.3f,%.1f,%.5f)' % (alpha, w, a)
        w = np.ceil(w)
        xs = np.arange(w)
        ys = a * scipy.special.i0(np.pi * alpha * np.sqrt(1 - (2.0 * xs / (w - 1) - 1)**2))
        ys /= scipy.special.i0(np.pi * alpha)
        super(Kaiser, self).__init__(name, ys, chan=chan)

class CosinePulse(Pulse):
    def __init__(self, name, w, h, table, chan=1, norm=None):
        w = int(np.ceil(w))
        xs = np.arange(w) / (w - 1.0) * 2 * np.pi
        buf = table[0] * np.ones_like(xs)
        for i in range(1, len(table)):
            buf += table[i] * np.cos(i * xs)
        if norm is None:
            norm = np.max(np.abs(buf))
        self.norm = norm
        buf = buf / norm * h
        Pulse.__init__(self, name, buf, chan=chan)

class Hanning(CosinePulse):
    '''
    Hann / Hanning window pulse: 0.5 * (1 + cos(x))
    Width <w>, amplitude <a>
    '''

    def __init__(self, w, a, chan=1):
        name = 'hanning(%.1f,%.5f)' % (w, a)
        T = [0.5, -0.5]
        super(Hanning, self).__init__(name, w, a, T, chan=chan)

class Blackman(CosinePulse):
    '''
    Blackman window with width <w>, amplitude <a> and alpha <alpha> (0.16).
    '''

    def __init__(self, w, a, alpha=0.16, chan=1):
        name = 'blackman(%.5f,%.1f,%.5f)' % (alpha, w, a)
        T = [(1-alpha)/2.0, -0.5, alpha/2.0]
        super(Blackman, self).__init__(name, w, a, T, chan=chan)

class BlackmanNutall(CosinePulse):
    '''
    Blackman window with width <w> and amplitude <a>.
    '''

    def __init__(self, w, a, chan=1):
        name = 'blackmannutall(%.1f,%.5f)' % (w, a)
        T = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
        super(BlackmanNutall, self).__init__(name, w, a, T, chan=chan)

class BlackmanHarris(CosinePulse):
    '''
    Blackman-Harris window with width <w> and amplitude <a>.
    '''

    def __init__(self, w, a, chan=1):
        name = 'blackmanharris(%.1f,%.5f)' % (w, a)
        T = [0.35875, -0.48829, 0.14128, -0.01168]
        super(BlackmanHarris, self).__init__(name, w, a, T, chan=chan)

class FlatTop(CosinePulse):
    '''
    FlatTop window with width <w> and amplitude <a>.
    Has a reasonably flat spectral content.
    '''

    def __init__(self, w, a, chan=1):
        name = 'flattop(%.1f,%.5f)' % (w, a)
        T = [1, -1.93, 1.29, -0.388, 0.028]
        super(FlatTop, self).__init__(name, w, a, T, chan=chan)

class Parabola(Pulse):
    def __init__(self, w, a, chan=1, power=2, **kwargs):
        ts = np.arange(w)
        ys = a * (1 - np.abs((2*ts/float(w-1) - 1))**power)
        name = 'parabola(%.2f,%.5f,%d)' % (w, a, power)
        super(Parabola, self).__init__(name, ys, chan=chan, **kwargs)

class UburpPulse(CosinePulse):
    TABLE = (
        +0.27, -1.42, -0.33, -1.72,
        +4.47, -1.33, -0.04, -0.34,
        +0.50, -0.33, +0.18, -0.21,
        +0.24, -0.14, +0.07, -0.06,
        +0.06, -0.04, +0.03, -0.03, +0.02,
    )

    def __init__(self, w, h, chan=1):
        name = 'uburp(%.1f,%.5f)' % (w, h)
        CosinePulse.__init__(self, name, w, h, self.TABLE, chan=chan)

class UburpRotation:
    def __init__(self, pi_area, w, chan=1):
        self.pi_area = pi_area
        self.w = w
        self.chan = chan

    def __call__(self, alpha):
        area = alpha / np.pi * self.pi_area
        return UburpPulse(self.w, area, chan=self.chan)

class SQPulse(CosinePulse):
    TABLE = {
        'S1': {
            1: (0.25, -1.8963102551, 1.1337663752, 0.5125438801),
            2: (0.50, -1.2053193822, 0.4796467863, 0.2256725959),
            4: (1.00, -0.0237996956, -0.6226198703, -0.3535804341),
        },
        'S2': {
            1: (0.25, -1.9049987341, 1.9858884053, 0.1063314501, -0.4372211211),
            2: (0.50, -1.1950692860, 0.7841592117, 0.0737326786, -0.1628226043),
            4: (1.00, -0.0294359406, -1.1741824154, -0.2097531295, 0.4133714855),
        },
        'Q1': {
            1: (0.25, -1.8948543589, 0.5873324062, 0.5970352560, 0.4604866969),
            2: (0.50, -1.1374072085, 1.5774920785, -0.6825355002, -0.2575493698),
            4: (1.00, 2.1406171699, -2.3966480505, -0.6474844418, -0.0964846776),
        },
        'Q2': {
            1: (0.25, -2.1145695246, 0.6415685732, 1.6854185871, 0.4511145740, -0.9135322049),
            2: (0.50, -1.0964843348, 1.5308987822, -1.1472441408, 0.0025173181, 0.2103123753),
            4: (1.00, 1.4818894659, -2.6971749102, -0.4384679067, 0.3434236044, 0.3103297466),
        },
    }

    def __init__(self, w, h, ptype, npi_2, chan=1, norm=None):
        name = 'SQ(%s,%d,%.1f,%.5f)' % (ptype, npi_2, w, h)
        table = self.TABLE[ptype][npi_2]
        CosinePulse.__init__(self, name, w, h, table, chan=chan, norm=norm)

class SQRotation:
    def __init__(self, w1, h1, w2, h2, ptype, chans=(1,2)):
        '''
        This will apply the norm of the pi pulse to others
        '''
        self.w1 = w1
        self.h1 = h1
        self.w2 = w2
        self.h2 = h2
        self.ptype = ptype
        self.chans = chans

    def __call__(self, alpha, phase):
        if alpha < 0:
            alpha *= -1
            sign = -1
        else:
            sign = 1

        w, h, npi_2 = 0, 0, 0
        if np.abs(alpha - 0) < DELTA:
            return Sequence()
        elif np.abs(alpha - np.pi/2) < DELTA:
            w, h = self.w1, self.h1
            npi_2 = 1
        elif np.abs(alpha - np.pi) < DELTA:
            w, h = self.w1, self.h1
            npi_2 = 2
        else:
            raise ValueError('Unable to do SQ rotation with angle %s*pi' % (alpha / np.pi,))
        p0 = SQPulse(w, sign * h * np.cos(phase), self.ptype, npi_2, chan=self.chans[0])
        p1 = SQPulse(w, sign * h * np.sin(phase), self.ptype, npi_2, chan=self.chans[1])
        return Combined([p0, p1])

#########################################
# Composite pulse schemes
#########################################

class CorpseRotation:
    def __init__(self, base, **kwargs):
        self.n1 = kwargs.pop('n1', 1)
        self.n2 = kwargs.pop('n2', 1)
        self.n3 = kwargs.pop('n3', 0)
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        s = np.arcsin(np.sin(alpha / 2.0) / 2)
        a1 = self.n1 * 2 * np.pi + alpha / 2.0 - s
        a2 = self.n2 * 2 * np.pi - 2 * s
        a3 = self.n3 * 2 * np.pi + alpha / 2.0 - s
#        print 'Angles: %.01f, %.01f, %.01f' % (np.rad2deg(a1), np.rad2deg(a2), np.rad2deg(a3))
        pulses = [
            self.base(a1, phase),
            self.base(a2, phase + np.pi),
            self.base(a3, phase),
        ]
        return Sequence(pulses, join=True)

class ShortCorpseRotation(CorpseRotation):
    def __init__(self, base, **kwargs):
        kwargs['n1'] = 0
        kwargs['n2'] = 1
        kwargs['n3'] = 0
        CorpseRotation.__init__(self, base, **kwargs)

class OffResonanceCorr:
    def __init__(self, base, **kwargs):
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        pr = self.base(alpha, phase)

        phi1 = np.arccos(-np.sin(alpha/2.0)**2/4)
        p0 = self.base(np.pi, phase + np.pi - phi1)
        p1 = self.base(np.pi, phase - phi1)
        p2 = self.base(np.pi, phase + phi1 + np.pi)
        p3 = self.base(np.pi, phase + phi1)

        phi2 = -np.arcsin(np.sin(alpha)/4)
        p4 = self.base(phi2, phase)
        p5 = self.base(2*phi2, phase + np.pi)
        p6 = self.base(phi2, phase)
        pulses = [pr, p0, p1, p2, p3, p4, p5, p6]
        return CompositePulse(pulses)

# Corrected pi pulse, seems to behave better than single pi pulse for detuning
class OffResonanceCorr2:
    def __init__(self, base, **kwargs):
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        phi1 = np.arccos(-1.0/4)
        p0 = self.base(np.pi, phase)
        p1 = self.base(np.pi, phase + phi1)
        p2 = self.base(2 * np.pi, phase + 3 * phi1)
        p3 = self.base(np.pi, phase + phi1)
        p4 = self.base(np.pi, phase + np.pi - phi1)
        p5 = self.base(np.pi, phase - phi1)
        p6 = self.base(np.pi, phase + np.pi + phi1)
        p7 = self.base(np.pi, phase + phi1)
        pulses = [p0, p1, p2, p3, p4, p5, p6, p7]
        return CompositePulse(pulses)

class ReducedCinSK:
    def __init__(self, base, **kwargs):
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        s = np.arcsin(np.sin(alpha / 2.0) / 2)
        a1 = 2 * np.pi + alpha / 2.0 - s
        a2 = 2 * np.pi - 2 * s
        a3 = alpha / 2.0 - s

        p0 = self.base(a1, phase)
        p1 = self.base(a2, phase + np.pi)
        p2 = self.base(a3, phase)

        phi_sk1 = np.arccos(-alpha / 4 / np.pi)
        p3 = self.base(2 * np.pi, phase - phi_sk1)
        p4 = self.base(2 * np.pi, phase + phi_sk1)
        pulses = [p0, p1, p2, p3, p4]
        return Sequence(pulses)

class BB1Rotation:
    '''
    Robust against pulse length errors.
    '''
    def __init__(self, base, **kwargs):
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        phi1 = np.arccos(-alpha / 4 / np.pi)
        phi2 = 3 * phi1
        pulses = [
            self.base(np.pi, phi1 + phase),
            self.base(2 * np.pi, phi2 + phase),
            self.base(np.pi, phi1 + phase),
            self.base(alpha, phase),
        ]
        return Sequence(pulses)

class SK1Rotation:
    '''
    Robust against pulse length errors
    '''
    def __init__(self, base, **kwargs):
        self.base = base
        self.kwargs = kwargs

    def __call__(self, alpha, phase):
        phi1 = np.arccos(-alpha / 4 / np.pi)
        pulses = [
            self.base(alpha, phase),
            self.base(2 * np.pi, phase - phi1),
            self.base(2 * np.pi, phase + phi1),
        ]
        return Sequence(pulses)

class ComposedZRotation:
    def __init__(self, base):
        self.base = base

    def __call__(self, alpha):
        pulses = [
            self.base(np.pi / 2, 0),
            self.base(alpha, np.pi / 2),
            self.base(-np.pi / 2, 0),
        ]
        return Sequence(pulses)
