import numpy as np

class Interpolator:

    def __init__(self, xs=None, ys=None, order=2):
        if xs is not None:
            self.set_interpolation(xs, ys, order)

    def set_interpolation(self, xs, ys, order=2):
        self.params = np.polyfit(xs, ys, order)[::-1]

    def __call__(self, arg):
        y = 0
        x = 1
        for p in self.params:
            y += x * p
            x *= arg
        return y

class AmpGen:
    '''
    Rotation amplitude generator.

    <amp_spec> is either a single value representing the pi amplitude
    or a list of values. A list of 2 values is interpreted as the 0.5pi
    and pi rotation amplitudes, otherwise it describes an angle (in units
    of pi) and the required amplitude, e.g.:

        (0.25, 0.2, 1.0, 0.5)

    to specify an amplitude 0.2 for a rotation of 0.25pi and amplitude 0.5
    for a rotation of pi.
    '''

    def __init__(self, amp_spec=None):
        self.interp = Interpolator()
        if amp_spec is not None:
            self.set_amp_spec(amp_spec)

    def set_amp_spec(self, amp_spec):
        try:
            alen = len(amp_spec)
        except:
            amp_spec = [amp_spec,]
            alen = 1

        if alen == 1:
            self.interp.set_interpolation([0, np.pi], [0, amp_spec[0]], 1)
        elif alen == 2:
            self.interp.set_interpolation([0, 0.5*np.pi, np.pi], [0, amp_spec[0], amp_spec[1]], 2)
        else:
            xs = np.array([0,] + list(amp_spec[::2])) * np.pi
            ys = np.array([0,] + list(amp_spec[1::2]))
            self.interp.set_interpolation(xs, ys, order=len(ys)-1)

    def __call__(self, arg):
        s = np.sign(arg)
        arg = np.abs(arg)
        return s * self.interp(arg)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vs = [
        (0.3, 0.5),
        (0.25, 0.2, 1.0, 0.5)
    ]
    for v in vs:
        ag = AmpGen(v)
        xs = np.linspace(-np.pi, np.pi, 41)
        ys = [ag(x) for x in xs]
        plt.plot(xs/np.pi, ys)
