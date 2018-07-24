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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    i = Interpolator([0, 0.5, 1], [0, 0.3, 0.5])
    xs = np.linspace(0, 1, 41)
    ys = [i(x) for x in xs]
    plt.plot(xs, ys)

