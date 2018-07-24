# minimizer.py, simple parameter optimization scheme.
# Reinier Heeres, 2014

import numpy as np
import matplotlib.pyplot as plt

class Parameter:
    def __init__(self, name, value, vrange, minstep=0):
        self.name = name
        self.value = value
        self.vrange = vrange
        self.minstep = minstep

class Minimizer:
    """
    Straight-forward parameter optimizer.

    Optimization strategy:
    - Repeat <n_it> times (default: 5):
        - For each parameter:
            - sweep parameter from (value - range/2) to (value + range/2) in
              <n_eval> steps (default: 6) and evaluate function.
            - determine best parameter value.
            - reduce range by a factor <range_div> (default: 2.5).

    Specify optimization function <func>, it's arguments <args> and keyword
    arguments <kwargs>. The function should accept a dictionary of Parameter
    objects as it's first argument. It should return a scalar.
    """

    def __init__(self, func, args=(), kwargs={},
                 n_eval=6, n_it=5, range_div=2.5,
                 verbose=False, plot=False):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.params = {}
        self.n_eval = n_eval
        self.n_it = n_it
        self.verbose = verbose
        self.range_div = range_div
        self.plot = plot

    def add_parameter(self, p):
        self.params[p.name] = p

    def minimize(self, min_step=None):
        if self.plot:
            fig, [axes_list, min_list] = plt.subplots(2, len(self.params),
                                          sharex=False, sharey=False)

        for i_it in range(self.n_it):
            for i_p, (pname, p) in enumerate(self.params.iteritems()):
                p_val0 = p.value
                p_vals = np.linspace(p_val0-p.vrange/2, p_val0+p.vrange/2, self.n_eval)

                if np.abs(p_vals[1]-p_vals[0]) < p.minstep:
                    p_vals = np.arange(p_vals[0],p_vals[-1]+0.00001,p.minstep)
                if len(p_vals) == 1:
                    break

                vs = []
                for p_val in p_vals:
                    p.value = p_val
                    vs.append(self.func(self.params, *self.args, **self.kwargs))
                vs = np.array(vs)
                imin = np.argmin(vs)
                p.value = p_vals[imin]
                if self.verbose:
                    print 'It%d, p%s --> f(%.03f) = %.01f [delta: %.05f]' % \
                        (i_it, pname, p.value, vs[imin], (p_vals[1]-p_vals[0]))
                if self.plot:
                    axes_list[i_p].plot(p_vals, vs)

                    min_list[i_p].plot(p_vals, vs)
                    min_list[i_p].set_ylim(vs.min(), vs.max())
                    min_list[i_p].set_xlim(p_vals.min(), p_vals.max())
                    fig.canvas.draw()
                p.vrange /= self.range_div

        # If live optimizing, set final parameters
        self.func(self.params, *self.args, **self.kwargs)

        return self.params
