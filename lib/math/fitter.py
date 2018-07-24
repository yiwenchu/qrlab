# fitter.py, fitting functino helper class
# Reinier Heeres, 2014
#
# Uses python fitting functions in the fit_funcs sub-directory

import numpy as np
import importlib
import inspect
import lmfit
import os

# Make sure the dir we are in is in the path so we can find fit_funcs
import inspect
import sys
SRCDIR = os.path.split(os.path.abspath(inspect.getsourcefile(lambda _: None)))[0]
if SRCDIR not in sys.path:
    sys.path.append(SRCDIR)

class Fitter(object):
    '''
    Fitting helper class.

    Initialize as Fitter(<funcname>), where <funcname> specifies the python
    file in fit_funcs/ to use, i.e. Fitter('sine').

    Fitting function files should specify a function called 'func(xs, ...)'
    and optionally a function 'guess(xs, ys)' to estimate initial fitting
    parameters. See builtin_fit_funcs/sine.py for an example.

    Additional fit functions can be specified in a directory "fit_funcs"
    (which should contain an __init__.py so that import user_fit_funcs.bla
    works).
    '''

    def __init__(self, funcname):
        self.funcname = funcname
        try:
            self.module = importlib.import_module('fit_funcs.%s'%funcname)
        except:
            self.module = importlib.import_module('builtin_fit_funcs.%s'%funcname)
        if not hasattr(self.module, 'func'):
            raise Exception('Module %s does not contain a fit function (called "func")')

        self.fit_func = getattr(self.module, 'func')
        self.fit_args = inspect.getargspec(self.fit_func)
        if self.fit_args[0][0] != 'xs':
            raise Exception('First argument of fit functions should be "xs"')
        if len(self.fit_args[0]) > 1 and self.fit_args[0][1] == 'ys':
            self.is_2d = True
        else:
            self.is_2d = False

        self.guess_func = getattr(self.module, 'guess', None)

        self.fit_params = None

    def residual_func(self, params, xs, ys, zs=None):
        kwargs = {k: params[k].value for k in params.keys()}
        if self.is_2d:
            if zs is None:
                raise Exception('2D function but zs not specified')
            est = self.fit_func(xs, ys, **kwargs)
            residuals = (zs - est).flatten()
        else:
            est = self.fit_func(xs, **kwargs)
            residuals = ys - est

        if self.stderrs == None:
            return residuals
        else:
            return np.sqrt(residuals**2/self.stderrs**2)

    def eval_func(self, xs=None, ys=None, params=None, **kwargs):
        '''
        Evaluate fit function at <xs> (and <ys> if 2d).
        <xs> (and <ys> if 2d) will be taken from last fit if None.
        Uses parameters from last fit if <params> is None.
        Keyword arguments override these parameters.
        '''
        if params is None:
            if self.fit_params is not None:
                params = self.fit_params
            else:
                params = {}

        if xs is None:
            xs = self.last_xs
        if self.is_2d and ys is None:
            ys = self.last_ys

        f_kwargs = {k: v.value for k, v in params.items()}
        f_kwargs.update(kwargs)

        if self.is_2d:
            if ys is None:
                raise Exception('2D function but ys not specified')
            return self.fit_func(xs, ys, **f_kwargs)
        else:
            return self.fit_func(xs, **f_kwargs)

    def test_values(self, xs, ys=None, noise_amp=0.5, **kwargs):
        '''
        Return the fit function evaluated at <xs> with additional Gaussian
        noise of amplitude <noise_amp>.
        Keyword arguments are passed on to the fit function. If none are
        given it's default values are used.
        '''
        ret = self.eval_func(xs, ys, params={}, **kwargs)
        ret += np.random.normal(0, noise_amp, ret.shape)
        return ret

    def get_nargs(self):
        nargs = len(self.fit_args[0])
        if self.is_2d:
            return nargs - 2
        else:
            return nargs - 1

    def get_parameter_dict(self, xs, ys, zs=None):
        nargs = self.get_nargs()

        if self.guess_func:
            if self.is_2d:
                params = self.guess_func(xs, ys, zs)
            else:
                params = self.guess_func(xs, ys)
            if len(params) != nargs:
                raise Exception('guess() function did not return the right amount of parameters (expected %s, got %s)' % (nargs, params))
            return params

        if len(self.fit_args[3]) != nargs:
            raise Exception('Fit function %s does not specify enough default parameters' % self.funcname)

        ret = {self.fit_args[0][i+1]: self.fit_args[3][i] for i in range(nargs)}
        return ret

    def get_lmfit_parameters(self, xs, ys, zs=None):
        pdict = self.get_parameter_dict(xs, ys, zs=zs)
        p = lmfit.Parameters()
        for k, v in pdict.items():
            p.add(k, value=v)
        return p

    def get_fit_kwargs(self):
        '''
        Return dictionary with <parameter> <value> pairs from last fit.
        '''
        return {k: self.fit_params[k].value for k in self.fit_params}

    def perform_lmfit(self, xs, ys, zs=None, p=None, print_report=True, 
        plot=False, plot_guess=True, stderrs=None, **kwargs):
        '''
        Perform fit using lmfit.

        Optionally the arguemnt <p> specifies the lmfit.Parameters, if None
        get_lmfit_parameters() is called.

        <print_report> specifies whether the fit report will be printed.
        <plot> specifies whether the data and fit will be plotted.
        <stderrs> are the error bars on the data points. if specfied,
            the residuals will be weighed accordingly. 

        Keyword arguments specify parameters that should not be allowed to
        vary in the fitting process and be fixed at the given value.
        '''

        self.stderrs = stderrs

        if p is None:
            p = self.get_lmfit_parameters(xs, ys, zs=zs)
            for key, val in kwargs.items():
                p[key].value = val
                p[key].vary = False
                
        self.last_xs = xs
        if self.is_2d:
            self.last_ys = ys

        if plot:
            import matplotlib.pyplot as plt
            kwargs = {k: p[k].value for k in p.keys()}
            plt.figure()
            if self.is_2d:
                plt.pcolor(xs, ys, zs)
                if plot_guess:
                    zs_guess = self.fit_func(xs, ys, **kwargs)
                    plt.contour(xs, ys, zs_guess)
            else:
                plt.plot(xs, ys, 'ks', label='Data')
                ys_guess = self.fit_func(xs, **kwargs)
                plt.plot(xs, ys_guess, label='Estimate')

        result = lmfit.minimize(self.residual_func, p, args=(xs, ys, zs))
        self.fit_params = p

        if print_report:
            print lmfit.fit_report(p)
        if plot:
            if self.is_2d:
                plt.contour(xs, ys, self.eval_func(xs, ys))
            else:
                plt.plot(xs, self.eval_func(xs), label='Fit')
                plt.legend()

        return result

    def test(self, **kwargs):
        xr = getattr(self.module, 'TEST_RANGE', (0, 1))
        xs = np.linspace(xr[0], xr[1], 101)
        params = getattr(self.module, 'TEST_PARAMS', {})
        params.update(kwargs)
        if self.is_2d:
            xs, ys = np.meshgrid(xs, xs)
            zs = self.test_values(xs, ys, **params)
            self.perform_lmfit(xs, ys, zs, plot=True)
        else:
            ys = self.test_values(xs, **params)
            self.perform_lmfit(xs, ys, plot=True)

    @staticmethod
    def get_fit_functions():
        fns = os.listdir(os.path.join(SRCDIR, 'builtin_fit_funcs'))
        ret = []
        for fn in fns:
            fn, ext = os.path.splitext(fn)
            if ext == '.py' and not fn.startswith('__') and fn != 'common':
                ret.append(fn)
        return ret

if __name__ == '__main__':
    print 'Available fitting functions: %s' % (Fitter.get_fit_functions(),)

    for fn in Fitter.get_fit_functions():
        print 'Testing %s' % (fn,)
        try:
            f = Fitter(fn)
            f.test()
        except Exception, e:
            print 'Function %s failed: %s' % (fn, e)

