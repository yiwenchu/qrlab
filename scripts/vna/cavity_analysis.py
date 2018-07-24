from matplotlib import pyplot as plt
import numpy as np
from lmfit import minimize, Parameters
from numpy import random
from scipy import optimize
import copy


def to_dB(x):
    return 20*np.log10(x)


def from_dB(x):
    return 10**(x/20.)


def fit_hfss_transmission(filename, lossy=False, show=False):
    with open(filename) as f:
        headers = f.readline()[1:-1].split('","')
        data = np.transpose([map(float, line.split(',')) for line in f.readlines()])
        freqs = data[0]*1e9
        res = {}
        for name, trace in zip(headers[1:], data[1:]):
            cutoff = trace.max() / 1e2
            region = np.where(trace > cutoff)[0]
            imin, imax = region[0], region[-1]
            freq_clip = freqs[imin:imax]
            trace_clip = trace[imin:imax]
            if 'mag' in name:
                params = fit_v_lorentzian(freq_clip, trace_clip, show=show)
            else:
                params = fit_db_lorentzian(freq_clip, trace_clip, show=show)
            res[name] = {n: params[n].value for n in ("qi", "qc", "f0")}
            res[name]['q'] = 1/(1/res[name]['qc'] + 1/res[name]['qi'])
        return res


def open_text_data(filename, delim=None):
    with open(filename) as f:
        try:
            return np.transpose([map(float, line.split(delim)) for line in f.readlines()])
        except ValueError:
            f.seek(0)
            headers = f.readline()
            print headers
            return np.transpose([map(float, line.split(delim)) for line in f.readlines()])


def send_to_igor(arr, wavenames, overwrite=True):
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = np.array([arr])
        wavenames = [wavenames]
    if len(wavenames) != arr.shape[1]:
        arr = arr.transpose()
    if len(wavenames) != arr.shape[1]:
        raise ValueError('wavenames not of length of the data array')
    import tempfile

    f = tempfile.NamedTemporaryFile(delete=False)
    fn = f.name
    fn = fn.replace('\\', ':').replace('::', ':')
    np.savetxt(f, arr, delimiter=',', header=','.join(wavenames))
    f.close()
    args = "/J/D/W/K=0/A"
    if overwrite:
        args += '/O'
    cmd = 'LoadWave%s "%s"'%(args, fn)
    igor_do(cmd)


def plot_in_igor(x, y, xname, yname, overwrite=False):
    xname = igorify_name(xname)
    yname = igorify_name(yname)
    send_to_igor([x, y], [xname, yname], overwrite=overwrite)
    igor_do(
        'Display %s vs %s'%(yname, xname),
        'Label left "%s"'%yname,
        'Label bottom "%s"'%xname
    )


def igor_do(*cmds):
    for cmd in cmds:
        print cmd
        import win32com.client

        igor = win32com.client.Dispatch("IgorPro.Application")
        igor.Execute(cmd)


def igorify_name(n):
    if len(n) > 32:
        print n, 'is too long shortening to 32 chars'
        n = n[:31]
    for c in set(n):
        if not c.isalnum():
            n = n.replace(c, '_')
    while n.endswith('_'):
        n = n[:-1]
    print n
    return n


def plot_complex_data(freqs, mags, phases):
    s21 = mags*np.exp(1j*np.pi*phases/180.)
    s21inv = 1/s21
    plt.figure(2)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(np.real(s21inv), np.imag(s21inv))
    plt.axes().set_aspect('equal')
    plt.axes().grid(True)


def do_fit(xpts, data_pts, fit_func, params, show=False, force=False):
    """
    params is a dictionary of dictionaries containing optionally value, min, and max fields
    if show is enabled, matplotlib is used to compare the data, initial guess, and fitted result
    """
    data_pts = np.array(data_pts)
    param_names = params.keys()
    is_complex = data_pts.imag.any()
    if is_complex:
        print 'Complex Data Found'

    def apply_params(params, xs):
        values = {s: params[s].value for s in param_names}
        return fit_func(xs, **values)

    if is_complex:
        residual = lambda params, xs, ys: np.abs(ys - apply_params(params, xs))**2
    else:
        residual = lambda params, xs, ys: ys - apply_params(params, xs)

    if isinstance(params, Parameters):
        nlm_params = params
    else:
        nlm_params = Parameters()
        for name, kwargs in params.items():
            nlm_params.add(name, **kwargs)
    initial_params = copy.deepcopy(nlm_params)

    if not force and (apply_params(initial_params, xpts) == np.nan).any():
        raise ValueError("Function produced NaNs on initial params")

    minimize(residual, nlm_params, args=(xpts, data_pts))
    fitted_data = apply_params(nlm_params, xpts)

    if show:
        for n in param_names:
            print n, 'initial: ', initial_params[n].value, 'fitted: ', nlm_params[n].value
        if is_complex:
            plt.figure(0)
            plt.plot(xpts, np.abs(data_pts), label='data')
            plt.plot(xpts, np.abs(apply_params(initial_params, xpts)), label='initial')
            plt.plot(xpts, np.abs(apply_params(nlm_params, xpts)), label='fitted')
            plt.legend()
            plt.figure(1)
            plt.plot(xpts, np.angle(data_pts), label='data')
            plt.plot(xpts, np.angle(apply_params(initial_params, xpts)), label='initial')
            plt.plot(xpts, np.angle(apply_params(nlm_params, xpts)), label='fitted')
            plt.legend()
        else:
            print data_pts
            plt.plot(xpts, data_pts, label='data')
            plt.plot(xpts, apply_params(initial_params, xpts), label='initial')
            plt.plot(xpts, apply_params(nlm_params, xpts), label='fitted')

        plt.legend()
        if show is True:
            plt.show()
        else:
            plt.title(show)
            plt.savefig("fit_plot_%s.png"%show)
            plt.clf()

    return nlm_params, fitted_data


def param(value, min, max):
    'A parameter which can be passed as a keyword argument to do_fit'
    return {'value': value, 'min': min, 'max': max}

def complex_v_hanger(f, f0, qi, qc, scale):
    q = 1/(1/qi + 1/qc)
    x = (f - f0)/f0
    return scale*(1 - (q/qc) / (1 + 2j*q*x))

def v_hanger(f, f0, qi, qc, scale):
    return np.abs(complex_v_hanger(f, f0, qi, qc, scale))

def asymmetric_complex_v_hanger(f, f0, qi, qcr, qci, scale, phase=0):
    'See Kurtis: Asymmetric Hanger Equations'
    qc = qcr + 1j*qci
    x = (f - f0)/f0
    return np.exp(1j*phase)*scale*qc*(2*qi*x - 1j)/(2*qi*qc*x - 1j*(qi + qc))

def asymmetric_v_hanger(f, f0, qi, qcr, qci, scale):
    return np.abs(asymmetric_complex_v_hanger(f, f0, qi, qcr, qci, scale))

def asymmetric_db_hanger(f, f0, qi, qcr, qci, offset):
    return to_dB(asymmetric_v_hanger(f, f0, qi, qcr, qci, from_dB(offset)))

def fit_v_hanger(f, s21, show=False):
    params = asymmetric_hanger_guess(f, s21)
    params['qc'] = params['qcr']
    params.pop('qcr')
    params.pop('qci')
    print params
    return do_fit(f, s21, v_hanger, params, show=show)

def fit_shifted_circle(x, y, x_m=0, y_m=0):
    'From: http://wiki.scipy.org/Cookbook/Least_Squares_Circle'

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(*center_2b)
    R_2b = Ri_2b.mean()
    #residu_2b    = sum((Ri_2b - R_2b)**2)
    return xc_2b, yc_2b, R_2b

def asymmetric_hanger_guess(fpts, vpts):
    scale_guess = (vpts[0] + vpts[-1]) / 2.
    vpts /= scale_guess
    i0 = np.argmin(vpts)
    f0_guess = fpts[i0]
    delta_i_guess = min(i0, len(fpts) - i0)
    delta_f_guess = delta_i_guess*(fpts[1] - fpts[0])/10
    q_total_guess = f0_guess/delta_f_guess
    qi_guess = q_total_guess/vpts[i0]
    qc_guess = 1/(1/q_total_guess - 1/qi_guess)
    vpts *= scale_guess

    assert all(q > 0 for q in (q_total_guess, qi_guess, qc_guess))
    assert not (asymmetric_complex_v_hanger(fpts, f0_guess, qi_guess, qc_guess, 0, scale_guess) == np.nan).any()

    return {
        'f0': param(f0_guess, fpts[0], fpts[-1]),
        'qi': param(qi_guess, qi_guess/10., qi_guess*10),
        'qcr': param(qc_guess, qc_guess/10., qc_guess*10),
        'qci': param(0, -qc_guess, qc_guess),
        'scale': param(scale_guess, scale_guess/2., 2*scale_guess)
    }

def fit_asymmetric_v_hanger(fpts, vpts, show=False):
    fpts, vpts = np.array(fpts), np.array(vpts)
    params = asymmetric_hanger_guess(fpts, vpts)
    return do_fit(fpts, vpts, asymmetric_v_hanger, params, show=show)


def fit_asymmetric_db_hanger(fpts, db_pts, show=False):
    params, fitted_data = fit_asymmetric_v_hanger(fpts, from_dB(db_pts), show)
    return params, to_dB(fitted_data)


def simple_lorentzian(x, x0, width, max):
    return max*width**2 / ((x - x0)**2 + width**2)

def simple_lorentzian_guess(x, y):
    w_guess = (x[-1] - x[0]) / 8
    max_guess = max(y)
    return {
        'x0': param(x[np.argmax(y)], x[0], x[-1]),
        'width': param(w_guess, w_guess/10, 10*w_guess),
        'max': param(max_guess, max_guess/2, 2*max_guess)
    }

def fit_simple_lorentzian(x, y, show=False):
    x, y = np.array(x), np.array(y)
    params = simple_lorentzian_guess(x, y)
    return do_fit(x, y, simple_lorentzian, params, show=show)

def complex_v_lorentzian(f, f0, qi, qc):
    return 1/(1 + (qc/qi) - 2j*qc*(f - f0)/f0)


def v_lorentzian(f, f0, qi, qc):
    return np.abs(complex_v_lorentzian(f, f0, qi, qc))


def db_lorentzian(f, f0, qi, qc):
    return to_dB(v_lorentzian(f, f0, qi, qc))


def lorentzian_guess(freqs, s21):
    i0 = np.argmax(s21)
    f0_guess = freqs[i0]
    fspan = freqs[-1] - freqs[0]
    q_guess = 10*f0_guess/fspan
    qc_guess = q_guess/s21[i0]
    #assert qc_guess > q_guess
    qi_guess = 1/(1/q_guess - 1/qc_guess)
    #check_relerror(s21[i0], power_lorentzian(f0_guess, f0_guess, qi_guess, qc_guess), .05)

    return {
        'f0': param(f0_guess, freqs[0], freqs[-1]),
        'qc': param(qc_guess, qc_guess/100, qc_guess*100),
        'qi': param(qi_guess, qi_guess/100, qi_guess*100),
    }


def fit_v_lorentzian(freqs, s21, show=False):
    params = lorentzian_guess(freqs, s21)
    return do_fit(freqs, s21, v_lorentzian, params, show=show)


def fit_db_lorentzian(freqs, s21, show=False):
    params, fitted_data = fit_v_lorentzian(freqs, from_dB(s21), show)
    return params, to_dB(fitted_data)


#def fit_complex_power_lorentzian(freqs, s21_mags, s21_phases, show=False, showall=True):
#    params_1 = fit_power_lorentzian(freqs, s21_mags, show=showall)
#
#    for p in params_1:
#        params_1[p].vary = False
#
#    fspan = freqs[-1] - freqs[0]
#    n_jumps = 0
#    i = 0
#    sign = 0
#
#    while i < len(s21_phases)-1:
#        if s21_phases[i] * s21_phases[i+1] < 0:
#            if not sign:
#                sign = np.sign(s21_phases[i+1] - s21_phases[i])
#            n_jumps += 1
#            i += 20
#        else:
#            i += 1
#
#    tau_guess = sign * np.pi * n_jumps / fspan
#    params_1.add('tau', value=tau_guess, min=tau_guess/2 - 2*np.pi, max=tau_guess*2 + 2*np.pi)
#    params_1.add('phi', value=0, min=-np.pi, max=np.pi)
#    s21 = np.sqrt(s21_mags) * np.exp((-1j * np.pi / 180) * s21_phases)
#
#    params_2 = do_fit(freqs, s21, complex_v_lorentzian, params_1, show=showall)
#
#    for p in params_2:
#        params_2[p].vary = True
#
#    return do_fit(freqs, s21, complex_v_lorentzian, params_1, show=show)


#def fit_complex_db_lorentzian(freqs, s21_db, s21_phases, show=False):
#    return fit_complex_power_lorentzian(freqs, from_dB(s21_db), s21_phases, show=show)


def check_relerror(x0, x1, tolerance):
    error = (x0 - x1)/x0
    assert (x0 - x1)/x0 < tolerance, "error %s between %s,%s greater than %s"%(error, x0, x1, tolerance)


def test_fit_asymm_db_hanger():
    fpts = np.linspace(9.1e9, 9.101e9, 500)
    f0 = 9.1005e9
    qi = 4e5
    qcr = 1e6
    qci = 5e5
    offset = -3
    scale = from_dB(offset)
    fake = asymmetric_db_hanger(fpts, f0, qi, qcr, qci, offset) + random.normal(0, .05, len(fpts))
    result = fit_asymmetric_db_hanger(fpts, fake, show=True)
    check_relerror(f0, result['f0'].value, .08)
    check_relerror(qi, result['qi'].value, .08)
    check_relerror(qcr, result['qcr'].value, .08)
    check_relerror(qci, result['qci'].value, .08)
    check_relerror(scale, result['scale'].value, .08)

def test_fit_lorentzian():
    fpts = np.linspace(9.1e9, 9.101e9, 500)
    f0 = 9.1005e9
    qi = 4e5
    qc = 1e6

    fake = db_lorentzian(fpts, f0, qi, qc) + random.normal(0, .05, len(fpts))
    result = fit_db_lorentzian(fpts, fake, show=True)
    check_relerror(f0, result['f0'].value, .08)
    check_relerror(qi, result['qi'].value, .08)
    check_relerror(qc, result['qc'].value, .08)


#test_fit_asymm_db_hanger()
#test_fit_lorentzian()

#def lorentzian(f0, df, t0, f):
#    return np.abs(t0 / (1 - (1j*(f - f0)/df))) ** 2
#
#def complex_lorentzian(f0, df, t0, t0a, tau, f):
#    return np.exp(1j*tau*f) * (t0*np.exp(1j*t0a)) / (1 - (1j*(f - f0)/df))
#
#def complex_voltage(s21_squared, angle):
#    return np.sqrt(s21_squared) * np.exp(-1j * angle * np.pi / 180)
#
#def complex_hanger(f0, qt, qc, a_mag, a_phase, tau, phi_0, freqs):
#    t0 = (qt / qc) * np.exp(1j * phi_0)
#    df = qt * (freqs - f0) / f0
#    coef = a_mag * np.exp(a_phase - 2j*np.pi*tau*freqs)
#    return coef * (1 - t0 / (1 + 2j*df))
#
#def hanger(f0, qt, qc, offset_t, freqs):
#    return np.abs(complex_hanger(f0, qt, qc, offset_t, 0, 0, 0, freqs))**2


#
#def fit_lor(s21, freqs, plot=False, errors=False, power=None, phase=None):
#    params = Parameters()
#
#    offset = freqs[0]
#    freqs = freqs - offset
#
#    t0 = np.sqrt(max(s21))
#    i0 = np.argmax(s21)
#    f0 = freqs[i0]
#    frange = freqs[-1] - freqs[0]
#    df = frange / 15
#
#    params.add('f0', value=f0, min=freqs[1], max=freqs[-1])
#    params.add('df', value=df, min=0., max=freqs[-1]-freqs[0])
#    params.add('t0', value=t0, min=0., max=np.sqrt(2*max(s21)))
#
#    def apply_lorentzian(p, f):
#        f0, df, t0 = (p[s].value for s in ('f0', 'df', 't0'))
#        return lorentzian(f0, df, t0, f)
#
#    def apply_complex_lorentzian(p, f):
#        f0, df, t0, t0a, tau = (p[s].value for s in ('f0', 'df', 't0', 't0a', 'tau'))
#        return complex_lorentzian(f0, df, t0, t0a, tau, f)
#
#    def residual(p, f, s21):
#        return s21 - apply_lorentzian(p, f)
#
#    def complex_residual(p, f, v):
#        return np.abs(v - apply_complex_lorentzian(p, f))
#
#
#    def plot_params(params, use_phase=False, title=None):
#        if plot:
#            plt.figure(1)
#            plt.plot(freqs, s21)
#            plt.plot(freqs, apply_lorentzian(params, freqs))
#            if title:
#                plt.title(title)
#            if use_phase:
#                plt.figure(2)
#                plt.plot(freqs, phase)
#                plt.plot(freqs, get_angle(apply_complex_lorentzian(params, freqs)))
#                if title:
#                    plt.title(title)
#            plt.show()
#
#    #plot_params(params, title='Initial Mag')
#    minimize(residual, params, args=(freqs, s21), method='leastsq')
#    if phase is None:
#        plot_params(params, title='Fitted Mag')
#    else:
#        tau0 = 4*np.pi / ((freqs[-1]-freqs[1]))
#        params.add('tau', value=tau0, min=0, max=8/(freqs[-1]-freqs[1]))
#        middle_angle = -phase[len(freqs)/2] * np.pi / 180
#        middle_tau = tau0 * freqs[len(freqs)/2]
#        params.add('t0a', value=middle_angle - middle_tau, min=-np.pi, max=np.pi)
#        for s in ('f0', 'df', 't0'):
#            params[s].vary = False # Keep our initial values fixed for now
#
#        v = complex_voltage(s21, phase)
#
#        #plot_params(params, use_phase=True, title='Initial Phase')
#        minimize(complex_residual, params, args=(freqs, v), method='leastsq')
#        #plot_params(params, use_phase=True, title='Fitted Phase')
#
#        for s in ('f0', 'df', 't0'):
#            params[s].vary = True # do both now
#
#        minimize(complex_residual, params, args=(freqs, v), method='leastsq')
#        plot_params(params, use_phase=True, title='Fitted Both')
#
#
#
##        plt.figure(1)
##        plt.plot(freqs, s21, label='data')
##        f0, df, t0 = (out.params[s].value for s in ('f0', 'df', 't0'))
##        f02, df2, t02 = (init_params[s].value for s in ('f0', 'df', 't0'))
##        plt.plot(freqs, lorentzian(f02, df2, t02, freqs), label='initial guess')
##        plt.plot(freqs, lorentzian(f0, df, t0, freqs), label='fitted guess')
##
##        if phase is not None:
##            t0a, t0a2 = out.params['t0a'], init_params['t0a']
##            tau, tau2 = out.params['tau'], init_params['tau']
##            print tau2.value, tau.value
##            phase_init = get_angle(complex_lorentzian(f02, df2, t02, t0a2, tau2, freqs))
##            phase_final = get_angle(complex_lorentzian(f0, df, t0, t0a, tau, freqs))
##            plt.figure(2)
##            plt.plot(freqs, phase, label='data')
##            plt.plot(freqs, phase_init, label='initial guess')
##            plt.plot(freqs, phase_final, label='fitted guess')
##
##        plt.title('%s dBm' % power)
##        plt.legend(loc=2)
#
#    f0, df, t0 = [params[s].value for s in ('f0', 'df', 't0')]
#    values = f0 + offset, df, t0
#    plt.close()
#    if errors:
#        return values, [params[s].stderr for s in ('f0', 'df', 't0')]
#    else:
#        return values
#
#def gui_fit_lor(s21, freqs, errors=False, accept=False):
#    n = 1.0
#    f0 = FitParam('f0', freqs[len(freqs)/2], freqs[0]/n, n*freqs[-1])
#    df = FitParam('df', (freqs[-1] - freqs[0]) / 15, 0., freqs[-1]-freqs[0])
#    t0 = FitParam('t0', np.sqrt(max(s21)), 0., 2*np.sqrt(max(s21)))
#
#    def my_lor(f, params):
#        f0, df, t0 = params
#        return lorentzian(f0, df, t0, f)
#
#    params = [f0, df, t0]
#    values = guifit(freqs, s21, my_lor, params, xlabel="Frequency (Hz)", ylabel="S21", accept=accept)
#
#    if errors:
#        return [(p.value, (p.min, p.max)) for p in params]
#    else:
#        return [p.value for p in params]
#

def photon_number(power_dbm, q_int, t0, f0):
    hbar = 1.0546e-34
    s21 = np.sqrt(t0)
    omega = 2*np.pi*f0
    power_watts = from_dB(power_dbm)*1e-3
    return power_watts*q_int*s21*(1 - s21)/(hbar*omega**2)

#
#def photon_number_range(power_dbm, q_int, t0, f0):
#    q_int, q_int_err = q_int
#    t0, t0_err = t0
#    f0, f0_err = f0
#
#    f0_min, f0_max = f0 - f0_err, f0 + f0_err
#    q_int_min, q_int_max = q_int - q_int_err, q_int + q_int_err
#
#    n = photon_number(power_dbm, q_int, t0, f0)
#    n_min = photon_number(power_dbm, q_int_min, t0, f0_max)
#    n_max = photon_number(power_dbm, q_int_max, t0, f0_min)
#    n_err = max(n_max - n, n - n_min)
#    return n, n_err
#
#def get_qs(f0, df, t0):
#    q_tot = abs(f0 / (2*df)) # df is HWHM, divide by FWHM
#    q_couple = q_tot / t0
#    q_int = 1 / (1/q_tot - 1/q_couple)
#    return q_tot, q_couple, q_int
#
#def get_qs_range(f0, df, t0):
#    f0, f0_err = f0
#    df, df_err = df
#    t0, t0_err = t0
#    f0_min, f0_max = f0 - f0_err, f0 + f0_err
#    df_min, df_max = df - df_err, df + df_err
#    t0_min, t0_max = t0 - t0_err, t0 + t0_err
#    q_tot, q_couple, q_int = get_qs(f0, df, t0)
#    q_tot_min, q_couple_min, _ = get_qs(f0_min, df_max, t0_max)
#    q_tot_max, q_couple_max, _ = get_qs(f0_max, df_min, t0_min)
#    _, _, q_int_min = get_qs(f0_min, df_max, t0_min)
#    _, _, q_int_max = get_qs(f0_max, df_min, t0_max)
#    q_tot_err = max(q_tot_max - q_tot, q_tot - q_tot_min)
#    q_couple_err = max(q_couple_max - q_couple, q_couple - q_couple_min)
#    q_int_err = max(q_int_max - q_int, q_int - q_int_min)
#    return (q_tot, q_couple, q_int), (q_tot_err, q_couple_err, q_int_err)
#
#def fit_qs(freqs, s21_squared, phases=None, plot=True, gui=False, check=False):
#    if gui:
#        f0, df, t0 = gui_fit_lor(from_dB(s21_squared), freqs)
#    else:
#        f0, df, t0 = fit_lor(from_dB(s21_squared), freqs, phase=phases, plot=plot)
#    qt, qc, qi = get_qs(f0, df, t0)
#    print "f0:%.2e qt:%.2e qc:%.2e" % (f0, qt, qc)
#    if check and raw_input('keep? (y/n)') != 'y':
#        f0, df, t0 = fit_lor(from_dB(s21_squared), freqs, plot=plot)
#        qt, qc, qi = get_qs(f0, df, t0)
#    print f0, qt, qc
#    return qt, qc, qi

#def fit_hanger(s21, freqs, plot=True):
#    params = Parameters()
#
#    # Initial guesses
#    offset_t = np.sqrt(s21[0])
#    t_dip = np.sqrt(min(s21))
#    i0 = np.argmin(s21)
#    f0 = freqs[i0]
#    frange = freqs[-1] - freqs[0]
#    qt = f0 * 5 / frange
#    qc = qt / (1 - t_dip / offset_t)
#
#    params.add('f0', value=f0, min=freqs[0], max=freqs[-1])
#    params.add('qt', value=qt, min=qt / 10, max=10 * qt)
#    params.add('qc', value=qc, min=qc / 10, max=10 * qc)
#    params.add('offset_t', value=offset_t, min=offset_t / 5, max=2 * offset_t)
#
#    init_params = copy.deepcopy(params)
#
#    def apply_hanger(p, freqs):
#        _f0, _qt, _qc, _offset_t = (p[s].value for s in ('f0', 'qt', 'qc', 'offset_t'))
#        return hanger(_f0, _qt, _qc, _offset_t, freqs)
#
#    def residual(p, freqs, s21):
#        return s21 - apply_hanger(p, freqs)
#
#    minimize(residual, params, args=(freqs, s21))
#    #print 'init', [init_params[s].value for s in ('f0', 'qt', 'qc', 'offset_t')]
#    #print 'fitted', [params[s].value for s in ('f0', 'qt', 'qc', 'offset_t')]
#
#    if plot:
#        plt.figure(1)
#        plt.plot(freqs, s21, label='data')
#        plt.plot(freqs, apply_hanger(init_params, freqs), label='initial')
#        plt.plot(freqs, apply_hanger(params, freqs), label='fitted')
#        plt.legend()
#        plt.show()
#        plt.close()
#
#    return [params[s].value for s in ('f0', 'qt', 'qc', 'offset_t')]

