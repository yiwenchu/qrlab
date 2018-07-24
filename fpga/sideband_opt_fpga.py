# Optimize sideband modulation using spectrum analyzer called vspec

from lib.math.minimizer import Parameter, Minimizer
import time
import mclient

RFSRC = 'bncq'                                          # Qubit RF source
FREQ = mclient.instruments[RFSRC].get_frequency()+50e6  # Minimize power at this frequency
SPEC = mclient.instruments['vspec']
YNG = mclient.instruments['yngwie']
CHAN = 0                                                # FPGA ssb 'mode' (0 or 1)
THETA0 = YNG.get('ssbtheta%d'%CHAN)
RATIO0 = YNG.get('ssbratio%d'%CHAN)

def phase_func(params, delay=0.2):
#    yng.AnalogModes.load(0, 50000000, theta=float(params['theta'].value), ratio=params['ratio'].value)
    yng.set('ssbtheta%d'%CHAN, params['theta'].value)
    yng.set('ssbratio%d'%CHAN, params['ratio'].value)
    yng.update_modes()
    time.sleep(delay)
    val = SPEC.get_power()
    print 'Measuring at (theta=%.02f, ratio=%.02f): %.01f' % (params['theta'].value, params['ratio'].value, val)
    return val

SPEC.set_frequency(FREQ)
SPEC.set_rf_on(True)
time.sleep(1)

m = Minimizer(phase_func,
              n_it=4, n_eval=13, verbose=True)
m.add_parameter(Parameter('theta', value=THETA0, vrange=20))
m.add_parameter(Parameter('ratio', value=RATIO0, vrange=0.25))
m.minimize()

SPEC.set_rf_on(False)
SPEC.set_frequency(FREQ-1e9)
